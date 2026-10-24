"""PDF RAG API: upload PDF documents and search them semantically."""

from __future__ import annotations

import io
import logging
import re
import threading
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from html import escape as escape_html
from math import ceil
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

MODEL_ID = "intfloat/multilingual-e5-small"
ASK_MODEL_ID = "LiquidAI/LFM2-1.2B-RAG"
ASK_MAX_NEW_TOKENS = 250
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_QUERY_CHARS = 2000
CHUNK_CHARS = 1000
CHUNK_OVERLAP_SENTENCES = 1
UI_DIR = Path(__file__).parent / "ui"

# E5 models are trained with these instruction prefixes; dropping them degrades
# retrieval quality.
QUERY_PREFIX = "query: "
DOCUMENT_PREFIX = "passage: "


@dataclass(frozen=True)
class Chunk:
    page: int
    index: int
    text: str


def _normalize_line(line: str) -> str:
    return re.sub(r"\d+", "#", " ".join(line.split()))


def _strip_running_headers(pages: list[str]) -> list[str]:
    """Drop short lines repeated on at least 80% of pages (running heads).

    Page numbers differ per page, so lines are compared with digits masked.
    The 80% threshold keeps lines that repeat only on a few pages, such as
    table rows.
    """
    if len(pages) < 4:
        return pages
    counts: Counter[str] = Counter()
    for page in pages:
        counts.update(
            {_normalize_line(line) for line in page.splitlines() if line.strip()}
        )
    threshold = max(3, ceil(0.8 * len(pages)))
    noise = {
        key
        for key, count in counts.items()
        if count >= threshold and len(key) <= 120
    }
    return [
        "\n".join(
            line
            for line in page.splitlines()
            if _normalize_line(line) not in noise
        )
        for page in pages
    ]


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_HAS_LETTER = re.compile(r"[^\W\d_]")


def _hard_split(text: str, size: int) -> list[str]:
    part_count = ceil(len(text) / size)
    part_size = ceil(len(text) / part_count)
    return [text[i : i + part_size] for i in range(0, len(text), part_size)]


def _chunk_page(text: str, page: int, next_index: int) -> list[Chunk]:
    text = " ".join(text.split())
    if not _HAS_LETTER.search(text):
        return []

    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_len = 0

    def flush() -> None:
        nonlocal buffer, buffer_len, next_index
        if not buffer:
            return
        chunks.append(Chunk(page=page, index=next_index, text=" ".join(buffer)))
        next_index += 1
        # Carry the trailing sentence(s) into the next chunk for context.
        buffer = buffer[-CHUNK_OVERLAP_SENTENCES:]
        buffer_len = len(" ".join(buffer))

    sentences = [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]
    for sentence in sentences:
        for piece in (
            _hard_split(sentence, CHUNK_CHARS)
            if len(sentence) > CHUNK_CHARS
            else [sentence]
        ):
            if buffer and buffer_len + len(piece) + 1 > CHUNK_CHARS:
                flush()
                if buffer_len + len(piece) + 1 > CHUNK_CHARS:
                    # The overlap sentence alone would overflow the next chunk.
                    buffer = []
                    buffer_len = 0
            buffer.append(piece)
            buffer_len = len(" ".join(buffer))

    if buffer:
        chunks.append(Chunk(page=page, index=next_index, text=" ".join(buffer)))
    return chunks


def extract_chunks(pdf_bytes: bytes) -> tuple[int, list[Chunk]]:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except PdfReadError as exc:
        raise HTTPException(status_code=422, detail=f"unreadable PDF: {exc}") from exc

    if reader.is_encrypted:
        raise HTTPException(status_code=422, detail="encrypted PDFs are not supported")

    try:
        pages = [page.extract_text() or "" for page in reader.pages]
    except PdfReadError as exc:
        raise HTTPException(status_code=422, detail=f"unreadable PDF: {exc}") from exc

    pages = _strip_running_headers(pages)

    chunks: list[Chunk] = []
    for page_number, page_text in enumerate(pages, start=1):
        chunks.extend(_chunk_page(page_text, page_number, len(chunks)))

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="no extractable text: the PDF may be scanned; OCR is not supported",
        )
    return len(reader.pages), chunks


_model: SentenceTransformer | None = None


def _encode(texts: list[str], prefix: str) -> np.ndarray:
    if _model is None:
        raise HTTPException(status_code=503, detail="embedding model not loaded")
    return _model.encode(
        [prefix + text for text in texts],
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


@dataclass
class IndexedDocument:
    document_id: str
    filename: str
    page_count: int
    chunks: list[Chunk]
    embeddings: np.ndarray


@dataclass
class Index:
    """In-memory corpus. Process-local; restarts clear it."""

    documents: dict[str, IndexedDocument] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, document: IndexedDocument) -> None:
        with self.lock:
            self.documents[document.document_id] = document

    def remove(self, document_id: str) -> bool:
        with self.lock:
            return self.documents.pop(document_id, None) is not None

    def snapshot(self) -> list[IndexedDocument]:
        with self.lock:
            return list(self.documents.values())

    def count(self) -> int:
        with self.lock:
            return len(self.documents)

    def contains(self, document_id: str) -> bool:
        with self.lock:
            return document_id in self.documents

    def search(
        self, query_embedding: np.ndarray, top_k: int, document_id: str | None
    ) -> list[tuple[IndexedDocument, Chunk, float]]:
        with self.lock:
            if document_id is not None:
                # The document can be removed between the caller's existence
                # check and this lookup; a missing one yields no hits.
                document = self.documents.get(document_id)
                documents = [document] if document is not None else []
            else:
                documents = list(self.documents.values())

        hits: list[tuple[IndexedDocument, Chunk, float]] = []
        for document in documents:
            scores = document.embeddings @ query_embedding
            order = np.argsort(-scores)[:top_k]
            for position in order:
                hits.append(
                    (
                        document,
                        document.chunks[int(position)],
                        float(scores[int(position)]),
                    )
                )
        hits.sort(key=lambda hit: hit[2], reverse=True)
        return [
            (document, chunk, round(score, 4))
            for document, chunk, score in hits[:top_k]
        ]


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    page_count: int
    chunk_count: int


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    document_id: str | None = None

    @field_validator("query")
    @classmethod
    def _query_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class SearchRequest(QueryRequest):
    top_k: int = Field(default=5, ge=1, le=20)


class SearchHit(BaseModel):
    document_id: str
    filename: str
    page: int
    chunk_index: int
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchHit]


class HealthResponse(BaseModel):
    status: str
    embedding_model: str
    ask_model: str
    document_count: int


class AskRequest(QueryRequest):
    top_k: int = Field(default=4, ge=1, le=10)


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list[SearchHit]


index = Index()
embed_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _model
    _model = SentenceTransformer(MODEL_ID)
    yield


app = FastAPI(title="PDF RAG API", version="1.0.0", lifespan=lifespan)


def _document_info(document: IndexedDocument) -> DocumentInfo:
    return DocumentInfo(
        document_id=document.document_id,
        filename=document.filename,
        page_count=document.page_count,
        chunk_count=len(document.chunks),
    )


@app.post("/documents", response_model=DocumentInfo, status_code=201)
def upload_document(file: UploadFile = File(...)) -> DocumentInfo:
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="only .pdf files are accepted")

    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"file exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        )
    if b"%PDF-" not in content[:1024]:
        raise HTTPException(status_code=415, detail="file is not a PDF")

    page_count, chunks = extract_chunks(content)
    with embed_lock:
        embeddings = _encode([chunk.text for chunk in chunks], DOCUMENT_PREFIX)

    document = IndexedDocument(
        document_id=uuid.uuid4().hex,
        filename=filename,
        page_count=page_count,
        chunks=chunks,
        embeddings=embeddings,
    )
    index.add(document)
    return _document_info(document)


def retrieve(
    query: str, top_k: int, document_id: str | None
) -> list[tuple[IndexedDocument, Chunk, float]]:
    if document_id is not None and not index.contains(document_id):
        raise HTTPException(
            status_code=404, detail=f"document {document_id!r} not found"
        )
    if index.count() == 0:
        raise HTTPException(status_code=409, detail="no documents uploaded yet")

    with embed_lock:
        query_embedding = _encode([query], QUERY_PREFIX)[0]

    return index.search(query_embedding, top_k, document_id)


def _to_search_hits(
    hits: list[tuple[IndexedDocument, Chunk, float]],
) -> list[SearchHit]:
    return [
        SearchHit(
            document_id=document.document_id,
            filename=document.filename,
            page=chunk.page,
            chunk_index=chunk.index,
            text=chunk.text,
            score=score,
        )
        for document, chunk, score in hits
    ]


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    hits = retrieve(request.query, request.top_k, request.document_id)
    return SearchResponse(query=request.query, results=_to_search_hits(hits))


@app.get("/documents", response_model=list[DocumentInfo])
def list_documents() -> list[DocumentInfo]:
    return [_document_info(document) for document in index.snapshot()]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        embedding_model=MODEL_ID,
        ask_model=ASK_MODEL_ID,
        document_count=index.count(),
    )


# Optional answer layer. It builds on retrieval; upload and search do not use it.

_llm: tuple[AutoTokenizer, AutoModelForCausalLM] | None = None
llm_lock = threading.Lock()


def _load_llm() -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(ASK_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(ASK_MODEL_ID, dtype=torch.bfloat16)
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return tokenizer, model.to(device).eval()


def _generate_answer(
    query: str, hits: list[tuple[IndexedDocument, Chunk, float]]
) -> str:
    global _llm
    context = "\n\n".join(f"[page {chunk.page}]\n{chunk.text}" for _, chunk, _ in hits)
    messages = [
        {
            "role": "system",
            "content": (
                "Answer using only the provided context. Answer in the language of "
                "the question. If the context does not contain the answer, say so. "
                "Mention the page numbers you used."
            ),
        },
        {
            "role": "user",
            "content": (
                "Use the following context to answer questions:\n"
                f"<context>\n{context}\n</context>\n\nQuestion: {query}"
            ),
        },
    ]
    with llm_lock:
        if _llm is None:
            _llm = _load_llm()
        tokenizer, model = _llm
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=ASK_MAX_NEW_TOKENS, do_sample=False
            )
        return tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        ).strip()


def answer_question(
    query: str, hits: list[tuple[IndexedDocument, Chunk, float]]
) -> str:
    try:
        return _generate_answer(query, hits)
    except Exception as exc:
        # Loading and generation fail for external reasons (download, memory,
        # device). Log the traceback and return a clear service error instead of
        # an opaque 500.
        logger.exception("answer generation failed")
        raise HTTPException(
            status_code=503,
            detail=f"answer generation failed: {type(exc).__name__}: {exc}",
        ) from exc


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    hits = retrieve(request.query, request.top_k, request.document_id)
    return AskResponse(
        query=request.query,
        answer=answer_question(request.query, hits),
        sources=_to_search_hits(hits),
    )


def _error_html(message: str) -> str:
    return f'<p class="error">{escape_html(message)}</p>'


def _reject_cross_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin is None:
        return
    if urlsplit(origin).netloc != request.headers.get("host"):
        raise HTTPException(status_code=403, detail="cross-origin request rejected")


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" + ("" if count == 1 else "s")


def _documents_html() -> str:
    documents = index.snapshot()
    if not documents:
        return '<p class="muted" data-count="0">No documents indexed.</p>'
    items = "".join(
        f"<li>"
        f'<div><span class="doc-name">{escape_html(document.filename)}</span>'
        f'<span class="muted">{_plural(document.page_count, "page")}, '
        f"{_plural(len(document.chunks), 'chunk')}</span></div>"
        f'<button type="button" class="remove" '
        f'hx-delete="/ui/documents/{escape_html(document.document_id)}" '
        f'hx-target="#documents" hx-swap="innerHTML" '
        f'hx-confirm="Remove {escape_html(document.filename)}?" '
        f'aria-label="Remove {escape_html(document.filename)}">Remove</button>'
        f"</li>"
        for document in documents
    )
    return f'<ul class="document-list" data-count="{len(documents)}">{items}</ul>'


def _hits_html(hits: list[tuple[IndexedDocument, Chunk, float]]) -> str:
    if not hits:
        return '<p class="muted">No results.</p>'
    show_filename = index.count() > 1
    items = []
    for document, chunk, score in hits:
        flat = " ".join(chunk.text.split())
        snippet = flat if len(flat) <= 110 else flat[:110].rsplit(" ", 1)[0] + "…"
        filename = (
            f'<p class="doc">{escape_html(document.filename)}</p>'
            if show_filename
            else ""
        )
        items.append(
            f'<details class="source">'
            f"<summary>p. {chunk.page} · {score:.3f} · "
            f"{escape_html(snippet)}</summary>"
            f"{filename}<p>{escape_html(chunk.text)}</p></details>"
        )
    return f'<div class="sources">{"".join(items)}</div>'


@app.get("/", include_in_schema=False)
def ui_index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/ui/documents", response_class=HTMLResponse)
def ui_documents() -> str:
    return _documents_html()


@app.delete("/ui/documents/{document_id}", response_class=HTMLResponse)
def ui_delete_document(request: Request, document_id: str) -> HTMLResponse:
    _reject_cross_origin(request)
    if not index.remove(document_id):
        return HTMLResponse(
            _error_html(f"document {document_id!r} not found"), status_code=404
        )
    # The last upload message no longer describes the index.
    return HTMLResponse(
        _documents_html() + '<div id="upload-status" hx-swap-oob="innerHTML"></div>'
    )


@app.post("/ui/upload", response_class=HTMLResponse)
def ui_upload(
    request: Request, files: list[UploadFile] = File(...)
) -> HTMLResponse:
    _reject_cross_origin(request)
    started = time.perf_counter()
    indexed: list[DocumentInfo] = []
    errors: list[str] = []
    for file in files:
        try:
            indexed.append(upload_document(file))
        except HTTPException as exc:
            errors.append(f"{file.filename or 'file'}: {exc.detail}")
        except Exception as exc:
            # One malformed file must not abort the batch or leave the browser
            # with a stale document list.
            logger.exception("upload failed for %s", file.filename)
            errors.append(
                f"{file.filename or 'file'}: {type(exc).__name__}: {exc}"
            )
    elapsed = time.perf_counter() - started

    lines: list[str] = []
    if indexed:
        pages = sum(info.page_count for info in indexed)
        chunks = sum(info.chunk_count for info in indexed)
        lines.append(
            f"Indexed {_plural(len(indexed), 'document')} "
            f"({_plural(pages, 'page')}, {_plural(chunks, 'chunk')}) "
            f"in {elapsed:.1f}s."
        )
    lines.extend(f'<span class="error">{escape_html(error)}</span>' for error in errors)
    status = " ".join(lines) if lines else _error_html("no files uploaded")
    if not indexed:
        return HTMLResponse(status, status_code=422)
    return HTMLResponse(
        f'{status}<div id="documents" hx-swap-oob="innerHTML">{_documents_html()}</div>'
    )


@app.post("/ui/query", response_class=HTMLResponse)
def ui_query(
    request: Request, query: str = Form(""), mode: str | None = Form(None)
) -> HTMLResponse:
    _reject_cross_origin(request)
    query = query.strip()
    if not query:
        return HTMLResponse(_error_html("query is required"), status_code=422)
    if len(query) > MAX_QUERY_CHARS:
        return HTMLResponse(
            _error_html(f"query exceeds {MAX_QUERY_CHARS} characters"),
            status_code=422,
        )
    if mode not in {"ask", "search"}:
        return HTMLResponse(
            _error_html("mode must be 'ask' or 'search'"), status_code=422
        )

    started = time.perf_counter()
    try:
        hits = retrieve(query, 4 if mode == "ask" else 5, None)
    except HTTPException as exc:
        return HTMLResponse(_error_html(str(exc.detail)), status_code=exc.status_code)

    if mode == "search":
        elapsed = time.perf_counter() - started
        return HTMLResponse(
            f'<p class="meta">{_plural(len(hits), "passage")} in {elapsed:.2f}s</p>'
            f"{_hits_html(hits)}"
        )

    try:
        answer = answer_question(query, hits)
    except HTTPException as exc:
        return HTMLResponse(_error_html(str(exc.detail)), status_code=exc.status_code)
    elapsed = time.perf_counter() - started
    return HTMLResponse(
        f'<div class="answer">{escape_html(answer)}</div>'
        f'<p class="meta">Answered in {elapsed:.1f}s from '
        f'{_plural(len(hits), "passage")}</p>'
        f"<h2>Sources</h2>{_hits_html(hits)}"
    )


app.mount("/static", StaticFiles(directory=UI_DIR), name="static")
