"""PDF RAG API: upload PDF documents and search them semantically."""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from html import escape as escape_html
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from answer import ASK_MODEL_ID, answer_question, is_loaded, warm_up
from ingest import extract_chunks
from retrieval import (
    DOCUMENT_PREFIX,
    MODEL_ID,
    IndexedDocument,
    embed_lock,
    encode,
    index,
    load_model,
    retrieve,
    to_search_hits,
)
from schemas import (
    MAX_QUERY_CHARS,
    AskRequest,
    AskResponse,
    DocumentInfo,
    HealthResponse,
    SearchRequest,
    SearchResponse,
)
from ui import (
    documents_html,
    error_html,
    hits_html,
    plural,
    reject_cross_origin,
)

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
UI_DIR = Path(__file__).parent / "ui"


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    if os.environ.get("PRELOAD_ANSWER_MODEL", "1") != "0":
        threading.Thread(
            target=warm_up, name="answer-model-warmup", daemon=True
        ).start()
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
        embeddings = encode([chunk.text for chunk in chunks], DOCUMENT_PREFIX)

    document = IndexedDocument(
        document_id=uuid.uuid4().hex,
        filename=filename,
        page_count=page_count,
        chunks=chunks,
        embeddings=embeddings,
    )
    index.add(document)
    return _document_info(document)


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    hits = retrieve(request.query, request.top_k, request.document_id)
    return SearchResponse(query=request.query, results=to_search_hits(hits))


@app.get("/documents", response_model=list[DocumentInfo])
def list_documents() -> list[DocumentInfo]:
    return [_document_info(document) for document in index.snapshot()]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        embedding_model=MODEL_ID,
        ask_model=ASK_MODEL_ID,
        ask_model_loaded=is_loaded(),
        document_count=index.count(),
    )


# Optional answer layer. It builds on retrieval; upload and search do not use it.


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    hits = retrieve(request.query, request.top_k, request.document_id)
    return AskResponse(
        query=request.query,
        answer=answer_question(request.query, hits),
        sources=to_search_hits(hits),
    )


# Browser UI, an optional layer over the same retrieval functions.


@app.get("/", include_in_schema=False)
def ui_index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/ui/documents", response_class=HTMLResponse)
def ui_documents() -> str:
    return documents_html()


@app.delete("/ui/documents/{document_id}", response_class=HTMLResponse)
def ui_delete_document(request: Request, document_id: str) -> HTMLResponse:
    reject_cross_origin(request)
    if not index.remove(document_id):
        return HTMLResponse(
            error_html(f"document {document_id!r} not found"), status_code=404
        )
    # The last upload message no longer describes the index.
    return HTMLResponse(
        documents_html() + '<div id="upload-status" hx-swap-oob="innerHTML"></div>'
    )


@app.post("/ui/upload", response_class=HTMLResponse)
def ui_upload(request: Request, files: list[UploadFile] = File(...)) -> HTMLResponse:
    reject_cross_origin(request)
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
            errors.append(f"{file.filename or 'file'}: {type(exc).__name__}: {exc}")
    elapsed = time.perf_counter() - started

    lines: list[str] = []
    if indexed:
        pages = sum(info.page_count for info in indexed)
        chunks = sum(info.chunk_count for info in indexed)
        lines.append(
            f"Indexed {plural(len(indexed), 'document')} "
            f"({plural(pages, 'page')}, {plural(chunks, 'chunk')}) "
            f"in {elapsed:.1f}s."
        )
    lines.extend(f'<span class="error">{escape_html(error)}</span>' for error in errors)
    status = " ".join(lines) if lines else error_html("no files uploaded")
    if not indexed:
        return HTMLResponse(status, status_code=422)
    return HTMLResponse(
        f'{status}<div id="documents" hx-swap-oob="innerHTML">{documents_html()}</div>'
    )


@app.post("/ui/query", response_class=HTMLResponse)
def ui_query(
    request: Request, query: str = Form(""), mode: str | None = Form(None)
) -> HTMLResponse:
    reject_cross_origin(request)
    query = query.strip()
    if not query:
        return HTMLResponse(error_html("query is required"), status_code=422)
    if len(query) > MAX_QUERY_CHARS:
        return HTMLResponse(
            error_html(f"query exceeds {MAX_QUERY_CHARS} characters"),
            status_code=422,
        )
    if mode not in {"ask", "search"}:
        return HTMLResponse(
            error_html("mode must be 'ask' or 'search'"), status_code=422
        )

    started = time.perf_counter()
    try:
        hits = retrieve(query, 4 if mode == "ask" else 5, None)
    except HTTPException as exc:
        return HTMLResponse(error_html(str(exc.detail)), status_code=exc.status_code)

    if mode == "search":
        elapsed = time.perf_counter() - started
        return HTMLResponse(
            f'<p class="meta">{plural(len(hits), "passage")} in {elapsed:.2f}s</p>'
            f"{hits_html(hits)}"
        )

    try:
        answer = answer_question(query, hits)
    except HTTPException as exc:
        return HTMLResponse(error_html(str(exc.detail)), status_code=exc.status_code)
    elapsed = time.perf_counter() - started
    return HTMLResponse(
        f'<div class="answer">{escape_html(answer)}</div>'
        f'<p class="meta">Answered in {elapsed:.1f}s from '
        f"{plural(len(hits), 'passage')}</p>"
        f"<h2>Sources</h2>{hits_html(hits)}"
    )


app.mount("/static", StaticFiles(directory=UI_DIR), name="static")
