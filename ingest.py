"""PDF text extraction and chunking."""

from __future__ import annotations

import io
import re
from collections import Counter
from dataclasses import dataclass
from math import ceil

from fastapi import HTTPException
from pypdf import PdfReader
from pypdf.errors import PdfReadError

CHUNK_CHARS = 1000
CHUNK_OVERLAP_SENTENCES = 1


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
        key for key, count in counts.items() if count >= threshold and len(key) <= 120
    }
    return [
        "\n".join(
            line for line in page.splitlines() if _normalize_line(line) not in noise
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
