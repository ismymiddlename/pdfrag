"""HTML fragments and request guards for the browser UI."""

from __future__ import annotations

from html import escape as escape_html
from urllib.parse import urlsplit

from fastapi import HTTPException, Request

from ingest import Chunk
from retrieval import IndexedDocument, index


def error_html(message: str) -> str:
    return f'<p class="error">{escape_html(message)}</p>'


def reject_cross_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin is None:
        return
    if urlsplit(origin).netloc != request.headers.get("host"):
        raise HTTPException(status_code=403, detail="cross-origin request rejected")


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}" + ("" if count == 1 else "s")


def documents_html() -> str:
    documents = index.snapshot()
    if not documents:
        return '<p class="muted" data-count="0">No documents indexed.</p>'
    items = "".join(
        f"<li>"
        f'<div><span class="doc-name">{escape_html(document.filename)}</span>'
        f'<span class="muted">{plural(document.page_count, "page")}, '
        f"{plural(len(document.chunks), 'chunk')}</span></div>"
        f'<button type="button" class="remove" '
        f'hx-delete="/ui/documents/{escape_html(document.document_id)}" '
        f'hx-target="#documents" hx-swap="innerHTML" '
        f'hx-confirm="Remove {escape_html(document.filename)}?" '
        f'aria-label="Remove {escape_html(document.filename)}">Remove</button>'
        f"</li>"
        for document in documents
    )
    return f'<ul class="document-list" data-count="{len(documents)}">{items}</ul>'


def hits_html(hits: list[tuple[IndexedDocument, Chunk, float]]) -> str:
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
