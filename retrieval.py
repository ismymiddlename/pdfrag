"""Embeddings and the in-memory vector index."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np
from fastapi import HTTPException
from sentence_transformers import SentenceTransformer

from ingest import Chunk
from schemas import SearchHit

MODEL_ID = "intfloat/multilingual-e5-small"

# E5 models are trained with these instruction prefixes; dropping them degrades
# retrieval quality.
QUERY_PREFIX = "query: "
DOCUMENT_PREFIX = "passage: "

_model: SentenceTransformer | None = None


def load_model() -> None:
    global _model
    _model = SentenceTransformer(MODEL_ID)


def encode(texts: list[str], prefix: str) -> np.ndarray:
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


index = Index()
embed_lock = threading.Lock()


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
        query_embedding = encode([query], QUERY_PREFIX)[0]

    return index.search(query_embedding, top_k, document_id)


def to_search_hits(
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
