"""Request and response models for the JSON API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

MAX_QUERY_CHARS = 2000


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
    ask_model_loaded: bool
    document_count: int


class AskRequest(QueryRequest):
    top_k: int = Field(default=4, ge=1, le=10)


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list[SearchHit]
