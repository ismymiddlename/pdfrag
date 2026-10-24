"""Optional answer generation with a local language model."""

from __future__ import annotations

import logging
import threading

import torch
from fastapi import HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer

from ingest import Chunk
from retrieval import IndexedDocument

logger = logging.getLogger(__name__)

ASK_MODEL_ID = "LiquidAI/LFM2-1.2B-RAG"
ASK_MAX_NEW_TOKENS = 250

_llm: tuple[AutoTokenizer, AutoModelForCausalLM] | None = None
llm_load_lock = threading.Lock()
llm_generate_lock = threading.Lock()


def is_loaded() -> bool:
    # A status hint for the UI; no lock, so it cannot block a running load.
    return _llm is not None


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


def _get_llm() -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    global _llm
    with llm_load_lock:
        if _llm is None:
            _llm = _load_llm()
        return _llm


def warm_up() -> None:
    """Load the answer model ahead of the first question."""
    try:
        _get_llm()
    except Exception:
        # A failed background load must not stop the server; a later ask
        # retries and reports the error.
        logger.exception("answer model warm-up failed")


def _generate_answer(
    query: str, hits: list[tuple[IndexedDocument, Chunk, float]]
) -> str:
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
    tokenizer, model = _get_llm()
    with llm_generate_lock:
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
