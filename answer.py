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
