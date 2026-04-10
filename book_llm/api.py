from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from .config import SETTINGS
from .llm import LlmConfig, chat, load_llm
from .rag import format_context, load_embeddings_and_chunks, retrieve

app = FastAPI(title="BOOK LLM novv1")


class AskRequest(BaseModel):
    question: str
    book: str | None = None
    top_k: int = 6
    temperature: float = 0.2


class AskResponse(BaseModel):
    answer: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/books")
def books() -> dict[str, list[str]]:
    pdfs = sorted([p.name for p in Path(SETTINGS.books_dir).glob("*.pdf") if p.is_file()])
    return {"books": pdfs}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    embeddings_path = SETTINGS.data_dir / "embeddings.npy"
    chunks_path = SETTINGS.data_dir / "chunks.jsonl"
    try:
        embeddings, chunks = load_embeddings_and_chunks(
            embeddings_path=embeddings_path, chunks_path=chunks_path
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if not SETTINGS.model_path:
        raise HTTPException(
            status_code=400, detail="Missing BOOK_LLM_MODEL_PATH env var (GGUF model path)."
        )

    embedder = SentenceTransformer(SETTINGS.embed_model)
    retrieved = retrieve(
        query=req.question,
        embedder=embedder,
        embeddings=embeddings,
        chunks=chunks,
        top_k=req.top_k,
        book=req.book,
    )
    context = format_context(retrieved)

    cfg = LlmConfig(
        model_path=SETTINGS.model_path,
        n_ctx=SETTINGS.n_ctx,
        n_threads=SETTINGS.n_threads,
        n_gpu_layers=SETTINGS.n_gpu_layers,
        chat_format=os.getenv("BOOK_LLM_CHAT_FORMAT"),
    )
    llm = load_llm(cfg)
    system = (
        "You are BOOK LLM novv1. Use ONLY the provided context from books to answer.\n"
        "Write in clear, proper paragraphs. If unsure or missing info, say so.\n"
        "At the end, add a short 'Sources:' list with filename + page range you used."
    )
    user = f"QUESTION:\n{req.question}\n\nCONTEXT:\n{context}"
    answer = chat(llm, system=system, user=user, temperature=req.temperature)
    return AskResponse(answer=answer)
