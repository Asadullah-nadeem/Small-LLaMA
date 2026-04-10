from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .chunking import Chunk
from .vector_store import cosine_top_k, load_embeddings, read_chunks_jsonl


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


def load_embeddings_and_chunks(
    *, embeddings_path: Path, chunks_path: Path
) -> tuple[np.ndarray, list[Chunk]]:
    if not embeddings_path.exists() or not chunks_path.exists():
        raise FileNotFoundError("Index not found. Run: book-llm ingest")
    return load_embeddings(embeddings_path), read_chunks_jsonl(chunks_path)


def retrieve(
    *,
    query: str,
    embedder: SentenceTransformer,
    embeddings: np.ndarray,
    chunks: list[Chunk],
    top_k: int = 6,
    book: str | None = None,
) -> list[RetrievedChunk]:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    q = embedder.encode([query], normalize_embeddings=True)
    q = np.asarray(q, dtype=np.float32).reshape(-1)

    allow_mask = None
    if book:
        allow_mask = np.fromiter((c.source_file == book for c in chunks), dtype=bool, count=len(chunks))

    scores, ids = cosine_top_k(
        embeddings=embeddings,
        query_embedding=q,
        top_k=min(top_k * 3, len(chunks)),
        allow_mask=allow_mask,
    )
    scored: list[RetrievedChunk] = []
    for score, idx in zip(scores.tolist(), ids.tolist(), strict=False):
        if idx < 0:
            continue
        ch = chunks[idx]
        scored.append(RetrievedChunk(chunk=ch, score=float(score)))
        if len(scored) >= top_k:
            break
    return scored


def format_context(retrieved: list[RetrievedChunk], *, max_chars: int = 12000) -> str:
    parts: list[str] = []
    used = 0
    for r in retrieved:
        header = f"[Source: {r.chunk.source_file} p.{r.chunk.page_start}-{r.chunk.page_end} | score={r.score:.3f}]"
        body = r.chunk.text.strip()
        piece = header + "\n" + body
        if used + len(piece) + 2 > max_chars:
            break
        parts.append(piece)
        used += len(piece) + 2
    return "\n\n---\n\n".join(parts).strip()
