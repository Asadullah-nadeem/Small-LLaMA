from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .chunking import Chunk


def write_chunks_jsonl(chunks_path: Path, chunks: list[Chunk]) -> None:
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with chunks_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def read_chunks_jsonl(chunks_path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            chunks.append(
                Chunk(
                    chunk_id=obj["chunk_id"],
                    source_file=obj["source_file"],
                    page_start=int(obj["page_start"]),
                    page_end=int(obj["page_end"]),
                    text=obj["text"],
                )
            )
    return chunks


def save_embeddings(embeddings_path: Path, embeddings: np.ndarray) -> None:
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    embs = np.asarray(embeddings, dtype=np.float32)
    np.save(str(embeddings_path), embs)


def load_embeddings(embeddings_path: Path) -> np.ndarray:
    return np.load(str(embeddings_path), mmap_mode="r")


def cosine_top_k(
    *,
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
    top_k: int,
    allow_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (scores, indices) for cosine similarity.
    Assumes embeddings and query_embedding are already L2-normalized.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D")
    if query_embedding.ndim != 1:
        raise ValueError("query_embedding must be 1D")
    if embeddings.shape[1] != query_embedding.shape[0]:
        raise ValueError("embedding dims do not match")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    scores = embeddings @ query_embedding
    if allow_mask is not None:
        scores = np.where(allow_mask, scores, -1e9)

    k = min(top_k, embeddings.shape[0])
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return scores[idx], idx
