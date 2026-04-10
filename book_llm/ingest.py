from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .chunking import Chunk, chunk_pages
from .pdf_reader import read_pdf_pages
from .vector_store import save_embeddings, write_chunks_jsonl


@dataclass(frozen=True)
class IngestResult:
    books: int
    chunks: int


def ingest_books(
    *,
    books_dir: Path,
    embeddings_path: Path,
    chunks_path: Path,
    embed_model_name: str,
    chunk_chars: int = 1200,
    overlap_chars: int = 200,
    only_files: list[str] | None = None,
) -> IngestResult:
    all_pdfs = sorted([p for p in books_dir.glob("*.pdf") if p.is_file()])
    if only_files:
        allow = set(only_files)
        pdfs = [p for p in all_pdfs if p.name in allow]
    else:
        pdfs = all_pdfs
    all_chunks: list[Chunk] = []

    for pdf in pdfs:
        pages = read_pdf_pages(pdf)
        page_tuples = [(p.page_number, p.text) for p in pages]
        all_chunks.extend(
            chunk_pages(
                source_file=pdf.name,
                pages=page_tuples,
                chunk_chars=chunk_chars,
                overlap_chars=overlap_chars,
            )
        )

    if not all_chunks:
        raise RuntimeError(f"No chunks created. Are there PDFs in {books_dir}?")

    embedder = SentenceTransformer(embed_model_name)
    texts = [c.text for c in all_chunks]
    embeddings = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    save_embeddings(embeddings_path, embeddings)
    write_chunks_jsonl(chunks_path, all_chunks)

    return IngestResult(books=len(pdfs), chunks=len(all_chunks))
