from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except Exception:
    pass


@dataclass(frozen=True)
class Settings:
    books_dir: Path = Path("Book")
    data_dir: Path = Path("data")
    embeddings_path: Path = Path("data") / "embeddings.npy"
    chunks_path: Path = Path("data") / "chunks.jsonl"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    backend: str = os.getenv("BOOK_LLM_BACKEND", "ollama")  # ollama | llama_cpp

    model_path: str | None = os.getenv("BOOK_LLM_MODEL_PATH")
    n_ctx: int = int(os.getenv("BOOK_LLM_N_CTX", "4096"))
    n_threads: int = int(os.getenv("BOOK_LLM_N_THREADS", "0"))  # 0 -> llama.cpp default
    n_gpu_layers: int = int(os.getenv("BOOK_LLM_N_GPU_LAYERS", "0"))

    ollama_url: str = os.getenv("BOOK_LLM_OLLAMA_URL", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("BOOK_LLM_OLLAMA_MODEL", "llama3")


SETTINGS = Settings()
