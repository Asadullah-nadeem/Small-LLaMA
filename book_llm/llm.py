from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from llama_cpp import Llama  # type: ignore


@dataclass(frozen=True)
class LlmConfig:
    model_path: str
    n_ctx: int = 4096
    n_threads: int = 0
    n_gpu_layers: int = 0
    chat_format: str | None = None


def load_llm(cfg: LlmConfig) -> Llama:
    if not cfg.model_path:
        raise ValueError("model_path is required")
    if not os.path.exists(cfg.model_path):
        raise FileNotFoundError(f"Model not found: {cfg.model_path}")
    return Llama(
        model_path=cfg.model_path,
        n_ctx=cfg.n_ctx,
        n_threads=cfg.n_threads if cfg.n_threads > 0 else None,
        n_gpu_layers=cfg.n_gpu_layers,
        chat_format=cfg.chat_format,
        verbose=False,
    )


def chat(llm: Llama, *, system: str, user: str, temperature: float = 0.2) -> str:
    resp: dict[str, Any] = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return resp["choices"][0]["message"]["content"]

