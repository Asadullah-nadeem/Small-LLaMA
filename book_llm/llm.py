from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import requests


class ChatBackend(Protocol):
    def chat(self, *, system: str, user: str, temperature: float = 0.2) -> str: ...


@dataclass(frozen=True)
class OllamaConfig:
    url: str
    model: str


class OllamaBackend:
    def __init__(self, cfg: OllamaConfig):
        self._url = cfg.url.rstrip("/")
        self._model = cfg.model

    def chat(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        # Ollama chat API: https://github.com/ollama/ollama/blob/main/docs/api.md
        resp = requests.post(
            f"{self._url}/api/chat",
            json={
                "model": self._model,
                "stream": False,
                "options": {"temperature": temperature},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=600,
        )
        resp.raise_for_status()
        data = resp.json()
        msg = (data.get("message") or {}).get("content")
        if not isinstance(msg, str):
            raise RuntimeError("Unexpected response from Ollama /api/chat")
        return msg


@dataclass(frozen=True)
class LlamaCppConfig:
    model_path: str
    n_ctx: int = 4096
    n_threads: int = 0
    n_gpu_layers: int = 0
    chat_format: str | None = None


class LlamaCppBackend:
    def __init__(self, cfg: LlamaCppConfig):
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "llama-cpp-python is not installed. Install with: pip install -e .[llama_cpp]"
            ) from e

        self._llm = Llama(
            model_path=cfg.model_path,
            n_ctx=cfg.n_ctx,
            n_threads=cfg.n_threads if cfg.n_threads > 0 else None,
            n_gpu_layers=cfg.n_gpu_layers,
            chat_format=cfg.chat_format,
            verbose=False,
        )

    def chat(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        resp = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        return resp["choices"][0]["message"]["content"]

