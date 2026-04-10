from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class PdfPage:
    page_number: int  # 1-based
    text: str


def read_pdf_pages(pdf_path: Path) -> list[PdfPage]:
    reader = PdfReader(str(pdf_path))
    pages: list[PdfPage] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = _cleanup_text(text)
        pages.append(PdfPage(page_number=i + 1, text=text))
    return pages


def _cleanup_text(text: str) -> str:
    text = text.replace("\u0000", "")
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines).strip()

