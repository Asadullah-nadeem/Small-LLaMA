from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_file: str
    page_start: int
    page_end: int
    text: str


def chunk_pages(
    *,
    source_file: str,
    pages: list[tuple[int, str]],
    chunk_chars: int = 1200,
    overlap_chars: int = 200,
) -> list[Chunk]:
    """
    Chunk a list of (page_number, text) into overlapping chunks by character count.
    Keeps a page range per chunk so we can cite sources like: filename p.12-13.
    """
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be > 0")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")
    if overlap_chars >= chunk_chars:
        raise ValueError("overlap_chars must be < chunk_chars")

    out: list[Chunk] = []
    buffer = ""
    page_start = None
    page_end = None
    idx = 0

    def flush() -> None:
        nonlocal buffer, page_start, page_end, idx
        if not buffer.strip():
            buffer = ""
            page_start = None
            page_end = None
            return
        assert page_start is not None and page_end is not None
        chunk_id = f"{source_file}:{idx}"
        out.append(
            Chunk(
                chunk_id=chunk_id,
                source_file=source_file,
                page_start=page_start,
                page_end=page_end,
                text=buffer.strip(),
            )
        )
        idx += 1
        if overlap_chars > 0 and len(buffer) > overlap_chars:
            buffer = buffer[-overlap_chars:]
        else:
            buffer = ""
        page_start = page_end

    for page_number, page_text in pages:
        if not page_text.strip():
            continue
        if page_start is None:
            page_start = page_number
        page_end = page_number

        if buffer:
            buffer += "\n\n"
        buffer += page_text

        while len(buffer) >= chunk_chars:
            flush()

    if buffer.strip():
        flush()

    return out

