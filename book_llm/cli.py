from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer

from .config import SETTINGS
from .ingest import ingest_books
from .llm import LlmConfig, chat, load_llm
from .pdf_reader import read_pdf_pages
from .rag import format_context, load_embeddings_and_chunks, retrieve

app = typer.Typer(add_completion=False, help="BOOK LLM novv1 - local book Q&A and summarizer")
console = Console()


def _model_path_or_fail(model_path: str | None) -> str:
    model_path = model_path or SETTINGS.model_path
    if not model_path:
        raise typer.BadParameter(
            "Missing model path. Set BOOK_LLM_MODEL_PATH to your .gguf file."
        )
    return model_path


def _parse_page_range(s: str) -> tuple[int, int]:
    s = s.strip()
    if "-" not in s:
        n = int(s)
        return n, n
    a, b = s.split("-", 1)
    start = int(a.strip())
    end = int(b.strip())
    if start <= 0 or end <= 0 or end < start:
        raise ValueError("Invalid page range")
    return start, end


@app.command()
def list(books_dir: Path = typer.Option(SETTINGS.books_dir, "--books-dir")) -> None:
    """List PDFs found in your books folder."""
    pdfs = sorted([p for p in books_dir.glob("*.pdf") if p.is_file()])
    t = Table(title=f"Books in {books_dir}")
    t.add_column("File")
    t.add_column("Size (MB)", justify="right")
    for p in pdfs:
        t.add_row(p.name, f"{p.stat().st_size / (1024 * 1024):.1f}")
    console.print(t)


@app.command()
def ingest(
    books_dir: Path = typer.Option(SETTINGS.books_dir, "--books-dir"),
    data_dir: Path = typer.Option(SETTINGS.data_dir, "--data-dir"),
    embed_model: str = typer.Option(SETTINGS.embed_model, "--embed-model"),
    chunk_chars: int = typer.Option(1200, "--chunk-chars"),
    overlap_chars: int = typer.Option(200, "--overlap-chars"),
) -> None:
    """Build / rebuild the local vector index from PDFs."""
    embeddings_path = data_dir / "embeddings.npy"
    chunks_path = data_dir / "chunks.jsonl"
    res = ingest_books(
        books_dir=books_dir,
        embeddings_path=embeddings_path,
        chunks_path=chunks_path,
        embed_model_name=embed_model,
        chunk_chars=chunk_chars,
        overlap_chars=overlap_chars,
    )
    console.print(f"[green]Ingested[/green] {res.books} PDFs -> {res.chunks} chunks")
    console.print(f"Wrote {embeddings_path} and {chunks_path}")


@app.command()
def ask(
    question: str = typer.Argument(...),
    data_dir: Path = typer.Option(SETTINGS.data_dir, "--data-dir"),
    embed_model: str = typer.Option(SETTINGS.embed_model, "--embed-model"),
    model_path: str | None = typer.Option(None, "--model-path"),
    book: str | None = typer.Option(None, "--book", help="Exact PDF filename to focus on"),
    top_k: int = typer.Option(6, "--top-k"),
    temperature: float = typer.Option(0.2, "--temp"),
) -> None:
    """Ask a question. Answers in clean paragraphs, using the book context."""
    embeddings_path = data_dir / "embeddings.npy"
    chunks_path = data_dir / "chunks.jsonl"
    embeddings, chunks = load_embeddings_and_chunks(
        embeddings_path=embeddings_path, chunks_path=chunks_path
    )

    embedder = SentenceTransformer(embed_model)
    retrieved = retrieve(
        query=question,
        embedder=embedder,
        embeddings=embeddings,
        chunks=chunks,
        top_k=top_k,
        book=book,
    )
    context = format_context(retrieved)

    cfg = LlmConfig(
        model_path=_model_path_or_fail(model_path),
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
    user = f"QUESTION:\n{question}\n\nCONTEXT:\n{context}"
    answer = chat(llm, system=system, user=user, temperature=temperature)
    console.print(answer)


@app.command()
def summarize(
    book: str = typer.Option(..., "--book", help="Exact PDF filename"),
    pages: str | None = typer.Option(None, "--pages", help='Page range like "1-10"'),
    books_dir: Path = typer.Option(SETTINGS.books_dir, "--books-dir"),
    model_path: str | None = typer.Option(None, "--model-path"),
    temperature: float = typer.Option(0.2, "--temp"),
) -> None:
    """Summarize a book (or a page range) into clean paragraphs."""
    pdf_path = books_dir / book
    if not pdf_path.exists():
        raise typer.BadParameter(f"Book not found: {pdf_path}")

    start, end = (1, 10**9) if pages is None else _parse_page_range(pages)
    selected = []
    for p in read_pdf_pages(pdf_path):
        if p.page_number < start:
            continue
        if p.page_number > end:
            break
        if p.text.strip():
            selected.append((p.page_number, p.text))

    if not selected:
        raise typer.BadParameter("No text found in that page range (maybe scanned PDF).")

    # Chunk the selected pages into manageable pieces for the LLM.
    pieces: list[str] = []
    current = ""
    for page_number, text in selected:
        block = f"[p.{page_number}]\n{text}"
        if len(current) + len(block) + 2 > 6000 and current:
            pieces.append(current)
            current = ""
        current = (current + "\n\n" + block).strip()
    if current:
        pieces.append(current)

    cfg = LlmConfig(
        model_path=_model_path_or_fail(model_path),
        n_ctx=SETTINGS.n_ctx,
        n_threads=SETTINGS.n_threads,
        n_gpu_layers=SETTINGS.n_gpu_layers,
        chat_format=os.getenv("BOOK_LLM_CHAT_FORMAT"),
    )
    llm = load_llm(cfg)

    system = (
        "You are BOOK LLM novv1. Summarize the provided book text.\n"
        "Write in clear, proper paragraphs. Do not invent facts not in the text.\n"
        "If the text is incomplete, say so."
    )

    partials: list[str] = []
    for i, piece in enumerate(pieces, start=1):
        user = (
            f"Summarize PART {i}/{len(pieces)} into 1-2 paragraphs.\n\nTEXT:\n{piece}"
        )
        partials.append(chat(llm, system=system, user=user, temperature=temperature).strip())

    if len(partials) == 1:
        console.print(partials[0])
        return

    user_final = (
        "Combine these partial summaries into a single well-structured summary (5-10 paragraphs). "
        "Keep it readable and coherent.\n\nPARTIAL SUMMARIES:\n"
        + "\n\n---\n\n".join(partials)
    )
    console.print(chat(llm, system=system, user=user_final, temperature=temperature))


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """Run the FastAPI server."""
    import uvicorn

    uvicorn.run("book_llm.api:app", host=host, port=port, reload=False)
