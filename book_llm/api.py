from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from .config import SETTINGS
from .ingest import ingest_books
from .llm import LlamaCppBackend, LlamaCppConfig, OllamaBackend, OllamaConfig
from .rag import format_context, load_embeddings_and_chunks, retrieve

app = FastAPI(title="BOOK LLM novv1")


class AskRequest(BaseModel):
    question: str
    book: str | None = None
    top_k: int = 6
    temperature: float = 0.2


class AskResponse(BaseModel):
    answer: str


def _load_backend() -> OllamaBackend | LlamaCppBackend:
    backend = (os.getenv("BOOK_LLM_BACKEND") or SETTINGS.backend).strip().lower()
    if backend == "ollama":
        return OllamaBackend(OllamaConfig(url=SETTINGS.ollama_url, model=SETTINGS.ollama_model))
    if backend == "llama_cpp":
        if not SETTINGS.model_path:
            raise RuntimeError("Missing BOOK_LLM_MODEL_PATH env var (GGUF model path).")
        return LlamaCppBackend(
            LlamaCppConfig(
                model_path=SETTINGS.model_path,
                n_ctx=SETTINGS.n_ctx,
                n_threads=SETTINGS.n_threads,
                n_gpu_layers=SETTINGS.n_gpu_layers,
                chat_format=os.getenv("BOOK_LLM_CHAT_FORMAT"),
            )
        )
    raise RuntimeError("BOOK_LLM_BACKEND must be 'ollama' or 'llama_cpp'")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>BOOK LLM novv1</title>
    <style>
      body { font-family: system-ui, Segoe UI, Arial; max-width: 900px; margin: 24px auto; padding: 0 16px; }
      h1 { margin: 0 0 12px; }
      .card { border: 1px solid #ddd; border-radius: 10px; padding: 14px; margin: 12px 0; }
      label { display:block; margin: 8px 0 4px; font-weight: 600; }
      input, textarea, select { width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #ccc; }
      button { margin-top: 10px; padding: 10px 14px; border-radius: 8px; border: 1px solid #333; background: #111; color: #fff; cursor: pointer; }
      pre { white-space: pre-wrap; background: #f6f6f6; padding: 12px; border-radius: 8px; border: 1px solid #eee; }
      .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
      @media (max-width: 720px) { .row { grid-template-columns: 1fr; } }
      .muted { color: #666; font-size: 0.95rem; }
    </style>
  </head>
  <body>
    <h1>BOOK LLM novv1</h1>
    <div class="muted">Upload PDF → Ingest → Ask questions (uses local GGUF model).</div>

    <div class="card">
      <h3>1) Upload PDF</h3>
      <form id="uploadForm">
        <input type="file" id="pdfFile" accept="application/pdf" required />
        <button type="submit">Upload</button>
      </form>
      <pre id="uploadOut"></pre>
    </div>

    <div class="card">
      <h3>2) Ingest</h3>
      <div class="row">
        <div>
          <label>Book (optional exact filename)</label>
          <select id="bookSelect"><option value="">(All books)</option></select>
        </div>
        <div>
          <label>&nbsp;</label>
          <button id="refreshBooks" type="button">Refresh Book List</button>
        </div>
      </div>
      <button id="ingestBtn" type="button">Ingest Now</button>
      <pre id="ingestOut"></pre>
    </div>

    <div class="card">
      <h3>3) Ask</h3>
      <label>Question</label>
      <textarea id="question" rows="4" placeholder="Ask something from the book(s)..."></textarea>
      <div class="row">
        <div>
          <label>Top-K</label>
          <input id="topk" type="number" min="1" value="6" />
        </div>
        <div>
          <label>Temperature</label>
          <input id="temp" type="number" min="0" max="2" step="0.1" value="0.2" />
        </div>
      </div>
      <label>Book (optional exact filename)</label>
      <select id="askBook"><option value="">(All books)</option></select>
      <button id="askBtn" type="button">Ask</button>
      <pre id="askOut"></pre>
    </div>

    <script>
      async function refreshBooks() {
        const res = await fetch("/books");
        const data = await res.json();
        const list = data.books || [];
        const selects = [document.getElementById("bookSelect"), document.getElementById("askBook")];
        for (const sel of selects) {
          const keepFirst = sel.options[0];
          sel.innerHTML = "";
          sel.appendChild(keepFirst);
          for (const b of list) {
            const opt = document.createElement("option");
            opt.value = b;
            opt.textContent = b;
            sel.appendChild(opt);
          }
        }
      }

      document.getElementById("refreshBooks").addEventListener("click", refreshBooks);
      refreshBooks();

      document.getElementById("uploadForm").addEventListener("submit", async (e) => {
        e.preventDefault();
        const f = document.getElementById("pdfFile").files[0];
        if (!f) return;
        const fd = new FormData();
        fd.append("file", f);
        const res = await fetch("/upload", { method: "POST", body: fd });
        document.getElementById("uploadOut").textContent = await res.text();
        await refreshBooks();
      });

      document.getElementById("ingestBtn").addEventListener("click", async () => {
        const book = document.getElementById("bookSelect").value;
        const url = book ? ("/ingest?book=" + encodeURIComponent(book)) : "/ingest";
        const res = await fetch(url, { method: "POST" });
        document.getElementById("ingestOut").textContent = await res.text();
      });

      document.getElementById("askBtn").addEventListener("click", async () => {
        const payload = {
          question: document.getElementById("question").value,
          book: document.getElementById("askBook").value || null,
          top_k: parseInt(document.getElementById("topk").value || "6", 10),
          temperature: parseFloat(document.getElementById("temp").value || "0.2")
        };
        const res = await fetch("/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        document.getElementById("askOut").textContent = await res.text();
      });
    </script>
  </body>
</html>
""".strip()


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, str]:
    """
    Upload a PDF into the books folder (Book/).
    """
    filename = Path(file.filename or "").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")
    SETTINGS.books_dir.mkdir(parents=True, exist_ok=True)
    dest = SETTINGS.books_dir / filename
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty upload.")
    dest.write_bytes(content)
    return {"saved_as": filename}


@app.post("/ingest")
def ingest(book: str | None = None) -> dict[str, int | str]:
    """
    Build / rebuild embeddings from PDFs. Optional: pass `book` (exact filename) to ingest only one PDF.
    """
    embeddings_path = SETTINGS.data_dir / "embeddings.npy"
    chunks_path = SETTINGS.data_dir / "chunks.jsonl"
    try:
        res = ingest_books(
            books_dir=SETTINGS.books_dir,
            embeddings_path=embeddings_path,
            chunks_path=chunks_path,
            embed_model_name=SETTINGS.embed_model,
            only_files=[book] if book else None,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"status": "ok", "books": res.books, "chunks": res.chunks}


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

    try:
        llm = _load_backend()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    system = (
        "You are BOOK LLM novv1. Use ONLY the provided context from books to answer.\n"
        "Write in clear, proper paragraphs. If unsure or missing info, say so.\n"
        "At the end, add a short 'Sources:' list with filename + page range you used."
    )
    user = f"QUESTION:\n{req.question}\n\nCONTEXT:\n{context}"
    answer = llm.chat(system=system, user=user, temperature=req.temperature)
    return AskResponse(answer=answer)
