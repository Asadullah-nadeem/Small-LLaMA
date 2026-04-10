# BOOK LLM novv1

Local “small LLaMA” style book reader: it ingests PDFs from `Book/`, builds a vector index, then answers questions and writes clean paragraphs using retrieved context (RAG).

## 1) Install

```powershell
cd C:\Users\Aadrika\Documents\BookLLM
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .
```

## 2) Put a GGUF model

Download any **GGUF** instruct model (small works best, e.g. 2B–8B) and set:

```powershell
$env:BOOK_LLM_MODEL_PATH="C:\path\to\model.gguf"
```

Optional:

```powershell
$env:BOOK_LLM_N_CTX="4096"
$env:BOOK_LLM_N_GPU_LAYERS="0"
```

## 3) Ingest books

```powershell
book-llm ingest --books-dir .\Book
```

This creates `.\data\embeddings.npy` and `.\data\chunks.jsonl`.

## 4) Ask questions

```powershell
book-llm ask "Summarize Anna Karenina in 5 paragraphs" --top-k 6
book-llm ask "Explain Law 1 from The 48 Laws of Power" --book "The+48+Laws+Of+Power.pdf"
```

## 4b) Summarize a book (or pages)

```powershell
book-llm summarize --book "The+48+Laws+Of+Power.pdf" --pages "1-20"
```

## 5) Run as an API (optional)

```powershell
book-llm serve --host 127.0.0.1 --port 8000
```

Then:

```powershell
curl http://127.0.0.1:8000/health
```

## Notes

- This project does **not** train a new model. It uses a small local model + retrieval so it can “read” your PDFs.
- If a PDF is scanned images (no text), you’ll need OCR first.
