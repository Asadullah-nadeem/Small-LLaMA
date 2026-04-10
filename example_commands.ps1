$ErrorActionPreference = "Stop"

# 1) Activate venv
# .\.venv\Scripts\Activate.ps1

# 2) Point to your GGUF model
$env:BOOK_LLM_MODEL_PATH = "C:\path\to\model.gguf"

# 3) Build index from Book/
book-llm ingest --books-dir .\Book

# 4) Ask a question (RAG)
book-llm ask "Summarize Anna Karenina in 5 paragraphs" --top-k 6

# 5) Summarize a specific book (or pages)
book-llm summarize --book "The+48+Laws+Of+Power.pdf" --pages "1-20"

