$ErrorActionPreference = "Stop"

# 1) Activate venv
# .\.venv\Scripts\Activate.ps1

# 2) Load .env if present (recommended: copy .env.example -> .env)
if (Test-Path ".\.env") {
  Get-Content ".\.env" | ForEach-Object {
    $line = $_.Trim()
    if (-not $line) { return }
    if ($line.StartsWith("#")) { return }
    $kv = $line.Split("=", 2)
    if ($kv.Count -ne 2) { return }
    $key = $kv[0].Trim()
    $val = $kv[1].Trim().Trim('"').Trim("'")
    if ($key) { Set-Item -Path "Env:$key" -Value $val }
  }
}

# 3) Or set model path directly (override)
# $env:BOOK_LLM_MODEL_PATH = "C:\path\to\model.gguf"

# 4) Build index from Book/
book-llm ingest --books-dir .\Book

# 5) Ask a question (RAG)
book-llm ask "Summarize Anna Karenina in 5 paragraphs" --top-k 6

# 6) Summarize a specific book (or pages)
book-llm summarize --book "The+48+Laws+Of+Power.pdf" --pages "1-20"
