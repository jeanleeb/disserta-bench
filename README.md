# DISSERTA-BENCH

Dataset of open-ended questions from Brazilian university entrance exams (FUVEST, ITA, Comvest), with official solutions and evaluation rubrics.

## Current Status

Dataset under construction.

## Setup

```bash
uv sync
```

### OCR via Ollama (required for extraction)

The extraction pipeline uses [Ollama](https://ollama.com/) with a vision model for PDF OCR. Install Ollama and pull the default OCR model:

```bash
ollama pull glm-ocr
```

## Scripts

### `scripts/download_fuvest.py`

Downloads FUVEST second-phase (dissertative) exam and answer key PDFs from official FUVEST sites (`acervo.fuvest.br` and `fuvest.br`). Uses only the Python standard library.

```bash
# Default years (1997-2025)
uv run scripts/download_fuvest.py

# Specific years
uv run scripts/download_fuvest.py --years 2020 2021 2022

# Year range, forcing re-download
uv run scripts/download_fuvest.py --years 1997-2025 --force
```

All distinct PDFs found under "Segunda Fase" on the archive page are downloaded with standardized filenames (`dia1.pdf`, `dia2.pdf`, `fisica.pdf`, `guia_respostas.pdf`, etc.). This handles per-subject PDFs (older years), multi-day exams, and answer keys automatically. Specific-skills exams (arts, music) are excluded.

PDFs are saved to:

- `raw_pdfs/fuvest/questoes/{year}/` — question papers
- `raw_pdfs/fuvest/gabaritos/{year}/` — answer keys

### `scripts/extraction_pipeline.py`

Extraction pipeline that converts exam PDFs into structured `PhysicsExample` objects in JSONL format. Steps:

1. **PDF to text** — pages rendered as images via [PyMuPDF](https://pymupdf.readthedocs.io/), then OCR via Ollama vision model (`glm-ocr` by default), cached to disk
2. **Text to JSON** — via LLM (Gemini or local Ollama)
3. **JSON to JSONL** — validation and serialization using the project schema

```bash
# Set environment variables in .env:
#   GOOGLE_API_KEY=...           (for Gemini)
#   LLM_PROVIDER=ollama          (to use local Ollama)
#   OLLAMA_MODEL=qwen2.5:14b     (Ollama extraction model, optional)
#   OLLAMA_OCR_MODEL=glm-ocr     (Ollama vision model for OCR, optional)
#   OCR_DPI=150                   (image resolution for OCR, optional)
#   GEMINI_MODEL=gemini-2.0-flash (Gemini model, optional)

uv run scripts/extraction_pipeline.py

# Specific years
uv run scripts/extraction_pipeline.py --years 2020 2021 2022
```

Only physics-relevant PDFs are processed (`dia2.pdf`, `dia3.pdf`, `fisica.pdf`); other subjects and first-day exams are skipped automatically.

Output saved to `data/fuvest/physics.jsonl`.

### `scripts/dataset_schema.py`

Defines the dataset schema (`PhysicsExample`, `ReferenceData`, `Constant`) and utility functions:

- `load_dataset(path)` — loads JSONL into a list of `PhysicsExample`
- `save_dataset(examples, path)` — serializes to JSONL
- `validate_dataset(examples)` — checks required fields and consistency

No external dependencies.

## License

[CC BY 4.0](LICENSE)

Rights to the original exam questions belong to their respective boards.
