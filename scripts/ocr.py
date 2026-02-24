"""
PDF to Markdown conversion via vision-LLM OCR.

Renders each PDF page as a PNG image using pymupdf, then sends it to a
vision model (Gemini Flash by default, Ollama as alternative) for text
extraction.  Results are cached **per page** so that a crash mid-document
does not lose work already done.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_OCR_PROMPT = (
    "Extract all text from this exam page. "
    "Preserve the original structure, including "
    "LaTeX equations as $...$ or $$...$$.\n"
    "Questions are identified by codes: one uppercase letter followed "
    "by two digits (e.g. F01, M03, Q05). Be very careful to transcribe "
    "these codes exactly as printed — do NOT confuse similar-looking "
    "letters (E vs F, G vs C, etc.).\n"
    "Return only the extracted text, no commentary."
)

_PAGE_SEPARATOR = "\n\n---\n\n"


# ---------------------------------------------------------------------------
# Internal: page-level OCR dispatching
# ---------------------------------------------------------------------------


def _ocr_page_fn():
    """Return a callable ``(img_bytes, page_num, total) -> str`` for the
    configured OCR provider.  Creating the client once and closing over it
    avoids re-authenticating on every page.
    """
    provider = os.environ.get("OCR_PROVIDER", "gemini")
    if provider not in ("gemini", "ollama"):
        raise ValueError(
            f"Unknown OCR_PROVIDER: {provider!r}. Use 'gemini' or 'ollama'."
        )
    if provider == "gemini":
        return _make_gemini_ocr()
    return _make_ollama_ocr()


def _make_gemini_ocr():
    """Build a closure that OCRs a single page via Gemini."""
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise OSError(
            "GOOGLE_API_KEY is required for OCR_PROVIDER=gemini. "
            "Set it in .env or use OCR_PROVIDER=ollama for local OCR."
        )
    client = genai.Client(api_key=api_key)
    model_name = os.environ.get("GEMINI_OCR_MODEL", "gemini-2.5-flash")

    def _ocr(img_bytes: bytes, *, page_num: int, total: int) -> str:
        logger.info("OCR page %d/%d with %s", page_num, total, model_name)
        resp = client.models.generate_content(
            model=model_name,
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                _OCR_PROMPT,
            ],
        )
        if resp.text is None:
            logger.warning("OCR page %d returned empty response", page_num)
            return ""
        return resp.text

    return _ocr


def _make_ollama_ocr():
    """Build a closure that OCRs a single page via Ollama."""
    import base64

    import ollama

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    model = os.environ.get("OLLAMA_OCR_MODEL", "glm-ocr")
    client = ollama.Client(host=host)

    def _ocr(img_bytes: bytes, *, page_num: int, total: int) -> str:
        img_b64 = base64.b64encode(img_bytes).decode("ascii")
        logger.info("OCR page %d/%d with %s", page_num, total, model)
        resp = client.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": _OCR_PROMPT,
                    "images": [img_b64],
                }
            ],
        )
        return resp["message"]["content"]

    return _ocr


# ---------------------------------------------------------------------------
# Internal: PDF rendering
# ---------------------------------------------------------------------------


def _pdf_to_images(pdf_path: Path) -> list[bytes]:
    """Render each page of a PDF as a PNG image using pymupdf."""
    import pymupdf

    dpi = int(os.environ.get("OCR_DPI", "200"))
    doc = pymupdf.open(str(pdf_path))
    images: list[bytes] = []
    for page in doc:
        pixmap = page.get_pixmap(dpi=dpi)
        images.append(pixmap.tobytes("png"))
    doc.close()
    return images

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------


ocr_fn = _ocr_page_fn()


def pdf_to_markdown(
    pdf_path: Path,
    cache_dir: Path,
    overwrite: bool = False,
) -> list[str]:
    """Convert a PDF to Markdown via vision-LLM OCR, with per-page caching.

    Cache layout::

        cache/
          <stem>-<hash>/        # one subdir per PDF
            page-01.md
            page-02.md
            ...
          <stem>-<hash>.md      # combined final file

    The cache key incorporates the resolved path, file size, and mtime so
    that renames, moves, and in-place edits all invalidate correctly.

    If the combined ``.md`` already exists and *overwrite* is False the
    cached version is returned immediately (fast path).  Otherwise each
    page is checked individually — only missing pages are sent through OCR,
    making interrupted runs resumable.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    resolved = pdf_path.resolve()
    stat = resolved.stat()
    material = f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}"
    cache_key = hashlib.sha256(material.encode()).hexdigest()[:12]

    combined_path = cache_dir / f"{pdf_path.stem}-{cache_key}.md"
    page_dir = cache_dir / f"{pdf_path.stem}-{cache_key}"

    # Fast path: combined file exists and no overwrite requested.
    if combined_path.exists() and not overwrite:
        logger.info("Cache hit: %s", combined_path)
        return combined_path.read_text(encoding="utf-8").split(_PAGE_SEPARATOR)

    # Render all pages to PNG images.
    images = _pdf_to_images(pdf_path)
    page_dir.mkdir(parents=True, exist_ok=True)

    # OCR each page, skipping those already cached.
    pages_text: list[str] = []

    for i, img_bytes in enumerate(images):
        page_cache = page_dir / f"page-{i + 1:02d}.md"

        if page_cache.exists() and not overwrite:
            logger.info("Page %d/%d cached", i + 1, len(images))
            pages_text.append(page_cache.read_text(encoding="utf-8"))
            continue

        text = ocr_fn(img_bytes=img_bytes, page_num=i + 1, total=len(images))
        page_cache.write_text(text, encoding="utf-8")
        pages_text.append(text)

    combined = _PAGE_SEPARATOR.join(pages_text)
    combined_path.write_text(combined, encoding="utf-8")
    logger.info("Cached Markdown to %s", combined_path)
    return pages_text
