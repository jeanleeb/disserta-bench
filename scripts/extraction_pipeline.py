"""
Extraction pipeline for disserta-bench.

Converts official exam PDFs into structured PhysicsExample objects using:
  1. Marker  — PDF to Markdown with LaTeX OCR (cached to disk)
  2. LLM     — Markdown to structured JSON (Ollama locally, Gemini in production)
  3. Schema  — JSON to PhysicsExample + JSONL output

Directory conventions:
  raw_pdfs/   — original PDFs, gitignored, downloaded manually
  cache/      — intermediate Markdown files, gitignored
  data/       — final JSONL files, versioned in git
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

from dataset_schema import (
    Constant,
    PhysicsExample,
    ReferenceData,
    save_dataset,
    validate_dataset,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    Configuration for a single pipeline run.

    Attributes:
        vestibular:     Exam board name (e.g., "FUVEST").
        year:           Exam year.
        questions_pdf:  Path to the questions PDF.
        answers_pdf:    Path to the answers/gabarito PDF. If None, assumes
                        answers are embedded in questions_pdf.
        output_path:    Destination JSONL file.
        cache_dir:      Directory for intermediate Markdown files.
        model:          Ollama model name or "gemini" to use Gemini API.
        overwrite_cache: If True, re-runs Marker even if cache exists.
    """

    vestibular: str
    year: int
    questions_pdf: Path
    output_path: Path
    answers_pdf: Path | None = None
    cache_dir: Path = Path("cache")
    model: str = "llama3.2"
    overwrite_cache: bool = False


# ---------------------------------------------------------------------------
# Step 1: PDF → Markdown (Marker)
# ---------------------------------------------------------------------------

def pdf_to_markdown(pdf_path: Path, cache_dir: Path, overwrite: bool = False) -> str:
    """
    Convert a PDF to Markdown using Marker, with disk caching.

    The cache key is the PDF filename stem. If a cached .md file exists and
    overwrite is False, the cached version is returned immediately.

    Args:
        pdf_path:   Path to the source PDF.
        cache_dir:  Directory where .md cache files are stored.
        overwrite:  If True, ignore existing cache and re-run Marker.

    Returns:
        Markdown string with inline LaTeX equations.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{pdf_path.stem}.md"

    if cache_path.exists() and not overwrite:
        logger.info("Cache hit: %s", cache_path)
        return cache_path.read_text(encoding="utf-8")

    logger.info("Running Marker on %s ...", pdf_path)
    markdown = _run_marker(pdf_path)

    cache_path.write_text(markdown, encoding="utf-8")
    logger.info("Cached Markdown to %s", cache_path)

    return markdown


def _run_marker(pdf_path: Path) -> str:
    """
    Run Marker OCR on a PDF and return the resulting Markdown.

    Marker is imported lazily so that the pipeline module can be imported
    without Marker installed (e.g., in environments that only consume the
    dataset schema).
    """
    try:
        from marker.config.parser import ConfigParser
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
    except ImportError as e:
        raise ImportError(
            "Marker is not installed. Run: uv add marker-pdf"
        ) from e

    config_parser = ConfigParser({})
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
    )
    rendered = converter(str(pdf_path))
    markdown, _, _ = text_from_rendered(rendered)
    return markdown


# ---------------------------------------------------------------------------
# Step 2: Markdown → structured dicts (LLM)
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """
You are processing a Brazilian university entrance exam (vestibular) in physics.

Below is the Markdown of the QUESTIONS section followed by the ANSWERS section.
Extract each physics question as a JSON object.

Return a JSON array where each element has exactly these fields:
- question_number: string (e.g. "Q3", "Q5a")
- topic: string (physics subfield in English, e.g. "Thermodynamics")
- question: string (full question text, preserve inline LaTeX as $...$ or $$...$$)
- reference_data: object with "constants" array, each having "symbol", "value", "unit"
- expected_value: number (official numerical answer)
- expected_unit: string (SI unit, e.g. "m/s", "J", "N")
- solution_steps: string (step-by-step solution from the answer key)
- rubric: array of strings (3-5 specific evaluation criteria for this question)
- has_figure: boolean
- figure_description: string (empty if has_figure is false)

Rules:
- Include ONLY questions that have a clear numerical answer.
- Keep question text in Brazilian Portuguese.
- Write rubric criteria in Brazilian Portuguese.
- Write solution_steps in Brazilian Portuguese.
- Return ONLY the JSON array, no markdown fences, no explanation.

QUESTIONS MARKDOWN:
{questions_markdown}

ANSWERS MARKDOWN:
{answers_markdown}
"""


def extract_questions_with_llm(
    questions_markdown: str,
    answers_markdown: str,
    model: str,
) -> list[dict]:
    """
    Send Markdown content to an LLM and parse the returned JSON array.

    Args:
        questions_markdown: Marker output for the questions PDF.
        answers_markdown:   Marker output for the answers PDF.
        model:              Ollama model name or "gemini".

    Returns:
        List of raw dicts, one per extracted question.
    """
    prompt = _EXTRACTION_PROMPT.format(
        questions_markdown=questions_markdown,
        answers_markdown=answers_markdown,
    )

    if model == "gemini":
        raw = _call_gemini(prompt)
    else:
        raw = _call_ollama(prompt, model)

    return _parse_json_response(raw)


def _call_ollama(prompt: str, model: str) -> str:
    """Call a local Ollama model and return the raw text response."""
    try:
        import ollama
    except ImportError as e:
        raise ImportError("Ollama is not installed. Run: uv add ollama") from e

    logger.info("Calling Ollama model: %s", model)
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def _call_gemini(prompt: str) -> str:
    """Call the Gemini API and return the raw text response."""
    try:
        import google.generativeai as genai
    except ImportError as e:
        raise ImportError(
            "google-generativeai is not installed. Run: uv add google-generativeai"
        ) from e

    logger.info("Calling Gemini API ...")
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text


def _parse_json_response(raw: str) -> list[dict]:
    """
    Parse a JSON array from an LLM response, stripping markdown fences if present.

    Args:
        raw: Raw LLM output, possibly wrapped in ```json ... ```.

    Returns:
        Parsed list of dicts.

    Raises:
        ValueError: If the response cannot be parsed as a JSON array.
    """
    # Strip markdown code fences if the LLM wrapped the output
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"```\s*$", "", cleaned.strip(), flags=re.MULTILINE)
    cleaned = cleaned.strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM response is not valid JSON.\n"
            f"Error: {e}\n"
            f"Response (first 500 chars):\n{raw[:500]}"
        ) from e

    if not isinstance(result, list):
        raise ValueError(
            f"Expected a JSON array, got {type(result).__name__}."
        )

    return result


# ---------------------------------------------------------------------------
# Step 3: dicts → PhysicsExample objects
# ---------------------------------------------------------------------------

def dicts_to_examples(
    raw_questions: list[dict],
    vestibular: str,
    year: int,
) -> list[PhysicsExample]:
    """
    Convert raw LLM-extracted dicts to validated PhysicsExample objects.

    Questions that cannot be converted (missing required fields, malformed
    data) are skipped with a warning rather than crashing the pipeline.

    Args:
        raw_questions: List of dicts from the LLM extraction step.
        vestibular:    Exam board name (e.g., "FUVEST").
        year:          Exam year.

    Returns:
        List of PhysicsExample instances.
    """
    examples = []
    for i, raw in enumerate(raw_questions):
        try:
            example = _dict_to_example(raw, vestibular, year)
            examples.append(example)
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(
                "Skipping question %d (%s): %s",
                i,
                raw.get("question_number", "unknown"),
                e,
            )
    return examples


def _dict_to_example(raw: dict, vestibular: str, year: int) -> PhysicsExample:
    """Convert a single raw dict to a PhysicsExample, raising on bad data."""
    ref_data = raw.get("reference_data", {})
    constants = [
        Constant(
            symbol=c["symbol"],
            value=float(c["value"]),
            unit=c["unit"],
        )
        for c in ref_data.get("constants", [])
    ]

    return PhysicsExample(
        vestibular=vestibular,
        year=year,
        question_number=raw["question_number"],
        topic=raw["topic"],
        question=raw["question"],
        reference_data=ReferenceData(constants=constants),
        expected_value=(
            float(raw["expected_value"])
            if raw.get("expected_value") is not None
            else None
        ),
        expected_unit=raw.get("expected_unit", ""),
        solution_steps=raw.get("solution_steps", ""),
        rubric=raw.get("rubric", []),
        has_figure=bool(raw.get("has_figure", False)),
        figure_description=raw.get("figure_description", ""),
    )


# ---------------------------------------------------------------------------
# Main pipeline entrypoint
# ---------------------------------------------------------------------------

def run_pipeline(config: PipelineConfig) -> list[PhysicsExample]:
    """
    Run the full extraction pipeline for one exam.

    Steps:
        1. Convert questions PDF to Markdown (cached).
        2. Convert answers PDF to Markdown (cached), or reuse questions Markdown.
        3. Extract structured questions via LLM.
        4. Convert to PhysicsExample objects.
        5. Validate and log warnings.
        6. Save to JSONL.

    Args:
        config: Pipeline configuration for this exam.

    Returns:
        List of extracted PhysicsExample instances.
    """
    start = time.time()
    logger.info(
        "Starting pipeline: %s %d", config.vestibular, config.year
    )

    # Step 1: questions PDF → Markdown
    questions_md = pdf_to_markdown(
        config.questions_pdf,
        config.cache_dir,
        overwrite=config.overwrite_cache,
    )

    # Step 2: answers PDF → Markdown (or same as questions if not provided)
    if config.answers_pdf is not None:
        answers_md = pdf_to_markdown(
            config.answers_pdf,
            config.cache_dir,
            overwrite=config.overwrite_cache,
        )
    else:
        logger.info("No separate answers PDF — using questions Markdown for both.")
        answers_md = questions_md

    # Step 3: LLM extraction
    raw_questions = extract_questions_with_llm(
        questions_md, answers_md, config.model
    )
    logger.info("LLM extracted %d raw questions.", len(raw_questions))

    # Step 4: convert to schema objects
    examples = dicts_to_examples(raw_questions, config.vestibular, config.year)
    logger.info("Converted %d questions successfully.", len(examples))

    # Step 5: validate
    warnings = validate_dataset(examples)
    for warning in warnings:
        logger.warning(warning)

    # Step 6: save
    save_dataset(examples, config.output_path)
    logger.info(
        "Saved %d examples to %s (%.1fs)",
        len(examples),
        config.output_path,
        time.time() - start,
    )

    return examples


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Example: process FUVEST 2024
    config = PipelineConfig(
        vestibular="FUVEST",
        year=2024,
        questions_pdf=Path("raw_pdfs/fuvest_2024_questoes.pdf"),
        answers_pdf=Path("raw_pdfs/fuvest_2024_gabarito.pdf"),
        output_path=Path("data/fuvest/physics.jsonl"),
        cache_dir=Path("cache"),
        model="llama3.2",          # swap to "gemini" for production
    )

    examples = run_pipeline(config)
    print(f"\nDone — {len(examples)} questions extracted.")
