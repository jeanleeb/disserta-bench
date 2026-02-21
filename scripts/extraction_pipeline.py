"""
Extraction pipeline for disserta-bench.

Converts official exam PDFs into structured PhysicsExample objects using:
  1. OCR     — PDF pages rendered as images, then sent to a vision LLM
               for text extraction (``scripts.ocr``)
  2. LLM     — Markdown to structured JSON (``scripts.llm``)
  3. Schema  — JSON to PhysicsExample + JSONL output

Directory conventions:
  raw_pdfs/   — original PDFs, gitignored, downloaded manually
  cache/      — intermediate Markdown files, gitignored
  data/       — final JSONL files, versioned in git
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.dataset_schema import (
    Constant,
    PhysicsExample,
    ReferenceData,
    save_dataset,
    validate_dataset,
)
from scripts.llm import extract_questions_with_llm
from scripts.ocr import pdf_to_markdown

logger = logging.getLogger(__name__)


def _load_dotenv() -> None:
    """Load .env file from the project root if it exists, without overwriting."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    with env_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Configuration for a single pipeline run.

    Attributes:
        vestibular:     Exam board name (e.g., "FUVEST").
        year:           Exam year.
        questions_pdf:  Path to the questions PDF.
        answers_pdf:    Path to the answers/gabarito PDF. If None, assumes
                        answers are embedded in questions_pdf.
        cache_dir:      Directory for intermediate Markdown files.
        overwrite_cache: If True, re-runs OCR even if cache exists.
    """

    vestibular: str
    year: int
    questions_pdf: Path
    answers_pdf: Path | None = None
    cache_dir: Path = Path("cache")
    overwrite_cache: bool = False


# ---------------------------------------------------------------------------
# dicts → PhysicsExample objects
# ---------------------------------------------------------------------------


def dicts_to_examples(
    raw_questions: list[Any],
    vestibular: str,
    year: int,
) -> list[PhysicsExample]:
    """Convert raw LLM-extracted dicts to validated PhysicsExample objects.

    Questions that cannot be converted (missing required fields, malformed
    data) are skipped with a warning rather than crashing the pipeline.
    """
    examples = []
    for i, raw in enumerate(raw_questions):
        question_number = (
            raw.get("question_number", "unknown")
            if isinstance(raw, dict)
            else "unknown"
        )
        try:
            example = _dict_to_example(raw, vestibular, year)
            examples.append(example)
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.warning(
                "Skipping question %d (%s): %s",
                i,
                question_number,
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
    """Run the full extraction pipeline for one exam.

    Steps:
        1. Convert questions PDF to Markdown (cached).
        2. Convert answers PDF to Markdown (cached), or reuse questions Markdown.
        3. Extract structured questions via LLM.
        4. Convert to PhysicsExample objects.
        5. Validate and log warnings.
    """
    start = time.time()
    logger.info("Starting pipeline: %s %d", config.vestibular, config.year)

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
    raw_questions = extract_questions_with_llm(questions_md, answers_md)
    logger.info("LLM extracted %d raw questions.", len(raw_questions))

    # Step 4: convert to schema objects
    examples = dicts_to_examples(raw_questions, config.vestibular, config.year)
    logger.info("Converted %d questions successfully.", len(examples))

    # Step 5: validate
    warnings = validate_dataset(examples)
    for warning in warnings:
        logger.warning(warning)

    logger.info(
        "Pipeline done: %d examples (%.1fs)",
        len(examples),
        time.time() - start,
    )

    return examples


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

_BASE_DIR = Path(__file__).resolve().parent.parent

_DAY_RE = re.compile(r"^dia(\d+)\.pdf$", re.IGNORECASE)


def _is_physics_relevant(filename: str) -> bool:
    """Return True if a question PDF is likely to contain physics questions."""
    lower = filename.lower()
    if "fisica" in lower:
        return True
    day_match = _DAY_RE.match(lower)
    if day_match:
        return int(day_match.group(1)) >= 2
    return False


def _pick_answer_pdf(a_pdfs: list[Path]) -> Path | None:
    """Pick the best answer PDF from available files.

    Prefers physics-specific answer keys (``guia_respostas_fisica.pdf``)
    over general ones.
    """
    if not a_pdfs:
        return None
    for pdf in a_pdfs:
        if "fisica" in pdf.name.lower():
            return pdf
    return a_pdfs[0]


def _parse_years(raw: list[str]) -> list[int]:
    """Parse year arguments, supporting both ``2020`` and ``2020-2025``."""
    years: list[int] = []
    for token in raw:
        if "-" in token:
            lo, hi = token.split("-", 1)
            years.extend(range(int(lo), int(hi) + 1))
        else:
            years.append(int(token))
    years = [y for y in years if 1990 <= y <= 2100]
    return sorted(set(years))


def _process_year(
    year: int,
    *,
    vestibular: str,
    questions_dir: Path,
    answers_dir: Path,
    data_dir: Path,
    review_dir: Path,
    cache_dir: Path,
    overwrite_cache: bool,
) -> dict[str, int] | None:
    """Process all physics-relevant PDFs for a single year."""
    vest_lower = vestibular.lower()

    year_q_dir = questions_dir / str(year)
    if not year_q_dir.exists():
        logger.warning("SKIP %d — no directory: %s", year, year_q_dir)
        return None

    q_pdfs = sorted(year_q_dir.glob("*.pdf"))
    q_pdfs = [p for p in q_pdfs if _is_physics_relevant(p.name)]
    if not q_pdfs:
        logger.warning("SKIP %d — no physics-relevant PDFs", year)
        return None

    year_a_dir = answers_dir / str(year)
    a_pdfs = sorted(year_a_dir.glob("*.pdf")) if year_a_dir.exists() else []
    a_pdf = _pick_answer_pdf(a_pdfs)

    all_examples: list[PhysicsExample] = []
    for q_pdf in q_pdfs:
        logger.info("Processing %s", q_pdf.name)
        config = PipelineConfig(
            vestibular=vestibular,
            year=year,
            questions_pdf=q_pdf,
            answers_pdf=a_pdf,
            cache_dir=cache_dir,
            overwrite_cache=overwrite_cache,
        )
        all_examples.extend(run_pipeline(config))

    complete = [ex for ex in all_examples if ex.expected_value is not None]
    review = []
    for ex in all_examples:
        if ex.expected_value is None:
            ex.needs_review = True
            review.append(ex)

    if complete:
        out = data_dir / f"{vest_lower}_{year}.jsonl"
        save_dataset(complete, out)
        logger.info("Saved %d complete examples to %s", len(complete), out)

    if review:
        out = review_dir / f"{vest_lower}_{year}.jsonl"
        save_dataset(review, out)
        logger.info("Saved %d examples for review to %s", len(review), out)

    return {"complete": len(complete), "review": len(review)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the extraction pipeline on downloaded exam PDFs.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=["1997-2025"],
        help="Years to process (e.g. 2024 or 1997-2025). Default: 1997-2025",
    )
    parser.add_argument(
        "--vestibular",
        default="FUVEST",
        help="Exam board name. Default: FUVEST",
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Re-run OCR even if cached Markdown exists.",
    )
    args = parser.parse_args(argv)

    _load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    vest_lower = args.vestibular.lower()
    questions_dir = _BASE_DIR / "raw_pdfs" / vest_lower / "questoes"
    answers_dir = _BASE_DIR / "raw_pdfs" / vest_lower / "gabaritos"
    data_dir = _BASE_DIR / "data" / vest_lower
    review_dir = data_dir / "review"
    cache_dir = _BASE_DIR / "cache"

    years = _parse_years(args.years)
    logger.info("Processing %s for years: %s", args.vestibular, years)

    summary: dict[int, dict[str, int]] = {}

    for year in years:
        counts = _process_year(
            year,
            vestibular=args.vestibular,
            questions_dir=questions_dir,
            answers_dir=answers_dir,
            data_dir=data_dir,
            review_dir=review_dir,
            cache_dir=cache_dir,
            overwrite_cache=args.overwrite_cache,
        )
        if counts is not None:
            summary[year] = counts

    logger.info("--- Summary ---")
    for year, counts in summary.items():
        logger.info(
            "  %d  complete=%-3d  review=%d",
            year,
            counts["complete"],
            counts["review"],
        )


if __name__ == "__main__":
    main()
