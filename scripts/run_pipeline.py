from __future__ import annotations

import argparse
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import dspy

from scripts.ocr import pdf_to_markdown
from scripts.pipeline.answer_extractor import AnswerExtractor
from scripts.pipeline.models import GradedQuestion, Subject, Vestibular
from scripts.pipeline.page_classifier import PageClassifier
from scripts.pipeline.question_extractor import QuestionExtractor

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


def _configure_dspy() -> tuple[dspy.LM, dspy.LM]:
    default_lm = dspy.LM(os.getenv("DEFAULT_LM", "gemini/gemini-2.5-flash"))
    lite_lm = dspy.LM(os.getenv("LITE_LM", "gemini/gemini-2.5-flash-lite"))

    dspy.configure(
        lm=default_lm,
    )
    return default_lm, lite_lm


@dataclass
class PipelineConfig:
    """Configuration for a single pipeline run.

    Attributes:
        vestibular:      Exam board name (e.g., "FUVEST").
        year:            Exam year.
        questions_pdf:   Path to the questions PDF.
        answers_pdf:     Path to the answers/gabarito PDF. If None, assumes
                         answers are embedded in questions_pdf.
        subject:         Subject to extract (e.g., "physics").
        cache_dir:       Directory for intermediate Markdown files.
        overwrite_cache: If True, re-runs OCR even if cache exists.
    """

    vestibular: Vestibular
    year: int
    exam_day: str
    questions_pdf: Path
    default_lm: dspy.LM
    lite_lm: dspy.LM
    answers_pdf: Path | None = None
    subject: Subject = "physics"
    cache_dir: Path = Path("cache")
    data_dir: Path = Path("data")
    overwrite_cache: bool = False


def run_pipeline(config: PipelineConfig) -> list[GradedQuestion]:
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

    # Step 1: OCR - questions PDF → Markdown
    questions_pdf_pages = pdf_to_markdown(
        config.questions_pdf,
        config.cache_dir,
        overwrite=config.overwrite_cache,
    )
    logger.info("%d pages extracted from questions PDF", len(questions_pdf_pages))

    # Step 2: classify pages
    classifier = PageClassifier()
    with dspy.settings.context(lm=config.lite_lm):
        question_pages = [
            page
            for page in questions_pdf_pages
            if classifier(page_markdown=page) == "questions"
        ]
    logger.info("%d question pages found", len(question_pages))

    # Step 3: extract questions
    extractor = QuestionExtractor(cache_dir=config.cache_dir / "questions")
    questions = extractor(
        pages=question_pages, vestibular=config.vestibular, year=config.year
    )
    logger.info("Questions extracted: %d questions", len(questions))

    # Step 4: OCR - solution PDF → Markdown
    solution_pdf_pages: list[str] = []
    if config.answers_pdf is not None:
        solution_pdf_pages = pdf_to_markdown(
            config.answers_pdf,
            config.cache_dir,
            overwrite=config.overwrite_cache,
        )
        logger.info("%d pages extracted from answers PDF", len(solution_pdf_pages))

    # Step 5: classify solution pages
    with dspy.settings.context(lm=config.lite_lm):
        solution_pages = [
            page
            for page in solution_pdf_pages
            if classifier(page_markdown=page) != "other"
        ]
    logger.info("%d solution pages after filtering", len(solution_pages))

    # Step 6: extract solutions
    answer_extractor = AnswerExtractor(
        cache_dir=config.cache_dir / "graded",
    )
    graded = answer_extractor(
        questions=questions,
        solution_pages=solution_pages,
    )
    logger.info("Solutions extracted: %d questions graded", len(graded))

    logger.info(
        "Pipeline done: %d questions (%.1fs)",
        len(questions),
        time.time() - start,
    )

    return graded


def _parse_exam_day(filename: str) -> str | None:
    """Extract exam day from PDF filename.

    Examples:
        'dia2.pdf' → '2nd day'
        'dia3.pdf' → '3rd day'
        'prova.pdf' → None
    """
    match = _DAY_RE.match(filename.lower())
    if not match:
        return None
    day = int(match.group(1))
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(day, "th")
    return f"{day}{suffix} day"


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


_DAY_RE = re.compile(r"^dia(\d+)\.pdf$", re.IGNORECASE)


def _is_pdf_relevant(filename: str) -> bool:
    """
    Return True if a question PDF is likely to contain relevant open-ended questions.
    """
    lower = filename.lower()
    day_match = _DAY_RE.match(lower)
    if day_match:
        return int(day_match.group(1)) >= 2
    return False


def _process_year(
    year: int,
    *,
    vestibular: Vestibular,
    questions_dir: Path,
    answers_dir: Path,
    cache_dir: Path,
    overwrite_cache: bool,
    default_lm: dspy.LM,
    lite_lm: dspy.LM,
) -> dict[str, int] | None:
    """Process all physics-relevant PDFs for a single year."""

    year_q_dir = questions_dir / str(year)
    if not year_q_dir.exists():
        logger.warning("SKIP %d — no directory: %s", year, year_q_dir)
        return None

    q_pdfs = sorted(year_q_dir.glob("*.pdf"))
    q_pdfs = [p for p in q_pdfs if _is_pdf_relevant(p.name)]
    if not q_pdfs:
        logger.warning("SKIP %d — no physics-relevant PDFs", year)
        return None

    year_a_dir = answers_dir / str(year)
    a_pdfs = sorted(year_a_dir.glob("*.pdf")) if year_a_dir.exists() else []
    a_pdf = _pick_answer_pdf(a_pdfs)

    for q_pdf in q_pdfs:
        exam_day = _parse_exam_day(q_pdf.name)
        if exam_day is None:
            logger.warning("SKIP %s — could not parse exam day", q_pdf.name)
            continue

        logger.info("Processing %s", q_pdf.name)
        config = PipelineConfig(
            vestibular=vestibular,
            year=year,
            exam_day=exam_day,
            questions_pdf=q_pdf,
            answers_pdf=a_pdf,
            cache_dir=cache_dir,
            overwrite_cache=overwrite_cache,
            default_lm=default_lm,
            lite_lm=lite_lm,
        )
        run_pipeline(config)


_BASE_DIR = Path(__file__).resolve().parent.parent


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

    default_lm, lite_lm = _configure_dspy()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    vest_lower = args.vestibular.lower()
    questions_dir = _BASE_DIR / "raw_pdfs" / vest_lower / "questoes"
    answers_dir = _BASE_DIR / "raw_pdfs" / vest_lower / "gabaritos"
    cache_dir = _BASE_DIR / "cache"

    years = _parse_years(args.years)
    logger.info("Processing %s for years: %s", args.vestibular, years)

    summary: dict[int, dict[str, int]] = {}

    for year in years:
        _process_year(
            year,
            vestibular=args.vestibular,
            questions_dir=questions_dir,
            answers_dir=answers_dir,
            cache_dir=cache_dir,
            overwrite_cache=args.overwrite_cache,
            default_lm=default_lm,
            lite_lm=lite_lm,
        )

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
