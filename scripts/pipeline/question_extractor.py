import logging
from pathlib import Path

import dspy

from scripts.pipeline.models import ExtractedQuestion, Vestibular
from scripts.pipeline.signatures import ExtractQuestion, IndexPageQuestions

logger = logging.getLogger(__name__)


class QuestionExtractor(dspy.Module):
    """Extracts all physics questions from a list of exam page Markdowns.

    Usage:
        extractor = QuestionExtractor()
        questions = extractor(pages=["## Questão 1...", "## Questão 2..."])
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        super().__init__()
        self.cache_dir = cache_dir

        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

        self.index = dspy.Predict(IndexPageQuestions)
        self.extract = dspy.Predict(ExtractQuestion)

    def _cache_path(
        self, question_number: str, vestibular: str, year: int
    ) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{vestibular.lower()}_{year}_{question_number}.json"

    def _read_cache(
        self, question_number: str, vestibular: str, year: int
    ) -> ExtractedQuestion | None:
        path = self._cache_path(question_number, vestibular, year)
        if path is None or not path.exists():
            return None
        logger.info("Cache hit: %s", question_number)
        return ExtractedQuestion.model_validate_json(path.read_text(encoding="utf-8"))

    def _write_cache(self, question: ExtractedQuestion) -> None:
        path = self._cache_path(
            question.question_number, question.vestibular, question.year
        )
        if path is None:
            return
        path.write_text(question.model_dump_json(indent=2), encoding="utf-8")

    def forward(
        self, pages: list[str], vestibular: Vestibular, year: int
    ) -> list[ExtractedQuestion]:
        """Extract all questions from a list of page Markdowns.

        Args:
            pages: One Markdown string per PDF page. Should already be
                   filtered to 'questions' pages by PageClassifier.

        Returns:
            Flat list of AIExtractedQuestion, in page order.
        """
        results: list[ExtractedQuestion] = []

        for page_markdown in pages:
            index = self.index(page_markdown=page_markdown)

            if not index.question_numbers:
                continue

            for question_number in index.question_numbers:
                cached = self._read_cache(question_number, vestibular, year)
                if cached is not None:
                    results.append(cached)
                    continue

                prediction = self.extract(
                    page_markdown=page_markdown,
                    question_number=question_number,
                )
                extracted_question = ExtractedQuestion(
                    **prediction.question_data.model_dump(),
                    vestibular=vestibular,
                    year=year,
                )
                self._write_cache(extracted_question)
                results.append(extracted_question)
                logger.info("Extracted %s successfully.", question_number)

        return results
