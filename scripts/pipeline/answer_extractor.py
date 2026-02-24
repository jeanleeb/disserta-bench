"""DSPy module for extracting answers and generating rubrics.

This module implements the second major pipeline stage: taking
previously extracted questions and enriching them with expected
answers (from official solution keys) and correction rubrics.

Pipeline flow:
    ExtractedQuestion[QuestionItem]
        → AnswerExtractor
    GradedQuestion[GradedQuestionItem]
"""

from __future__ import annotations

import logging
from pathlib import Path

import dspy

from .models import ExtractedQuestion, GradedQuestion, GradedQuestionItem, QuestionItem
from .signatures import (
    ExtractAnswers,
    ExtractedAnswerSet,
    GenerateRubric,
    IndexPageQuestions,
)

logger = logging.getLogger(__name__)


class AnswerExtractor(dspy.Module):
    """Extract answers from solution pages and generate rubrics.

    This module takes a list of ExtractedQuestions and solution page
    markdowns, and produces GradedQuestions with answers and rubrics
    populated.

    The module is idempotent — cached results are skipped on re-runs.

    Stages:
        1. Index solution pages to find which questions they cover.
        2. For each question found in solution pages, extract answers.
        3. For items still missing a rubric, generate one via LLM.
        4. Promote QuestionItem → GradedQuestionItem, persist to cache.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        super().__init__()
        self.index_questions = dspy.Predict(IndexPageQuestions)
        self.extract_answers = dspy.Predict(ExtractAnswers)
        self.generate_rubric = dspy.Predict(GenerateRubric)
        self.cache_dir = cache_dir

    # ── Stage 1: Map questions to solution pages ─────────────────────

    def _build_solution_map(self, solution_pages: list[str]) -> dict[str, list[str]]:
        """Map each question number to the solution page(s) that cover it.

        A question may span multiple pages; a page may cover multiple
        questions. The mapping is built by running IndexPageQuestions
        on each solution page.

        Returns:
            Dict of question_number → list of page markdown strings.
        """
        solution_map: dict[str, list[str]] = {}

        for page_md in solution_pages:
            result = self.index_questions(page_markdown=page_md)
            for qnum in result.question_numbers:
                solution_map.setdefault(qnum, []).append(page_md)

        logger.info(
            "Solution map: %d questions across %d pages",
            len(solution_map),
            len(solution_pages),
        )
        return solution_map

    # ── Stage 2: Extract answers for one question ────────────────────

    @staticmethod
    def _build_question_text(q: ExtractedQuestion) -> str:
        """Reconstruct full question text for use as extraction context."""
        parts: list[str] = []
        if q.question_preamble:
            parts.append(q.question_preamble)
        for item in q.items:
            prefix = f"{item.label} " if item.label else ""
            parts.append(f"{prefix}{item.item_text}")
        return "\n\n".join(parts)

    def _extract_for_question(
        self,
        q: ExtractedQuestion,
        solution_pages: list[str],
    ) -> ExtractedAnswerSet | None:
        """Call ExtractAnswers for a single question.

        Concatenates all relevant solution pages so the LM sees the
        full resolution context in one call.

        Returns:
            ExtractedAnswerSet on success, None on failure.
        """
        combined = "\n\n---PAGE BREAK---\n\n".join(solution_pages)

        try:
            result = self.extract_answers(
                solution_text=combined,
                question_number=q.question_number,
                question_text=self._build_question_text(q),
            )
            logger.info("Extracted answers for %s", q.question_number)
            return result.answers
        except Exception:
            logger.exception("ExtractAnswers failed for %s", q.question_number)
            return None

    # ── Stage 3: Generate rubric for a single item ───────────────────

    def _ensure_rubric(
        self,
        item: GradedQuestionItem,
        q: ExtractedQuestion,
    ) -> None:
        """Generate a rubric via LLM if the item has an answer but no rubric.

        Mutates `item` in place. Sets rubric_is_generated=True so the
        result is flagged for human review.
        """
        if item.rubric is not None:
            logger.info(
                "Rubric already exists for %s item '%s'", q.question_number, item.label
            )
            return
        if item.expected_answer is None:
            logger.info(
                "Skipping rubric for question without answer: %s item '%s'",
                q.question_number,
                item.label,
            )
            return

        try:
            result = self.generate_rubric(
                question_preamble=q.question_preamble,
                item_text=item.item_text,
                expected_answer=item.expected_answer,
                reference_data=q.reference_data,
            )
            item.rubric = result.rubric
            item.rubric_is_generated = True
            logger.info(
                "Generated rubric for %s item '%s'",
                q.question_number,
                item.label,
            )
        except Exception:
            logger.exception(
                "GenerateRubric failed for %s item '%s'",
                q.question_number,
                item.label,
            )

    # ── Stage 4: Merge answers into items ────────────────────────────

    @staticmethod
    def _merge_answers(
        graded_items: list[GradedQuestionItem],
        answers: ExtractedAnswerSet,
        question_number: str,
    ) -> int:
        """Merge extracted answers into graded items by label matching.

        Returns the number of items successfully updated.
        """
        answer_lookup: dict[str | None, tuple[str | None, str | None]] = {
            a.item_label: (a.expected_answer, a.rubric) for a in answers.items
        }

        merged = 0
        for item in graded_items:
            if item.label not in answer_lookup:
                logger.warning(
                    "No answer found for %s item '%s'",
                    question_number,
                    item.label,
                )
                continue

            expected, rubric = answer_lookup[item.label]
            if expected is not None:
                logger.info(
                    "Merged answer for %s item '%s'",
                    question_number,
                    item.label,
                )
                item.expected_answer = expected
            if rubric is not None:
                logger.info(
                    "Merged rubric for %s item '%s'",
                    question_number,
                    item.label,
                )
                item.rubric = rubric
                item.rubric_is_generated = False
            merged += 1

        return merged

    # ── Promotion: QuestionItem → GradedQuestionItem ─────────────────

    @staticmethod
    def _promote_item(item: QuestionItem) -> GradedQuestionItem:
        """Promote a QuestionItem to GradedQuestionItem with default fields."""
        return GradedQuestionItem(**item.model_dump())

    @staticmethod
    def _promote_question(q: ExtractedQuestion) -> GradedQuestion:
        """Promote an ExtractedQuestion to GradedQuestion.

        All items are promoted to GradedQuestionItem with default
        (None/False) answer and rubric fields.
        """
        data = q.model_dump()
        data["items"] = [GradedQuestionItem(**item_data) for item_data in data["items"]]
        return GradedQuestion(**data)

    # ── Cache ────────────────────────────────────────────────────────

    def _cache_path(self, q: GradedQuestion) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{q.vestibular}_{q.year}_{q.question_number}.json"

    def _load_from_cache(self, q: ExtractedQuestion) -> GradedQuestion | None:
        """Try to load a previously graded question from cache."""
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{q.vestibular}_{q.year}_{q.question_number}.json"
        if not path.exists():
            return None
        try:
            return GradedQuestion.model_validate_json(path.read_text("utf-8"))
        except Exception:
            logger.warning("Corrupt cache file %s — will reprocess", path)
            return None

    def _save_to_cache(self, q: GradedQuestion) -> None:
        path = self._cache_path(q)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(q.model_dump_json(indent=2), encoding="utf-8")
        logger.debug("Cached graded question: %s", path)

    # ── Forward ──────────────────────────────────────────────────────

    def forward(
        self,
        questions: list[ExtractedQuestion],
        solution_pages: list[str],
    ) -> list[GradedQuestion]:
        """Process all questions against solution pages.

        Args:
            questions: Previously extracted questions (not mutated).
            solution_pages: Markdown of pages classified as "solutions".
                Empty list when no official solutions are available.

        Returns:
            List of GradedQuestion with answers and rubrics populated
            where available.
        """
        # 1) Build solution map (empty dict if no solution pages)
        solution_map = (
            self._build_solution_map(solution_pages) if solution_pages else {}
        )

        graded: list[GradedQuestion] = []

        for q in questions:
            # Check cache first — idempotency
            cached = self._load_from_cache(q)
            if cached is not None:
                logger.debug("Cache hit for %s", q.question_number)
                graded.append(cached)
                continue

            # Promote to graded types
            gq = self._promote_question(q)

            # 2) Extract answers if solution pages exist for this question
            pages = solution_map.get(q.question_number, [])
            if pages:
                answers = self._extract_for_question(q, pages)
                logger.info(
                    "Extracted %d answers for %s",
                    len(answers.items) if answers is not None else 0,
                    q.question_number,
                )
                if answers:
                    self._merge_answers(gq.items, answers, q.question_number)

            # 3) Generate rubrics for items that still need one
            for item in gq.items:
                self._ensure_rubric(item, q)

            # 4) Cache and collect
            self._save_to_cache(gq)
            graded.append(gq)

        # Summary log
        total = sum(len(gq.items) for gq in graded)
        with_answer = sum(1 for gq in graded for it in gq.items if it.expected_answer)
        with_rubric = sum(1 for gq in graded for it in gq.items if it.rubric)
        generated = sum(1 for gq in graded for it in gq.items if it.rubric_is_generated)
        logger.info(
            "AnswerExtractor: %d items | %d with answers | "
            "%d with rubrics (%d generated)",
            total,
            with_answer,
            with_rubric,
            generated,
        )

        return graded
