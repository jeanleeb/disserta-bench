from typing import Literal

from pydantic import BaseModel, Field, model_validator

Vestibular = Literal["FUVEST", "ITA", "Unicamp"]

Subject = Literal[
    "physics",
    "mathematics",
    "chemistry",
    "biology",
    "geography",
    "history",
    "portuguese",
]

Topic = Literal[
    # Physics
    "kinematics",
    "dynamics",
    "energy and work",
    "rotational mechanics",
    "fluid mechanics",
    "thermodynamics",
    "waves",
    "optics",
    "electrostatics",
    "electrodynamics",
    "electromagnetism",
    "modern physics",
    # Mathematics
    "algebra",
    "functions",
    "trigonometry",
    "analytic geometry",
    "plane geometry",
    "spatial geometry",
    "combinatorics and probability",
    "sequences and series",
    "complex numbers and polynomials",
    # Chemistry
    "general chemistry",
    "atomic structure",
    "chemical bonding",
    "stoichiometry",
    "solutions",
    "thermochemistry",
    "chemical kinetics",
    "chemical equilibrium",
    "electrochemistry",
    "organic chemistry",
    "biochemistry",
    # Biology
    "cell biology",
    "genetics",
    "evolution",
    "ecology",
    "human physiology",
    "plant physiology",
    "zoology",
    "microbiology",
    # Geography
    "physical geography",
    "human geography",
    "brazilian geography",
    "geopolitics",
    "environmental geography",
    # History
    "ancient history",
    "medieval history",
    "modern history",
    "contemporary history",
    "brazilian colonial history",
    "brazilian imperial and republican history",
    # Portuguese
    "literature",
    "grammar",
    "text interpretation",
]


class QuestionItem(BaseModel):
    """A single evaluable unit — the atomic element of the dataset.

    A question without sub-items is represented as a single QuestionItem
    with label=None. This avoids special-casing throughout the pipeline.
    """

    label: str | None = Field(
        description=(
            "Sub-item label exactly as printed (e.g. 'a)', 'b)', 'i.'). "
            "None for standalone questions without sub-items."
        )
    )
    topic: Topic = Field(
        description=(
            "Subject subfield tested by this specific item."
            "Must be under parent question's subject."
        )
    )

    item_text: str = Field(
        description=(
            "Full text of this sub-item or standalone question, "
            "with LaTeX for equations."
        )
    )


class AIExtractedQuestion(BaseModel):
    subject: Subject = Field(description="Subject of the question")

    question_number: str = Field(
        description=(
            "Question identifier exactly as printed (e.g., 'F01', 'M03')."
            "If question has sub-items, this is the identifier of the parent question."
        )
    )
    question_preamble: str | None = Field(
        description=(
            "Shared preamble common to all sub-items: context, data, "
            "figure references. None if there is no shared preamble, "
            "full question text if there are no sub-items."
        )
    )
    reference_data: list[str] = Field(
        description=(
            "Reference data provided with the question: constants, formulas, "
            "and assumptions shared across all sub-items. "
            "Examples: 'g = 10 m/s²', 'despreze a resistência do ar', "
            "'$v^2 = v_0^2 + 2a\\Delta x$'. "
            "Empty list if none are provided."
        ),
        default_factory=list,
    )
    has_figure: bool = Field(description="Whether the question has a figure")
    figure_description: str | None = Field(
        description="Description of the figure if present"
    )
    items: list[QuestionItem] = Field(
        description=(
            "Evaluable items in order of appearance. "
            "Single-element list for standalone questions."
        )
    )


class ExtractedQuestion(AIExtractedQuestion):
    """
    A single open-ended question from a Brazilian university entrance exam.

    Agent inputs (what a solver should receive):
        question, reference_data

    Evaluation labels (for metrics and judges only):
        expected_answers, solution_steps, rubric
    """

    # --- Metadata ---
    vestibular: Vestibular = Field(description="Name of the exam")
    year: int = Field(description="Exam year")


ExamPageType = Literal["questions", "solutions", "other"]


class ExtractedItemAnswer(BaseModel):
    """Answer and rubric for a single question item."""

    item_label: str | None = Field(
        description=(
            "Sub-item label exactly as printed, e.g. 'a)', 'b)'. "
            "Must be None (not empty string) when the question has "
            "no sub-items — in that case, return exactly one item."
        )
    )
    expected_answer: str | None = Field(
        description=(
            "The expected final answer as stated in the solution. "
            "Include units when present. None if the solution does not "
            "provide a clear expected answer for this item."
        )
    )
    rubric: str | None = Field(
        description=(
            "Correction criteria or grading rubric as stated in the "
            "official solution. None if no explicit criteria are given."
        )
    )


class ExtractedAnswerSet(BaseModel):
    """All answers for a single question, grouped by item."""

    items: list[ExtractedItemAnswer] = Field(
        description=(
            "One entry per sub-item, in the same order as the exam. "
            "For questions without sub-items, this list has exactly "
            "one element with item_label=None."
        )
    )

    @model_validator(mode="after")
    def _check_label_consistency(self) -> "ExtractedAnswerSet":
        labels = [it.item_label for it in self.items]
        has_none = None in labels
        has_labels = any(lb is not None for lb in labels)

        if has_none and has_labels:
            msg = (
                "Inconsistent item labels: cannot mix None and "
                f"labeled items. Got: {labels}"
            )
            raise ValueError(msg)

        if has_none and len(self.items) != 1:
            msg = (
                "Questions without sub-items must have exactly one item, "
                f"got {len(self.items)}"
            )
            raise ValueError(msg)

        return self


class GradedQuestionItem(QuestionItem):
    """QuestionItem enriched with answer extraction and rubric data."""

    expected_answer: str | None = Field(
        description=(
            "Expected answer if available from an official solution. "
            "None if not yet populated — filled in a later pipeline stage."
        ),
        default=None,
    )
    rubric: str | None = Field(
        description=(
            "Correction rubric for this item. Either extracted from "
            "an official solution key or generated by LLM. "
            "None if not yet populated."
        ),
        default=None,
    )
    rubric_is_generated: bool = Field(
        description=(
            "Whether the rubric was generated by LLM rather than "
            "extracted from an official source. Generated rubrics "
            "require human review before use in evaluation."
        ),
        default=False,
    )


class GradedQuestion(ExtractedQuestion):
    """ExtractedQuestion with items promoted to GradedQuestionItem."""

    items: list[GradedQuestionItem] = Field(
        description=(
            "Evaluable items enriched with answer and rubric data. "
            "Single-element list for standalone questions."
        )
    )
