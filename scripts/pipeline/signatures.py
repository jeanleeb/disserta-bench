from dspy import InputField, OutputField, Signature

from scripts.pipeline.models import (
    AIExtractedQuestion,
    ExamPageType,
    ExtractedAnswerSet,
)


class ClassifyPage(Signature):
    """Classify an exam page as containing questions, solutions, or other content.

    'Other' covers: cover pages, instructions, blank pages, formula sheets,
    and any page that does not contain actual exam questions or their solutions.
    """

    page_markdown: str = InputField(
        desc="Markdown of a single exam page, with equations in LaTeX"
    )
    page_type: ExamPageType = OutputField(
        desc="Type of content on the page. "
        "'questions': contains exam questions. "
        "'solutions': contains official answers or solutions. "
        "'other': cover, instructions, blank, or unrelated content."
    )


class IndexPageQuestions(Signature):
    """List all question identifiers on an exam page, in order of appearance."""

    page_markdown: str = InputField(
        desc="Markdown of a single exam page, equations in LaTeX"
    )
    question_numbers: list[str] = OutputField(
        desc=(
            "Question identifiers exactly as printed, in order of appearance. "
            "Prefixed (e.g. 'F01', 'Q03') or generic (e.g. 'Questão 1', '3'). "
            "Empty list if no questions are found."
            "Do not include sub-item labels (e.g. 'a)', 'b)', 'i.')."
        ),
    )


class ExtractQuestion(Signature):
    """Extract a single question from an exam page.

    Separate shared preamble from individual sub-items. Each sub-item
    carries its own topic based on the concept it specifically tests.
    For questions without lettered sub-items, produce a single item
    with label=None.
    """

    page_markdown: str = InputField(
        desc="Markdown of the exam page, equations in LaTeX"
    )
    question_number: str = InputField(
        desc="Identifier of the question to extract, exactly as printed"
    )
    question_data: AIExtractedQuestion = OutputField()


class ExtractAnswers(Signature):
    """Extract expected answers and any grading criteria for a single
    question from official solution text."""

    solution_text: str = InputField(
        desc=(
            "Markdown of the official solution page(s) for this exam. "
            "May contain solutions for multiple questions — extract only the "
            "one matching question_number."
        )
    )
    question_number: str = InputField(
        desc="Question identifier to extract answers for, e.g. 'Questão 3', 'F01'."
    )
    answers: ExtractedAnswerSet = OutputField(
        desc=(
            "Answers and rubric for each sub-item of this question. "
            "Must match the sub-item structure of question_text. "
            "Extract only what is explicitly stated — set fields to "
            "None when the solution does not provide them."
        )
    )


class GenerateRubric(Signature):
    """Create a step-by-step correction rubric for a physics question item,
    suitable for grading a student's handwritten solution."""

    question_preamble: str | None = InputField(
        desc=(
            "Shared preamble common to all sub-items: context, data, "
            "figure references. None if there is no shared preamble, "
            "full question text if there are no sub-items."
        )
    )
    item_text: str = InputField(
        desc=(
            "Full text of this sub-item or standalone question, "
            "with LaTeX for equations."
        )
    )
    expected_answer: str = InputField(
        desc="The correct final answer, including units when applicable."
    )
    reference_data: list[str] = InputField(
        desc=(
            "Given constants and/or simplifications from the problem "
            "statement, e.g. ['g = 10 m/s²', 'ignore air resistance']. "
            "The rubric must accept solutions that use these values. "
            "May be empty."
        )
    )
    rubric: str = OutputField(
        desc=(
            "Correction rubric in Portuguese. Include: the essential "
            "reasoning steps, key intermediate results for calculations, "
            "the expected final value or explanation, acceptable tolerance "
            "for numerical answers, and common errors to watch for."
        )
    )
