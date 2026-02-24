from dspy import InputField, OutputField, Signature

from scripts.pipeline.models import AIExtractedQuestion, ExamPageType


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
