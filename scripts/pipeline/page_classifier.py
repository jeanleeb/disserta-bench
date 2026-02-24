from dspy import Module, Predict

from scripts.pipeline.models import ExamPageType

from .signatures import ClassifyPage


class PageClassifier(Module):
    def __init__(self) -> None:
        super().__init__()
        self.classify = Predict(ClassifyPage)

    def forward(self, page_markdown: str) -> ExamPageType:
        return self.classify(page_markdown=page_markdown).page_type
