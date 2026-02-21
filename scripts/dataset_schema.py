"""
Dataset schema for disserta-bench.

Defines the data contract between the extraction pipeline and any downstream
consumer (DSPy agents, HuggingFace datasets, Argilla, plain JSON, etc.).

This module has zero external dependencies — only the Python standard library.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Constant:
    """A physical constant or given value provided in the question statement."""

    symbol: str
    value: float
    unit: str

    def to_dict(self) -> dict:
        return {"symbol": self.symbol, "value": self.value, "unit": self.unit}


@dataclass
class ReferenceData:
    """
    Data provided in the question statement that both the agent and the judge
    may use. Does not include information that should be retrieved from memory
    (e.g., universal constants not stated in the problem).
    """

    constants: list[Constant] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"constants": [c.to_dict() for c in self.constants]}

    @classmethod
    def from_dict(cls, data: dict) -> ReferenceData:
        constants = [Constant(**c) for c in data.get("constants", [])]
        return cls(constants=constants)


@dataclass
class PhysicsExample:
    """
    A single physics question from a Brazilian university entrance exam.

    Agent inputs (what a solver should receive):
        question, reference_data

    Evaluation labels (for metrics and judges only):
        expected_value, expected_unit, solution_steps, rubric
    """

    # --- Metadata ---
    vestibular: str  # Exam board (e.g., "FUVEST", "ITA", "Unicamp")
    year: int  # Exam year
    question_number: str  # Question identifier (e.g., "Q3", "Q5a")
    topic: str  # Physics subfield (e.g., "Thermodynamics")

    # --- Solver input ---
    question: str  # Full question text with inline LaTeX
    reference_data: ReferenceData = field(default_factory=ReferenceData)

    # --- Evaluation labels ---
    expected_value: float | None = None
    expected_unit: str | None = None
    solution_steps: str | None = None  # Official step-by-step solution
    rubric: list[str] = field(default_factory=list)  # Per-question criteria

    # --- Figure handling ---
    has_figure: bool = False
    figure_description: str = ""

    # --- Review status ---
    needs_review: bool = False

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary (one line in JSONL)."""
        return {
            "vestibular": self.vestibular,
            "year": self.year,
            "question_number": self.question_number,
            "topic": self.topic,
            "question": self.question,
            "reference_data": self.reference_data.to_dict(),
            "expected_value": self.expected_value,
            "expected_unit": self.expected_unit,
            "solution_steps": self.solution_steps,
            "rubric": self.rubric,
            "has_figure": self.has_figure,
            "figure_description": self.figure_description,
            "needs_review": self.needs_review,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PhysicsExample:
        """Deserialize from a dictionary (e.g., one parsed JSONL line)."""
        return cls(
            vestibular=data["vestibular"],
            year=data["year"],
            question_number=data["question_number"],
            topic=data["topic"],
            question=data["question"],
            reference_data=ReferenceData.from_dict(data.get("reference_data", {})),
            expected_value=data.get("expected_value"),
            expected_unit=data.get("expected_unit"),
            solution_steps=data.get("solution_steps"),
            rubric=data.get("rubric", []),
            has_figure=data.get("has_figure", False),
            figure_description=data.get("figure_description", ""),
            needs_review=data.get("needs_review", False),
        )


def load_dataset(path: Path) -> list[PhysicsExample]:
    """
    Load a disserta-bench JSONL file into a list of PhysicsExample objects.

    Args:
        path: Path to a .jsonl file where each line is a JSON object.

    Returns:
        List of PhysicsExample instances, skipping blank lines.
    """
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        examples.append(PhysicsExample.from_dict(json.loads(line)))
    return examples


def save_dataset(examples: list[PhysicsExample], path: Path) -> None:
    """
    Save a list of PhysicsExample objects to a JSONL file.

    Args:
        examples: List of PhysicsExample instances to serialize.
        path:     Destination .jsonl file path (will be created or overwritten).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")


def validate_dataset(examples: list[PhysicsExample]) -> list[str]:
    """
    Run basic quality checks on a list of examples.

    Returns:
        A list of warning messages. Empty list means all checks passed.
    """
    warnings = []
    for ex in examples:
        label = f"[{ex.vestibular} {ex.year} {ex.question_number}]"

        if not ex.question.strip():
            warnings.append(f"{label} Empty question text.")
        if not ex.needs_review:
            if ex.expected_value is None:
                warnings.append(f"{label} Missing expected_value.")
            if not ex.expected_unit:
                warnings.append(f"{label} Missing expected_unit.")
            if not ex.solution_steps:
                warnings.append(f"{label} Missing solution_steps.")
        if not ex.rubric:
            warnings.append(f"{label} No rubric criteria defined.")
        if ex.has_figure and not ex.figure_description:
            warnings.append(f"{label} has_figure=True but figure_description is empty.")

    return warnings
