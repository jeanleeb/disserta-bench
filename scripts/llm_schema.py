"""
Pydantic models for LLM structured output.

These models are passed as ``response_schema`` to the Gemini API so that the
model is forced to return JSON conforming to this exact structure.  Field
descriptions serve double duty: they guide the model *and* document the schema.

NOTE: This module intentionally uses Pydantic (a transitive dependency of
``google-genai``) rather than the stdlib dataclasses used in
``dataset_schema.py``, which is kept dependency-free for downstream consumers.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Valid topics per subject, based on Brazilian high-school curriculum.
VALID_TOPICS: dict[str, list[str]] = {
    "physics": [
        "Kinematics",
        "Dynamics",
        "Energy and Work",
        "Rotational Mechanics",
        "Fluid Mechanics",
        "Thermodynamics",
        "Waves",
        "Optics",
        "Electrostatics",
        "Electrodynamics",
        "Electromagnetism",
        "Modern Physics",
    ],
    "math": [
        "Algebra",
        "Functions",
        "Trigonometry",
        "Analytic Geometry",
        "Plane Geometry",
        "Spatial Geometry",
        "Combinatorics and Probability",
        "Sequences and Series",
        "Complex Numbers and Polynomials",
    ],
    "chemistry": [
        "General Chemistry",
        "Atomic Structure",
        "Chemical Bonding",
        "Stoichiometry",
        "Solutions",
        "Thermochemistry",
        "Chemical Kinetics",
        "Chemical Equilibrium",
        "Electrochemistry",
        "Organic Chemistry",
        "Biochemistry",
    ],
    "biology": [
        "Cell Biology",
        "Genetics",
        "Evolution",
        "Ecology",
        "Human Physiology",
        "Plant Physiology",
        "Zoology",
        "Microbiology",
    ],
    "geography": [
        "Physical Geography",
        "Human Geography",
        "Brazilian Geography",
        "Geopolitics",
        "Environmental Geography",
    ],
    "history": [
        "Ancient History",
        "Medieval History",
        "Modern History",
        "Contemporary History",
        "Brazilian Colonial History",
        "Brazilian Imperial and Republican History",
    ],
    "portuguese": [
        "Literature",
        "Grammar",
        "Text Interpretation",
    ],
}


def _build_topic_description() -> str:
    """Build a Field description listing all valid topics per subject."""
    parts = []
    for subject, topics in VALID_TOPICS.items():
        parts.append(f"{subject}: {', '.join(topics)}")
    return "Must be one of the following per subject — " + "; ".join(parts)


class ConstantSchema(BaseModel):
    """A physical constant or given value from the question statement."""

    symbol: str
    value: str = Field(
        description=(
            "Numeric value as a string. PRESERVE scientific notation "
            "(e.g. '1.6e-19', '3e8'). Never expand into decimal form."
        ),
    )
    unit: str


class ReferenceDataSchema(BaseModel):
    """Data provided in the question statement (constants, given values)."""

    constants: list[ConstantSchema] = []


class ExpectedAnswerSchema(BaseModel):
    """One distinct expected result from the question.

    Each entry represents a single numerical or qualitative result,
    regardless of whether the question has sub-items (a, b, c) or not.
    """

    label: str = Field(
        description=(
            "Short description of the result in Brazilian Portuguese "
            "(e.g. 'massa de gelo', 'espessura da camada', 'distância h')."
        ),
    )
    value: str | None = Field(
        default=None,
        description="Numeric answer as string, or null for qualitative answers.",
    )
    unit: str = ""
    explanation: str = Field(
        default="",
        description="Brief expected reasoning for this result.",
    )


class QuestionExtractionSchema(BaseModel):
    """Schema for a single extracted exam question."""

    question_number: str
    subject: str = Field(
        description=(
            "One of: physics, math, chemistry, biology, "
            "geography, history, portuguese"
        ),
    )
    topic: str = Field(description=_build_topic_description())
    question: str = Field(
        description=(
            "Full question text in Brazilian Portuguese. "
            "Preserve inline LaTeX."
        ),
    )
    reference_data: ReferenceDataSchema = ReferenceDataSchema()
    expected_answers: list[ExpectedAnswerSchema] = []
    solution_steps: str = Field(
        default="",
        description="Full step-by-step solution from the answer key, preserving LaTeX.",
    )
    rubric: list[str] = Field(
        default_factory=list,
        description="3-5 specific evaluation criteria in Brazilian Portuguese.",
    )
    has_figure: bool = False
    figure_description: str = ""
