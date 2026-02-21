"""
LLM providers for structured question extraction.

Sends OCR-produced Markdown to a text LLM (Gemini or Ollama) and parses
the returned JSON array of question dicts.

Both providers use native JSON mode (constrained decoding) to guarantee
valid JSON output.  A configurable retry loop handles transient failures.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable

logger = logging.getLogger(__name__)

_SYSTEM_MESSAGE = (
    "You are a JSON extraction assistant. "
    "You MUST respond with a valid JSON array only. "
    "No explanations, no markdown fences, no commentary."
)

_EXTRACTION_PROMPT = """
You are processing a Brazilian university entrance exam (vestibular) in physics.

Below is the Markdown of the QUESTIONS section followed by the ANSWERS section.
Extract each physics question as a JSON object.

Return a JSON array where each element has exactly these fields:
- question_number: string (e.g. "Q3", "Q5a")
- topic: string (physics subfield in English, e.g. "Thermodynamics")
- question: string (full question text, preserve inline LaTeX as $...$ or $$...$$)
- reference_data: object with "constants" array, each having "symbol", "value", "unit"
- expected_value: number (official numerical answer)
- expected_unit: string (SI unit, e.g. "m/s", "J", "N")
- solution_steps: string (step-by-step solution from the answer key)
- rubric: array of strings (3-5 specific evaluation criteria for this question)
- has_figure: boolean
- figure_description: string (empty if has_figure is false)

Rules:
- Include ONLY questions that have a clear numerical answer.
- Keep question text in Brazilian Portuguese.
- Write rubric criteria in Brazilian Portuguese.
- Write solution_steps in Brazilian Portuguese.
- Return ONLY the JSON array, no markdown fences, no explanation.

QUESTIONS MARKDOWN:
{questions_markdown}

ANSWERS MARKDOWN:
{answers_markdown}
"""


def extract_questions_with_llm(
    questions_markdown: str,
    answers_markdown: str,
) -> list[dict]:
    """Send Markdown content to an LLM and parse the returned JSON array.

    Uses native JSON mode on both providers and retries on parse failure.
    """
    prompt = _EXTRACTION_PROMPT.format(
        questions_markdown=questions_markdown,
        answers_markdown=answers_markdown,
    )

    llm_provider = os.environ.get("LLM_PROVIDER", "gemini")

    if llm_provider not in ("gemini", "ollama"):
        raise ValueError(
            f"Unknown LLM_PROVIDER: {llm_provider!r}. Use 'gemini' or 'ollama'."
        )

    if llm_provider == "gemini":
        if not os.environ.get("GOOGLE_API_KEY"):
            raise OSError(
                "GOOGLE_API_KEY environment variable is not set. "
                "Copy sample.env to .env and fill in your key."
            )
        call_fn: Callable[[str], str] = _call_gemini
    else:
        call_fn = _call_ollama

    max_retries = int(os.environ.get("LLM_MAX_RETRIES", "2"))

    for attempt in range(1, max_retries + 1):
        raw = call_fn(prompt)
        try:
            return _parse_json_response(raw)
        except ValueError:
            if attempt >= max_retries:
                raise
            logger.warning(
                "Attempt %d/%d failed to parse JSON, retrying... "
                "Response starts with: %.200s",
                attempt,
                max_retries,
                raw[:200],
            )

    # Unreachable, but satisfies type checkers.
    raise RuntimeError("Unexpected exit from retry loop")  # pragma: no cover


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------


def _call_ollama(prompt: str) -> str:
    """Call a local Ollama model with JSON mode and return the raw response."""
    import ollama

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "llama3.2")
    logger.info("Calling Ollama model: %s", model)

    num_ctx = int(os.environ.get("OLLAMA_NUM_CTX", "32768"))
    client = ollama.Client(host=host)
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ],
        options={"num_ctx": num_ctx},
        format="json",
    )
    return response["message"]["content"]


def _call_gemini(prompt: str) -> str:
    """Call the Gemini API with JSON mode and return the raw response."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    logger.info("Calling Gemini model: %s", model_name)

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )
    return response.text or ""


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def _parse_json_response(raw: str) -> list[dict]:
    """Parse a JSON array from an LLM response, stripping markdown fences."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"```\s*$", "", cleaned.strip(), flags=re.MULTILINE)
    cleaned = cleaned.strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM response is not valid JSON.\n"
            f"Error: {e}\n"
            f"Response (first 500 chars):\n{raw[:500]}"
        ) from e

    if not isinstance(result, list):
        raise ValueError(f"Expected a JSON array, got {type(result).__name__}.")

    return result
