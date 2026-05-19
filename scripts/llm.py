"""
LLM providers for structured question extraction.

Splits OCR-produced Markdown into individual questions, then sends each
one to a text LLM (Gemini or Ollama) for structured JSON extraction.
Results are cached per question so re-runs skip already-processed items.

Both providers use native JSON mode (constrained decoding) to guarantee
valid JSON output.  A configurable retry loop handles transient failures.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from pathlib import Path

from scripts.splitter import split_questions

logger = logging.getLogger(__name__)

_SYSTEM_MESSAGE = (
    "You are a JSON extraction assistant. "
    "You MUST respond with a valid JSON object only. "
    "No explanations, no markdown fences, no commentary."
)

_SINGLE_QUESTION_PROMPT = """
You are processing a single question from a Brazilian
university entrance exam (vestibular).

Rules:
- Keep question text in Brazilian Portuguese.
- Write rubric criteria in Brazilian Portuguese.
- Write solution_steps in Brazilian Portuguese.
- Determine the subject from the question content, NOT the code prefix.
- Use ONLY a topic from the predefined list for each subject (see schema).
- For constants, write the value as a STRING preserving scientific
  notation (e.g. "1.6e-27", NOT 0.0000000000000000000000000016).
- Each expected_answers entry represents one distinct result. Use a
  descriptive label in Brazilian Portuguese (e.g. "massa de gelo",
  "distância h"), not just a sub-item letter.
- Preserve inline LaTeX as $...$ or $$...$$.

QUESTION:
{question_text}

ANSWER:
{answer_text}
"""


def extract_questions_with_llm(
    questions_markdown: str,
    answers_markdown: str,
    *,
    vestibular: str = "",
    year: int = 0,
    cache_dir: Path | None = None,
    overwrite_cache: bool = False,
) -> list[dict]:
    """Split markdown and extract each physics question individually.

    Each question is processed by a separate LLM call and cached to disk.
    """
    q_result = split_questions(questions_markdown)
    a_result = split_questions(answers_markdown)
    question_sections = q_result.sections
    answer_sections = a_result.sections

    if not question_sections:
        logger.warning("No questions found in markdown.")
        return []

    logger.info(
        "Found %d questions (%s): %s",
        len(question_sections),
        "coded" if q_result.is_coded else "generic",
        ", ".join(sorted(question_sections)),
    )

    # Set up LLM cache directory.
    llm_cache: Path | None = None
    if cache_dir is not None and vestibular and year:
        llm_cache = cache_dir / "llm"
        llm_cache.mkdir(parents=True, exist_ok=True)

    call_fn = _get_call_fn()

    results: list[dict] = []
    for code in sorted(question_sections):
        question_text = question_sections[code]
        answer_text = answer_sections.get(code, "")
        if not answer_text:
            logger.warning("No answer found for %s, extracting without it.", code)

        result = _extract_single_question(
            code=code,
            question_text=question_text,
            answer_text=answer_text,
            call_fn=call_fn,
            cache_dir=llm_cache,
            vestibular=vestibular,
            year=year,
            overwrite=overwrite_cache,
        )
        if result is not None:
            result["_from_generic"] = not q_result.is_coded
            results.append(result)

    return results


# ---------------------------------------------------------------------------
# Per-question extraction with cache
# ---------------------------------------------------------------------------


def _extract_single_question(
    *,
    code: str,
    question_text: str,
    answer_text: str,
    call_fn: Callable[[str], str],
    cache_dir: Path | None,
    vestibular: str,
    year: int,
    overwrite: bool,
) -> dict | None:
    """Extract a single question, using cache if available."""
    # Check cache.
    cache_path: Path | None = None
    if cache_dir is not None and vestibular and year:
        cache_path = cache_dir / f"{vestibular.lower()}_{year}_{code}.json"
        if cache_path.exists() and not overwrite:
            logger.info("Cache hit for %s", code)
            with cache_path.open(encoding="utf-8") as f:
                return json.load(f)

    prompt = _SINGLE_QUESTION_PROMPT.format(
        question_text=question_text,
        answer_text=answer_text,
    )

    max_retries = int(os.environ.get("LLM_MAX_RETRIES", "2"))
    last_error: ValueError | None = None

    for attempt in range(1, max_retries + 1):
        raw = call_fn(prompt)
        try:
            result = _parse_json_object(raw)
            # Override question_number — the splitter knows the real code.
            result["question_number"] = code
            # Save to cache.
            if cache_path is not None:
                cache_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            logger.info("Extracted %s successfully.", code)
            return result
        except ValueError as e:
            last_error = e
            if attempt >= max_retries:
                break
            logger.warning(
                "%s: attempt %d/%d failed, retrying... Response: %.200s",
                code,
                attempt,
                max_retries,
                raw[:200],
            )

    logger.error(
        "Failed to extract %s after %d attempts: %s",
        code, max_retries, last_error,
    )
    return None


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------


def _get_call_fn() -> Callable[[str], str]:
    """Return the appropriate LLM call function based on env config."""
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
        return _call_gemini

    return _call_ollama


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------


def _call_ollama(prompt: str) -> str:
    """Call a local Ollama model with structured output via JSON schema."""
    import ollama

    from scripts.llm_schema import QuestionExtractionSchema

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
        format=QuestionExtractionSchema.model_json_schema(),
    )
    return response["message"]["content"]


def _call_gemini(prompt: str) -> str:
    """Call the Gemini API with structured output and return the raw response."""
    from google import genai
    from google.genai import types

    from scripts.llm_schema import QuestionExtractionSchema

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    logger.info("Calling Gemini model: %s", model_name)

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=QuestionExtractionSchema,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    if response.usage_metadata is not None:
        usage = response.usage_metadata
        logger.info(
            "Tokens — input: %d, output: %d, thinking: %d",
            usage.prompt_token_count,
            usage.candidates_token_count,
            getattr(usage, "thoughts_token_count", 0) or 0,
        )
    return response.text or ""


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def _parse_json_object(raw: str) -> dict:
    """Parse a JSON object from an LLM response, stripping markdown fences."""
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

    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object, got {type(result).__name__}.")

    return result
