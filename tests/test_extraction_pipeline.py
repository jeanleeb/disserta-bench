from unittest.mock import patch

import pytest

from scripts.extraction_pipeline import (
    PipelineConfig,
    dicts_to_examples,
    extract_questions_with_llm,
    pdf_to_markdown,
    run_pipeline,
)


def test_cache_key_uses_full_path_context(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    pdf_a = tmp_path / "run_a" / "exam.pdf"
    pdf_b = tmp_path / "run_b" / "exam.pdf"

    pdf_a.parent.mkdir(parents=True, exist_ok=True)
    pdf_b.parent.mkdir(parents=True, exist_ok=True)
    pdf_a.write_bytes(b"%PDF-1.4")
    pdf_b.write_bytes(b"%PDF-1.4")

    with patch(
        "scripts.extraction_pipeline._run_marker",
        side_effect=["markdown-a", "markdown-b"],
    ) as marker_mock:
        first_a = pdf_to_markdown(pdf_a, cache_dir, overwrite=False)
        first_b = pdf_to_markdown(pdf_b, cache_dir, overwrite=False)
        second_a = pdf_to_markdown(pdf_a, cache_dir, overwrite=False)

    assert first_a == "markdown-a"
    assert first_b == "markdown-b"
    assert second_a == "markdown-a"
    assert marker_mock.call_count == 2

    cached_files = sorted(cache_dir.glob("exam-*.md"))
    assert len(cached_files) == 2
    assert cached_files[0].name != cached_files[1].name


def test_cache_is_invalidated_when_pdf_changes_in_place(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    pdf_path = tmp_path / "exam.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 original")

    with patch(
        "scripts.extraction_pipeline._run_marker",
        side_effect=["markdown-v1", "markdown-v2"],
    ) as marker_mock:
        first = pdf_to_markdown(pdf_path, cache_dir, overwrite=False)
        pdf_path.write_bytes(b"%PDF-1.4 updated with new content")
        second = pdf_to_markdown(pdf_path, cache_dir, overwrite=False)

    assert first == "markdown-v1"
    assert second == "markdown-v2"
    assert marker_mock.call_count == 2

    cached_files = sorted(cache_dir.glob("exam-*.md"))
    assert len(cached_files) == 2


def test_extract_questions_fails_without_gemini_api_key(monkeypatch) -> None:
    """extract_questions_with_llm should raise OSError if GOOGLE_API_KEY is missing
    and provider is gemini.
    """
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")

    with pytest.raises(OSError, match="GOOGLE_API_KEY"):
        extract_questions_with_llm("questions md", "answers md")


def test_run_pipeline_skips_api_key_check_for_ollama(tmp_path, monkeypatch) -> None:
    """Pipeline should NOT raise about GOOGLE_API_KEY when using ollama provider."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    config = PipelineConfig(
        vestibular="TEST",
        year=2024,
        questions_pdf=tmp_path / "q.pdf",
        output_path=tmp_path / "out.jsonl",
    )

    # Should fail on Marker (PDF doesn't exist), NOT on API key validation
    with pytest.raises(Exception) as exc_info:
        run_pipeline(config)
    assert "GOOGLE_API_KEY" not in str(exc_info.value)


def test_extract_questions_defaults_to_gemini(monkeypatch) -> None:
    """
    extract_questions_with_llm should default to gemini when LLM_PROVIDER is unset.
    """
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    with patch("scripts.extraction_pipeline._call_gemini", return_value="[]") as mock:
        result = extract_questions_with_llm("questions md", "answers md")

    mock.assert_called_once()
    assert result == []


def test_extract_questions_uses_ollama_when_configured(monkeypatch) -> None:
    """extract_questions_with_llm should use ollama when LLM_PROVIDER=ollama."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    with patch("scripts.extraction_pipeline._call_ollama", return_value="[]") as mock:
        result = extract_questions_with_llm("questions md", "answers md")

    mock.assert_called_once()
    assert result == []


def test_dicts_to_examples_skips_non_dict_items() -> None:
    raw_questions = [
        "bad-item",
        {
            "question_number": "Q1",
            "topic": "Mechanics",
            "question": "Compute acceleration.",
            "reference_data": {"constants": []},
            "expected_value": 9.8,
            "expected_unit": "m/s^2",
            "solution_steps": "Use F=ma.",
            "rubric": [],
            "has_figure": False,
            "figure_description": "",
        },
    ]

    examples = dicts_to_examples(raw_questions, vestibular="TEST", year=2024)

    assert len(examples) == 1
    assert examples[0].question_number == "Q1"
