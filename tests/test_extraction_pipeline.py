from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.extraction_pipeline import (
    PipelineConfig,
    _is_physics_relevant,
    _pick_answer_pdf,
    dicts_to_examples,
    run_pipeline,
)
from scripts.llm import extract_questions_with_llm
from scripts.ocr import pdf_to_markdown

# ---------------------------------------------------------------------------
# Cache tests (per-page caching in scripts.ocr)
# ---------------------------------------------------------------------------


def _fake_ocr_page_fn(pages: list[str]):
    """Return a closure that yields successive page texts."""
    it = iter(pages)

    def _ocr(img_bytes: bytes, *, page_num: int, total: int) -> str:
        return next(it)

    return _ocr


def test_cache_key_uses_full_path_context(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    pdf_a = tmp_path / "run_a" / "exam.pdf"
    pdf_b = tmp_path / "run_b" / "exam.pdf"

    pdf_a.parent.mkdir(parents=True, exist_ok=True)
    pdf_b.parent.mkdir(parents=True, exist_ok=True)
    pdf_a.write_bytes(b"%PDF-1.4")
    pdf_b.write_bytes(b"%PDF-1.4")

    call_count = 0

    def fake_ocr_fn():
        nonlocal call_count
        call_count += 1
        return _fake_ocr_page_fn([f"markdown-{call_count}"])

    with (
        patch("scripts.ocr._pdf_to_images", return_value=[b"png"]),
        patch("scripts.ocr._ocr_page_fn", side_effect=fake_ocr_fn),
    ):
        first_a = pdf_to_markdown(pdf_a, cache_dir, overwrite=False)
        first_b = pdf_to_markdown(pdf_b, cache_dir, overwrite=False)
        second_a = pdf_to_markdown(pdf_a, cache_dir, overwrite=False)

    assert first_a == "markdown-1"
    assert first_b == "markdown-2"
    assert second_a == "markdown-1"  # cached
    assert call_count == 2

    cached_files = sorted(cache_dir.glob("exam-*.md"))
    assert len(cached_files) == 2
    assert cached_files[0].name != cached_files[1].name


def test_cache_is_invalidated_when_pdf_changes_in_place(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    pdf_path = tmp_path / "exam.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 original")

    call_count = 0

    def fake_ocr_fn():
        nonlocal call_count
        call_count += 1
        return _fake_ocr_page_fn([f"markdown-v{call_count}"])

    with (
        patch("scripts.ocr._pdf_to_images", return_value=[b"png"]),
        patch("scripts.ocr._ocr_page_fn", side_effect=fake_ocr_fn),
    ):
        first = pdf_to_markdown(pdf_path, cache_dir, overwrite=False)
        pdf_path.write_bytes(b"%PDF-1.4 updated with new content")
        second = pdf_to_markdown(pdf_path, cache_dir, overwrite=False)

    assert first == "markdown-v1"
    assert second == "markdown-v2"
    assert call_count == 2

    cached_files = sorted(cache_dir.glob("exam-*.md"))
    assert len(cached_files) == 2


def test_per_page_cache_resumes_after_crash(tmp_path) -> None:
    """If OCR crashes mid-PDF, re-running should skip already-cached pages."""
    cache_dir = tmp_path / "cache"
    pdf_path = tmp_path / "exam.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    ocr_calls: list[int] = []

    def crash_on_page_2(img_bytes: bytes, *, page_num: int, total: int) -> str:
        ocr_calls.append(page_num)
        if page_num == 2:
            raise RuntimeError("Simulated crash")
        return f"page-{page_num}-text"

    # First run: crashes on page 2
    with (
        patch("scripts.ocr._pdf_to_images", return_value=[b"p1", b"p2", b"p3"]),
        patch("scripts.ocr._ocr_page_fn", return_value=crash_on_page_2),
        pytest.raises(RuntimeError, match="Simulated crash"),
    ):
        pdf_to_markdown(pdf_path, cache_dir, overwrite=False)

    assert ocr_calls == [1, 2]

    # Second run: page 1 cached, page 2 and 3 need OCR
    ocr_calls.clear()

    def succeed_all(img_bytes: bytes, *, page_num: int, total: int) -> str:
        ocr_calls.append(page_num)
        return f"page-{page_num}-text"

    with (
        patch("scripts.ocr._pdf_to_images", return_value=[b"p1", b"p2", b"p3"]),
        patch("scripts.ocr._ocr_page_fn", return_value=succeed_all),
    ):
        result = pdf_to_markdown(pdf_path, cache_dir, overwrite=False)

    # Page 1 was cached, only 2 and 3 needed OCR
    assert ocr_calls == [2, 3]
    assert "page-1-text" in result
    assert "page-2-text" in result
    assert "page-3-text" in result


# ---------------------------------------------------------------------------
# LLM extraction tests (scripts.llm)
# ---------------------------------------------------------------------------


def test_extract_questions_fails_without_gemini_api_key(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")

    with pytest.raises(OSError, match="GOOGLE_API_KEY"):
        extract_questions_with_llm("questions md", "answers md")


def test_extract_questions_defaults_to_gemini(monkeypatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    with patch("scripts.llm._call_gemini", return_value="[]") as mock:
        result = extract_questions_with_llm("questions md", "answers md")

    mock.assert_called_once()
    assert result == []


def test_extract_questions_uses_ollama_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    with patch("scripts.llm._call_ollama", return_value="[]") as mock:
        result = extract_questions_with_llm("questions md", "answers md")

    mock.assert_called_once()
    assert result == []


def test_extract_questions_rejects_unknown_provider(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "unknown")

    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        extract_questions_with_llm("questions md", "answers md")


def test_extract_questions_retries_on_invalid_json(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MAX_RETRIES", "3")

    with patch(
        "scripts.llm._call_ollama",
        side_effect=["not json at all", "still not json", '[{"question_number":"Q1"}]'],
    ) as mock:
        result = extract_questions_with_llm("q", "a")

    assert mock.call_count == 3
    assert len(result) == 1


def test_extract_questions_raises_after_max_retries(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")

    with (
        patch("scripts.llm._call_ollama", return_value="not json"),
        pytest.raises(ValueError, match="not valid JSON"),
    ):
        extract_questions_with_llm("q", "a")


# ---------------------------------------------------------------------------
# OCR provider tests (scripts.ocr)
# ---------------------------------------------------------------------------


def test_ocr_defaults_to_gemini(monkeypatch) -> None:
    monkeypatch.delenv("OCR_PROVIDER", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    with patch("scripts.ocr._make_gemini_ocr") as mock_maker:
        mock_maker.return_value = lambda img, *, page_num, total: "page text"
        from scripts.ocr import _ocr_page_fn

        fn = _ocr_page_fn()
        result = fn(b"png", page_num=1, total=1)

    mock_maker.assert_called_once()
    assert result == "page text"


def test_ocr_uses_ollama_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("OCR_PROVIDER", "ollama")

    with patch("scripts.ocr._make_ollama_ocr") as mock_maker:
        mock_maker.return_value = lambda img, *, page_num, total: "page text"
        from scripts.ocr import _ocr_page_fn

        fn = _ocr_page_fn()
        result = fn(b"png", page_num=1, total=1)

    mock_maker.assert_called_once()
    assert result == "page text"


def test_ocr_gemini_fails_without_api_key(monkeypatch) -> None:
    monkeypatch.setenv("OCR_PROVIDER", "gemini")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    from scripts.ocr import _make_gemini_ocr

    with pytest.raises(OSError, match="GOOGLE_API_KEY"):
        _make_gemini_ocr()


def test_ocr_rejects_unknown_provider(monkeypatch) -> None:
    monkeypatch.setenv("OCR_PROVIDER", "unknown")

    from scripts.ocr import _ocr_page_fn

    with pytest.raises(ValueError, match="Unknown OCR_PROVIDER"):
        _ocr_page_fn()


# ---------------------------------------------------------------------------
# Pipeline integration tests
# ---------------------------------------------------------------------------


def test_run_pipeline_skips_api_key_check_for_ollama(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OCR_PROVIDER", "ollama")

    config = PipelineConfig(
        vestibular="TEST",
        year=2024,
        questions_pdf=tmp_path / "q.pdf",
    )

    with pytest.raises(Exception) as exc_info:
        run_pipeline(config)
    assert "GOOGLE_API_KEY" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# Schema conversion tests
# ---------------------------------------------------------------------------


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


def test_dicts_to_examples_handles_none_expected_value() -> None:
    raw_questions = [
        {
            "question_number": "Q2",
            "topic": "Optics",
            "question": "Describe the phenomenon.",
            "reference_data": {"constants": []},
            "expected_value": None,
            "expected_unit": "",
            "solution_steps": "",
            "rubric": [],
            "has_figure": False,
            "figure_description": "",
        },
    ]

    examples = dicts_to_examples(raw_questions, vestibular="TEST", year=2024)

    assert len(examples) == 1
    assert examples[0].expected_value is None


# ---------------------------------------------------------------------------
# Physics relevance + answer picking tests
# ---------------------------------------------------------------------------


def test_is_physics_relevant_accepts_dia2_and_above() -> None:
    assert _is_physics_relevant("dia2.pdf")
    assert _is_physics_relevant("dia3.pdf")
    assert _is_physics_relevant("Dia2.pdf")


def test_is_physics_relevant_rejects_dia1() -> None:
    assert not _is_physics_relevant("dia1.pdf")


def test_is_physics_relevant_accepts_fisica() -> None:
    assert _is_physics_relevant("fisica.pdf")
    assert _is_physics_relevant("Fisica.pdf")


def test_is_physics_relevant_rejects_other_subjects() -> None:
    assert not _is_physics_relevant("quimica.pdf")
    assert not _is_physics_relevant("biologia.pdf")
    assert not _is_physics_relevant("matematica.pdf")


def test_pick_answer_pdf_prefers_physics(tmp_path: Path) -> None:
    general = tmp_path / "guia_respostas.pdf"
    physics = tmp_path / "guia_respostas_fisica.pdf"
    general.touch()
    physics.touch()
    assert _pick_answer_pdf([general, physics]) == physics


def test_pick_answer_pdf_falls_back_to_first(tmp_path: Path) -> None:
    general = tmp_path / "guia_respostas.pdf"
    general.touch()
    assert _pick_answer_pdf([general]) == general


def test_pick_answer_pdf_returns_none_for_empty() -> None:
    assert _pick_answer_pdf([]) is None
