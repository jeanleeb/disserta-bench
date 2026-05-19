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
from scripts.splitter import filter_by_prefix, split_questions

# ---------------------------------------------------------------------------
# Splitter tests (scripts.splitter)
# ---------------------------------------------------------------------------

_SAMPLE_QUESTIONS = """\
Some preamble text.

---

F01

Question about ice melting.

Note e adote:
Density = 0.92 g/cm^3

---

[35]
PROVA
FUVEST 2024
0
1
2
3
4
5
-
-

---

F02

Question about water filters.

---

**E05**

Question about Born-Oppenheimer.

---

M01

Math question, should be ignored for physics.
"""

_SAMPLE_ANSWERS = """\
F01 DEGELO DO ÁRTICO

a) Solution for F01...

---

F02 FILTRO DE TORNEIRA

a) Solution for F02...

---

**E05** BORN-OPPENHEIMER

a) Solution for F05...

---

M01 MOBILE MATEMÁTICO

a) Solution for M01...
"""


def test_split_questions_extracts_all_codes() -> None:
    result = split_questions(_SAMPLE_QUESTIONS)
    assert result.is_coded is True
    assert "F01" in result.sections
    assert "F02" in result.sections
    assert "E05" in result.sections  # kept as-is, no correction
    assert "M01" in result.sections


def test_split_questions_strips_junk() -> None:
    result = split_questions(_SAMPLE_QUESTIONS)
    assert "PROVA" not in result.sections["F01"]
    assert "FUVEST 2024" not in result.sections["F01"]


def test_split_questions_preserves_content() -> None:
    result = split_questions(_SAMPLE_QUESTIONS)
    assert "ice melting" in result.sections["F01"]
    assert "Density = 0.92" in result.sections["F01"]
    assert "water filters" in result.sections["F02"]


def test_split_questions_preserves_original_codes() -> None:
    result = split_questions(_SAMPLE_QUESTIONS)
    assert "E05" in result.sections  # no renaming, kept as OCR produced it
    assert "Born-Oppenheimer" in result.sections["E05"]


def test_split_questions_empty_input() -> None:
    assert split_questions("").sections == {}
    assert split_questions("No question codes here.").sections == {}


def test_filter_by_prefix_returns_only_prefix() -> None:
    result = split_questions(_SAMPLE_QUESTIONS)
    physics = filter_by_prefix(result.sections, "F")
    assert set(physics.keys()) == {"F01", "F02"}
    assert "M01" not in physics


def test_split_answers() -> None:
    result = split_questions(_SAMPLE_ANSWERS)
    assert "F01" in result.sections
    assert "F02" in result.sections
    assert "E05" in result.sections  # kept as-is
    assert "M01" in result.sections


# --- Generic "Questão NN" format tests ---

_SAMPLE_GENERIC = """\
FUVEST 2017
2ª Fase

Questão 01

A question about thermodynamics...

Questão 02

A question about kinematics...

Questão 03

A question about optics...
"""


def test_split_generic_format() -> None:
    result = split_questions(_SAMPLE_GENERIC)
    assert result.is_coded is False
    assert "01" in result.sections
    assert "02" in result.sections
    assert "03" in result.sections


def test_split_generic_preserves_content() -> None:
    result = split_questions(_SAMPLE_GENERIC)
    assert "thermodynamics" in result.sections["01"]
    assert "kinematics" in result.sections["02"]


def test_split_coded_takes_priority() -> None:
    """When both formats exist, coded (F01) takes priority."""
    mixed = """\
Questão 01
Some preamble

F01 DEGELO
The real question about ice...

F02 FILTROS
Another question...
"""
    result = split_questions(mixed)
    assert result.is_coded is True
    assert "F01" in result.sections
    assert "F02" in result.sections


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

# Markdown with a single physics question for LLM tests.
_LLM_TEST_QUESTIONS = "F01\n\nA test physics question."
_LLM_TEST_ANSWERS = "F01\n\nSolution for the test question."
_VALID_JSON_RESPONSE = '{"question_number":"F01","topic":"Mechanics","question":"test"}'


def test_extract_questions_fails_without_gemini_api_key(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")

    with pytest.raises(OSError, match="GOOGLE_API_KEY"):
        extract_questions_with_llm(_LLM_TEST_QUESTIONS, _LLM_TEST_ANSWERS)


def test_extract_questions_defaults_to_gemini(monkeypatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    with patch("scripts.llm._call_gemini", return_value=_VALID_JSON_RESPONSE) as mock:
        result = extract_questions_with_llm(_LLM_TEST_QUESTIONS, _LLM_TEST_ANSWERS)

    mock.assert_called_once()
    assert len(result) == 1
    assert result[0]["question_number"] == "F01"


def test_extract_questions_uses_ollama_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    with patch("scripts.llm._call_ollama", return_value=_VALID_JSON_RESPONSE) as mock:
        result = extract_questions_with_llm(_LLM_TEST_QUESTIONS, _LLM_TEST_ANSWERS)

    mock.assert_called_once()
    assert len(result) == 1


def test_extract_questions_rejects_unknown_provider(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "unknown")

    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        extract_questions_with_llm(_LLM_TEST_QUESTIONS, _LLM_TEST_ANSWERS)


def test_extract_questions_retries_on_invalid_json(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MAX_RETRIES", "3")

    with patch(
        "scripts.llm._call_ollama",
        side_effect=["not json", "still not json", _VALID_JSON_RESPONSE],
    ) as mock:
        result = extract_questions_with_llm(_LLM_TEST_QUESTIONS, _LLM_TEST_ANSWERS)

    assert mock.call_count == 3
    assert len(result) == 1


def test_extract_questions_skips_after_max_retries(monkeypatch) -> None:
    """Failed questions are skipped (not raised), pipeline continues."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")

    with patch("scripts.llm._call_ollama", return_value="not json"):
        result = extract_questions_with_llm(_LLM_TEST_QUESTIONS, _LLM_TEST_ANSWERS)

    assert result == []


def test_extract_questions_caches_results(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    with patch("scripts.llm._call_ollama", return_value=_VALID_JSON_RESPONSE) as mock:
        result1 = extract_questions_with_llm(
            _LLM_TEST_QUESTIONS, _LLM_TEST_ANSWERS,
            vestibular="FUVEST", year=2024, cache_dir=tmp_path,
        )
        result2 = extract_questions_with_llm(
            _LLM_TEST_QUESTIONS, _LLM_TEST_ANSWERS,
            vestibular="FUVEST", year=2024, cache_dir=tmp_path,
        )

    # LLM called only once; second call uses cache.
    assert mock.call_count == 1
    assert len(result1) == 1
    assert result1 == result2

    # Cache file exists.
    cache_file = tmp_path / "llm" / "fuvest_2024_F01.json"
    assert cache_file.exists()


def test_extract_questions_processes_all_subjects(monkeypatch) -> None:
    """LLM extraction no longer filters by prefix — all questions are processed."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    math_response = (
        '{"question_number":"M01","subject":"math",'
        '"topic":"Algebra","question":"test"}'
    )
    with patch("scripts.llm._call_ollama", return_value=math_response) as mock:
        result = extract_questions_with_llm(
            "M01\n\nMath question only.", "M01\n\nMath answer."
        )

    mock.assert_called_once()
    assert len(result) == 1
    assert result[0]["subject"] == "math"


def test_extract_questions_returns_empty_for_empty_markdown(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    result = extract_questions_with_llm("No question codes here.", "")
    assert result == []


def test_extract_questions_processes_multiple_subjects(monkeypatch) -> None:
    """All question codes are processed, regardless of prefix."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    responses = [
        '{"question_number":"F01","subject":"physics","topic":"Mechanics","question":"q1"}',
        '{"question_number":"M01","subject":"math","topic":"Algebra","question":"q2"}',
    ]

    with patch("scripts.llm._call_ollama", side_effect=responses) as mock:
        result = extract_questions_with_llm(
            "F01\n\nPhysics question.\n\nM01\n\nMath question.",
            "F01\n\nAnswer 1.\n\nM01\n\nAnswer 2.",
        )

    assert mock.call_count == 2
    assert len(result) == 2
    subjects = {r["subject"] for r in result}
    assert subjects == {"physics", "math"}


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


def test_run_pipeline_filters_physics_only(tmp_path, monkeypatch) -> None:
    """run_pipeline should only return physics questions after LLM extraction."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OCR_PROVIDER", "ollama")

    mixed_results = [
        {
            "question_number": "F01",
            "subject": "physics",
            "topic": "Mechanics",
            "question": "Physics q.",
            "reference_data": {"constants": []},
            "expected_answers": [
                {"label": "a", "value": "120", "unit": "N", "explanation": "F=PA"},
            ],
            "solution_steps": "",
            "rubric": [],
            "has_figure": False,
            "figure_description": "",
        },
        {
            "question_number": "M01",
            "subject": "math",
            "topic": "Algebra",
            "question": "Math q.",
            "reference_data": {"constants": []},
            "expected_answers": [],
            "solution_steps": "",
            "rubric": [],
            "has_figure": False,
            "figure_description": "",
        },
    ]

    with (
        patch("scripts.extraction_pipeline.pdf_to_markdown", return_value="mock md"),
        patch(
            "scripts.extraction_pipeline.extract_questions_with_llm",
            return_value=mixed_results,
        ),
    ):
        config = PipelineConfig(
            vestibular="TEST",
            year=2024,
            questions_pdf=tmp_path / "q.pdf",
        )
        examples = run_pipeline(config)

    assert len(examples) == 1
    assert examples[0].question_number == "F01"


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
            "reference_data": {"constants": [
                {"symbol": "g", "value": "9.8", "unit": "m/s^2"},
            ]},
            "expected_answers": [
                {"label": "a", "value": "9.8", "unit": "m/s^2",
                 "explanation": "F=ma"},
            ],
            "solution_steps": "Use F=ma.",
            "rubric": [],
            "has_figure": False,
            "figure_description": "",
        },
    ]

    examples = dicts_to_examples(raw_questions, vestibular="TEST", year=2024)

    assert len(examples) == 1
    assert examples[0].question_number == "Q1"
    assert examples[0].reference_data.constants[0].value == "9.8"
    assert examples[0].expected_answers[0].value == "9.8"


def test_dicts_to_examples_handles_empty_expected_answers() -> None:
    raw_questions = [
        {
            "question_number": "Q2",
            "topic": "Optics",
            "question": "Describe the phenomenon.",
            "reference_data": {"constants": []},
            "expected_answers": [],
            "solution_steps": "",
            "rubric": [],
            "has_figure": False,
            "figure_description": "",
        },
    ]

    examples = dicts_to_examples(raw_questions, vestibular="TEST", year=2024)

    assert len(examples) == 1
    assert examples[0].expected_answers == []


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
