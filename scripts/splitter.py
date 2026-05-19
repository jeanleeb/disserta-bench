"""
Split OCR-produced Markdown into individual sections by question code.

Supports two formats found in FUVEST exams:

1. **Coded** — ``F01``, ``M02``, ``Q03`` (subject prefix + number).
2. **Generic** — ``Questão 01``, ``Questão 02`` (number only, no subject).

Returns a :class:`SplitResult` with the sections dict and a flag
indicating which format was detected.  Consumers use ``is_coded`` to
decide whether they need the LLM to classify the subject.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Matches codes like F01, M02, **Q03**, **E05** at the start of a line.
_CODED_RE = re.compile(
    r"^(?:\*\*)?([A-Z]\d{2})(?:\*\*)?(?:\s|$)", re.MULTILINE
)

# Matches "Questão 01", "[07] Questão 07", etc.
_GENERIC_RE = re.compile(
    r"^(?:\[?\d+\]?\s*)?[Qq]uest[ãa]o\s+(\d{2})\b", re.MULTILINE
)

# Junk lines commonly produced by OCR from score grids and page headers.
_JUNK_RE = re.compile(
    r"^(?:"
    r"PROVA(?:\s+\d)?"
    r"|FUVEST\s*\d{4}"
    r"|\[\d+\]"
    r"|[0-5]"
    r"|-"
    r")\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class SplitResult:
    """Result of splitting a markdown document into question sections."""

    sections: dict[str, str]
    is_coded: bool  # True → codes like F01; False → codes like 01


def split_questions(
    markdown: str,
    *,
    clean_junk: bool = True,
) -> SplitResult:
    """Split a Markdown document into individual questions.

    Tries the coded format first (``F01``, ``M02``).  If no coded
    questions are found, falls back to the generic format
    (``Questão 01``).

    Args:
        markdown: Full OCR-produced markdown text.
        clean_junk: If True (default), remove common OCR noise lines.

    Returns:
        A :class:`SplitResult` with ``sections`` (code → text) and
        ``is_coded`` indicating which format was detected.
    """
    # Try coded format first.
    matches = list(_CODED_RE.finditer(markdown))
    if matches:
        return SplitResult(
            sections=_extract_sections(markdown, matches, clean_junk),
            is_coded=True,
        )

    # Fallback to generic "Questão NN" format.
    matches = list(_GENERIC_RE.finditer(markdown))
    if matches:
        return SplitResult(
            sections=_extract_sections(markdown, matches, clean_junk),
            is_coded=False,
        )

    return SplitResult(sections={}, is_coded=True)


def _extract_sections(
    markdown: str,
    matches: list[re.Match[str]],
    clean_junk: bool,
) -> dict[str, str]:
    """Extract text sections between regex matches."""
    sections: dict[str, str] = {}
    for i, match in enumerate(matches):
        code = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        text = markdown[start:end]

        if clean_junk:
            text = _JUNK_RE.sub("", text)

        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        if text:
            sections[code] = text

    return sections


def filter_by_prefix(
    sections: dict[str, str], prefix: str
) -> dict[str, str]:
    """Filter sections to only those whose code starts with *prefix*."""
    return {
        code: text
        for code, text in sections.items()
        if code.startswith(prefix)
    }
