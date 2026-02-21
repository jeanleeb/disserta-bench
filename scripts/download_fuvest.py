"""Download FUVEST second-phase exam PDFs from the official archive.

Scrapes ``fuvest.br/acervo-vestibular-{year}/`` to find the correct PDFs
based on link text within the "Segunda Fase" section.  Falls back to static
URL patterns when the archive page is unavailable.

Downloads all distinct PDFs found (multiple days, subjects, answer keys),
organizing them by year with standardized filenames.

Only uses the Python standard library.

Usage:
    python scripts/download_fuvest.py
    python scripts/download_fuvest.py --years 2020 2021 2022
    python scripts/download_fuvest.py --years 1997-2025 --force
"""

from __future__ import annotations

import argparse
import logging
import re
import ssl
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

LOG = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).resolve().parent.parent / "raw_pdfs" / "fuvest"
QUESTIONS_DIR = _BASE_DIR / "questoes"
ANSWERS_DIR = _BASE_DIR / "gabaritos"

DELAY_SECONDS = 1.0
USER_AGENT = "disserta-bench/0.1 (academic research)"

SSL_CONTEXT = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)
SSL_CONTEXT.check_hostname = True
SSL_CONTEXT.verify_mode = ssl.CERT_REQUIRED
SSL_CONTEXT.minimum_version = ssl.TLSVersion.TLSv1_2

_SECTION_HEADERS = re.compile(
    r"(primeira\s+fase|segunda\s+fase|listas?|estat[ií]stica"
    r"|document|question[aá]rio)",
    re.IGNORECASE,
)

_DAY_RE = re.compile(r"(\d)\s*[º°]?\s*dia", re.IGNORECASE)

# Tokens that indicate specific-skills exams (artes, música, etc.)
# These are excluded from downloads.
_SPECIFIC_SKILLS_RE = re.compile(
    r"artes[_-]?visuais|artes[_-]?cenicas|artes[_-]?c[eê]nicas"
    r"|musica|m[uú]sica|prova[_-]?pratica|habilidades[_-]?espec"
    r"|escrita[_-]?artes",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Fallback static patterns (used when the archive page is unreachable)
# ---------------------------------------------------------------------------

# Each entry is (url_pattern, standardized_name).
# Patterns sharing the same name are mirrors and will be tried in order.
_WP = "https://www.fuvest.br/wp-content/uploads"

_DAY_FILE_NAME_PREFIX = "dia"

FALLBACK_QUESTIONS: list[tuple[str, str]] = [
    (f"{_WP}/fuvest_{{year}}_prova_2fase_fis.pdf", "fisica.pdf"),
    (f"{_WP}/fuvest_{{year}}_2fase_dia2.pdf", f"{_DAY_FILE_NAME_PREFIX}2.pdf"),
    (f"{_WP}/fuvest_{{year}}_2fase_dia3.pdf", f"{_DAY_FILE_NAME_PREFIX}3.pdf"),
    (f"{_WP}/fuv{{year}}_2fase_dia_2.pdf", f"{_DAY_FILE_NAME_PREFIX}2.pdf"),
    (f"{_WP}/fuv{{year}}_2fase_dia2.pdf", f"{_DAY_FILE_NAME_PREFIX}2.pdf"),
    (f"{_WP}/fuvest_{{year}}_segunda_fase_dia_2.pdf", f"{_DAY_FILE_NAME_PREFIX}2.pdf"),
    (
        f"{_WP}/fuvest{{year}}_segunda_fase_prova_2dia.pdf",
        f"{_DAY_FILE_NAME_PREFIX}2.pdf",
    ),
    (f"{_WP}/fuvest{{year}}_prova_2fase_dia2.pdf", f"{_DAY_FILE_NAME_PREFIX}2.pdf"),
    (f"{_WP}/fuvest{{year}}-fase2-dia2-prova.pdf", f"{_DAY_FILE_NAME_PREFIX}2.pdf"),
]

_ANSWERS_FILE_NAME_PREFIX = "guia_respostas"

FALLBACK_ANSWERS: list[tuple[str, str]] = [
    (
        f"{_WP}/fuvest_{{year}}_guia_respostas.pdf",
        f"{_ANSWERS_FILE_NAME_PREFIX}.pdf",
    ),
    (
        f"{_WP}/fuvest_{{year}}_{{year}}.01.22_RESPOSTAS_USP_Guia.pdf",
        f"{_ANSWERS_FILE_NAME_PREFIX}.pdf",
    ),
    (
        f"{_WP}/fuvest{{year}}_abordagens_esperadas_2fase.pdf",
        f"{_ANSWERS_FILE_NAME_PREFIX}.pdf",
    ),
    (
        f"{_WP}/fuvest{{year}}-fase2-dia2-respostas-esperadas-fisica.pdf",
        f"{_ANSWERS_FILE_NAME_PREFIX}_fisica.pdf",
    ),
    (
        f"{_WP}/fuvest{{year}}-fase2-dia2-respostas-esperadas.pdf",
        f"{_ANSWERS_FILE_NAME_PREFIX}.pdf",
    ),
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ArchiveLinks:
    """Links discovered from the archive page for a single year.

    Each entry is a (url, standardized_name) tuple.  URLs sharing the same
    standardized name are mirrors of the same file.
    """

    questions: list[tuple[str, str]] = field(default_factory=list)
    answers: list[tuple[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    """Convert text to an ASCII slug: lowercase, underscores, no accents."""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = nfkd.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_text.lower())
    return slug.strip("_")


def _normalize_link_text(text: str) -> str:
    """Derive a standardized filename from a link's visible text.

    Examples:
        "2º Dia"                         -> "dia2.pdf"
        "Física"                         -> "fisica.pdf"
        "Abordagens Esperadas da 2ª Fase" -> "abordagens_esperadas_da_2_fase.pdf"
    """
    day_match = _DAY_RE.search(text)
    if day_match:
        return f"dia{day_match.group(1)}.pdf"
    return _slugify(text) + ".pdf"


_PHYSICS_RE = re.compile(r"f[ií]sica", re.IGNORECASE)


def _normalize_answer_name(link_text: str, href: str = "") -> str:
    """Standardize answer key filenames to use a consistent prefix.

    Returns ``guia_respostas_fisica.pdf`` when the link text or href
    mentions physics, otherwise ``guia_respostas.pdf``.
    """
    combined = f"{link_text} {href}"
    if _PHYSICS_RE.search(combined):
        return f"{_ANSWERS_FILE_NAME_PREFIX}_fisica.pdf"
    return f"{_ANSWERS_FILE_NAME_PREFIX}.pdf"


# ---------------------------------------------------------------------------
# HTML Parser — section-aware link extraction
# ---------------------------------------------------------------------------

_ANSWER_TEXT_RE = re.compile(
    r"abordagen|respostas?\s+esperadas?|gabarito.*fase", re.IGNORECASE
)


class _ArchivePageParser(HTMLParser):
    """Parse a FUVEST archive page and classify PDF links by section context.

    Tracks the current section ("primeira fase", "segunda fase", etc.)
    by watching for header-like text in any tag.  Within "segunda fase",
    links are classified as questions or answers based on their visible text,
    and specific-skills exams (artes, música, etc.) are excluded.
    """

    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.result = ArchiveLinks()

        self._section: str = ""
        self._current_href: str | None = None
        self._link_text_parts: list[str] = []
        self._in_link = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            for key, value in attrs:
                if key.lower() == "href" and value:
                    self._current_href = value
                    self._link_text_parts = []
                    self._in_link = True
                    return

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._in_link:
            self._flush_link()

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text:
            return

        if self._in_link:
            self._link_text_parts.append(text)

        match = _SECTION_HEADERS.search(text)
        if match:
            self._section = match.group(0).strip().lower()

    def _flush_link(self) -> None:
        href = self._current_href
        link_text = " ".join(self._link_text_parts).strip()
        self._in_link = False
        self._current_href = None
        self._link_text_parts = []

        if not href or not link_text:
            return
        if not href.lower().endswith(".pdf"):
            return
        if "segunda" not in self._section:
            return

        # Skip specific-skills exams.
        combined = f"{link_text} {href}"
        if _SPECIFIC_SKILLS_RE.search(combined):
            LOG.debug("  archive: skip specific-skills %r -> %s", link_text, href)
            return

        url = urljoin(self.base_url, href)

        if _ANSWER_TEXT_RE.search(link_text):
            name = _normalize_answer_name(link_text, href)
            self.result.answers.append((url, name))
            LOG.debug("  archive: answer  %r -> %s (%s)", link_text, url, name)
        else:
            name = _normalize_link_text(link_text)
            self.result.questions.append((url, name))
            LOG.debug("  archive: question %r -> %s (%s)", link_text, url, name)


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------


def _make_request(url: str) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/pdf,text/html,*/*;q=0.8",
        },
    )


def _fetch_bytes(url: str) -> tuple[bytes, str]:
    """Fetch URL and return (payload, content_type)."""
    req = _make_request(url)
    with urllib.request.urlopen(req, timeout=30, context=SSL_CONTEXT) as resp:
        content_type = resp.headers.get("Content-Type", "")
        return resp.read(), content_type


def _read_text(url: str) -> str | None:
    try:
        payload, _ = _fetch_bytes(url)
        return payload.decode("utf-8", errors="ignore")
    except (urllib.error.URLError, TimeoutError):
        return None


def _is_valid_pdf(payload: bytes, content_type: str) -> bool:
    if len(payload) < 256:
        return False
    if b"%PDF-" not in payload[:1024]:
        return False
    if "text/html" in content_type.lower():
        return False
    return True


def _download(url: str, dest: Path) -> bool:
    try:
        payload, content_type = _fetch_bytes(url)
        if not _is_valid_pdf(payload, content_type):
            LOG.debug("  %s -> not a valid PDF", url)
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        return True
    except (urllib.error.URLError, TimeoutError) as exc:
        LOG.debug("  %s -> %s", url, exc)
        return False


# ---------------------------------------------------------------------------
# Archive scraping
# ---------------------------------------------------------------------------


def scrape_archive(year: int) -> ArchiveLinks | None:
    """Scrape the official FUVEST archive page for *year*.

    Returns None if the page cannot be fetched.
    """
    base_url = f"https://www.fuvest.br/acervo-vestibular-{year}/"
    html = _read_text(base_url)
    if not html:
        return None

    parser = _ArchivePageParser(base_url)
    parser.feed(html)
    return parser.result


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


class _HrefParser(HTMLParser):
    """Extract href values from anchor tags."""

    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.hrefs.append(value)
                return


def _extract_pdf_links(base_url: str, html: str) -> list[str]:
    """Extract all PDF link hrefs from *html*, resolved against *base_url*."""
    parser = _HrefParser()
    parser.feed(html)
    urls: list[str] = []
    for href in parser.hrefs:
        if href.startswith("mailto:") or href.startswith("javascript:"):
            continue
        absolute = urljoin(base_url, href)
        if ".pdf" in absolute.lower():
            urls.append(absolute)
    return urls


def _group_by_name(
    items: list[tuple[str, str]],
) -> dict[str, list[str]]:
    """Group (url, standardized_name) pairs by name, merging mirrors."""
    groups: dict[str, list[str]] = {}
    for url, name in items:
        groups.setdefault(name, []).append(url)
    return groups


# ---------------------------------------------------------------------------
# Download logic
# ---------------------------------------------------------------------------


def _download_groups(
    grouped: dict[str, list[str]],
    dest_dir: Path,
    *,
    force: bool,
) -> list[Path]:
    """Download each filename group, trying mirror URLs in order.

    Returns list of paths that were successfully downloaded (or already existed).
    """
    downloaded: list[Path] = []
    for filename, mirrors in grouped.items():
        dest = dest_dir / filename
        if dest.exists() and not force:
            LOG.info("  SKIP %s (already exists)", dest.name)
            downloaded.append(dest)
            continue

        ok = False
        for url in mirrors:
            LOG.info("  GET  %s", url)
            if _download(url, dest):
                LOG.info("  OK   -> %s", dest.name)
                downloaded.append(dest)
                ok = True
                break
            time.sleep(DELAY_SECONDS)

        if not ok:
            LOG.warning("  FAIL %s: no working mirror", filename)

    return downloaded


def download_year(year: int, *, force: bool = False) -> dict[str, int]:
    """Download question and answer PDFs for a single *year*.

    Downloads every distinct PDF found under "Segunda Fase" on the archive
    page, with standardized filenames organized into year subdirectories.
    Mirror URLs (same file on different hosts) are grouped and tried in order.
    """
    LOG.info("Year %d", year)

    archive = scrape_archive(year)

    if archive and (archive.questions or archive.answers):
        LOG.info(
            "  archive: %d questions, %d answers",
            len(archive.questions),
            len(archive.answers),
        )
        q_items = archive.questions
        a_items = archive.answers
    else:
        LOG.info("  archive page unavailable, using fallback patterns")
        q_items = [(p.format(year=year), name) for p, name in FALLBACK_QUESTIONS]
        a_items = [(p.format(year=year), name) for p, name in FALLBACK_ANSWERS]

    q_groups = _group_by_name(q_items)
    q_dir = QUESTIONS_DIR / str(year)
    q_downloaded = _download_groups(q_groups, q_dir, force=force)

    a_groups = _group_by_name(a_items)
    a_dir = ANSWERS_DIR / str(year)
    a_downloaded = _download_groups(a_groups, a_dir, force=force)

    time.sleep(DELAY_SECONDS)
    return {"questions": len(q_downloaded), "answers": len(a_downloaded)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_years(raw: list[str]) -> list[int]:
    """Parse year arguments, supporting both ``2020`` and ``2020-2025``."""
    years: list[int] = []
    for token in raw:
        if "-" in token:
            lo, hi = token.split("-", 1)
            years.extend(range(int(lo), int(hi) + 1))
        else:
            years.append(int(token))
    years = [y for y in years if 1990 <= y <= 2100]
    return sorted(set(years))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download FUVEST second-phase exam PDFs.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=["1997-2025"],
        help="Years to download (e.g. 2020 2021 or 1997-2025). Default: 1997-2025",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    years = parse_years(args.years)
    LOG.info("Downloading FUVEST exams for years: %s", years)

    results: dict[int, dict[str, int]] = {}
    for year in years:
        results[year] = download_year(year, force=args.force)

    LOG.info("--- Summary ---")
    for year, counts in results.items():
        LOG.info(
            "  %d  questions=%-3d  answers=%d",
            year,
            counts["questions"],
            counts["answers"],
        )


if __name__ == "__main__":
    main()
