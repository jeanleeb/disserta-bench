from scripts import download_fuvest as mod


def test_extract_pdf_links_resolves_relative_urls() -> None:
    html = """
    <html><body>
      <a href="prova.pdf">Prova</a>
      <a href="/foo/bar/gabarito.PDF">Gabarito</a>
      <a href="mailto:test@example.com">Mail</a>
    </body></html>
    """
    links = mod._extract_pdf_links(
        "https://acervo.fuvest.br/fuvest/2024/", html
    )
    assert "https://acervo.fuvest.br/fuvest/2024/prova.pdf" in links
    assert "https://acervo.fuvest.br/foo/bar/gabarito.PDF" in links
    assert len(links) == 2


def test_is_valid_pdf_rejects_html_payload() -> None:
    html_bytes = b"<html><body>not found</body></html>" * 20
    assert not mod._is_valid_pdf(html_bytes, "text/html")


def test_normalize_link_text_day() -> None:
    assert mod._normalize_link_text("2º Dia") == "dia2.pdf"
    assert mod._normalize_link_text("3° Dia") == "dia3.pdf"
    assert mod._normalize_link_text("1 dia") == "dia1.pdf"


def test_normalize_link_text_subject() -> None:
    assert mod._normalize_link_text("Física") == "fisica.pdf"
    assert mod._normalize_link_text("Matemática") == "matematica.pdf"


def test_group_by_name_merges_mirrors() -> None:
    items = [
        ("https://acervo.fuvest.br/fuvest/2024/prova_dia2.pdf", "dia2.pdf"),
        ("https://www.fuvest.br/wp-content/uploads/prova_dia2.pdf", "dia2.pdf"),
        ("https://www.fuvest.br/wp-content/uploads/prova_dia3.pdf", "dia3.pdf"),
    ]
    grouped = mod._group_by_name(items)
    assert list(grouped.keys()) == ["dia2.pdf", "dia3.pdf"]
    assert len(grouped["dia2.pdf"]) == 2


def test_download_groups_tries_fallback_url(monkeypatch, tmp_path) -> None:
    grouped = mod._group_by_name(
        [
            ("https://acervo.fuvest.br/fuvest/2024/prova_dia2.pdf", "dia2.pdf"),
            ("https://www.fuvest.br/wp-content/uploads/prova_dia2.pdf", "dia2.pdf"),
            ("https://www.fuvest.br/wp-content/uploads/prova_dia3.pdf", "dia3.pdf"),
        ]
    )

    calls: list[str] = []

    def fake_download(url: str, dest) -> bool:  # type: ignore[no-untyped-def]
        calls.append(url)
        if "acervo.fuvest.br" in url:
            return False
        dest.write_bytes(b"%PDF-1.7 fake payload" + b"x" * 300)
        return True

    monkeypatch.setattr(mod, "_download", fake_download)
    monkeypatch.setattr(mod, "DELAY_SECONDS", 0.0)

    downloaded = mod._download_groups(grouped, tmp_path, force=True)

    assert len(downloaded) == 2
    assert (tmp_path / "dia2.pdf").exists()
    assert (tmp_path / "dia3.pdf").exists()
    assert any("acervo.fuvest.br" in url for url in calls)
    assert any("wp-content/uploads" in url for url in calls)


def test_normalize_answer_name_default() -> None:
    assert mod._normalize_answer_name("Abordagens Esperadas") == "guia_respostas.pdf"
    assert mod._normalize_answer_name("Gabarito") == "guia_respostas.pdf"


def test_normalize_answer_name_physics() -> None:
    assert (
        mod._normalize_answer_name("Física") == "guia_respostas_fisica.pdf"
    )
    assert (
        mod._normalize_answer_name("Gabarito", href="fisica.pdf")
        == "guia_respostas_fisica.pdf"
    )


def test_specific_skills_exams_are_excluded() -> None:
    html = """
    <html><body>
      <p>Segunda Fase</p>
      <a href="prova_dia2.pdf">2º Dia</a>
      <a href="artes-visuais.pdf">Artes Visuais</a>
      <a href="prova-musica-sp.pdf">Música</a>
      <a href="fisica.pdf">Física</a>
    </body></html>
    """
    parser = mod._ArchivePageParser("https://example.com/")
    parser.feed(html)
    urls = [url for url, _ in parser.result.questions]
    assert len(urls) == 2
    assert any("prova_dia2.pdf" in u for u in urls)
    assert any("fisica.pdf" in u for u in urls)
    assert not any("artes" in u for u in urls)
    assert not any("musica" in u for u in urls)
