from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from doc_agent.app import format_cost
from doc_agent.pipeline import detect_image_mime
from doc_agent.samples import _TITLES, _lines, load_sample_font, render_sample_image
from doc_agent.schemas import CostSource, DocumentType, TokenUsage

APP_PATH = Path(__file__).parents[1] / "src" / "doc_agent" / "app.py"


@pytest.mark.parametrize("doc_type", list(DocumentType))
def test_sample_images_are_png(doc_type):
    image = render_sample_image(doc_type)
    assert detect_image_mime(image) == "image/png"
    assert image == render_sample_image(doc_type)


def _glyph(font, char: str) -> bytes:
    from PIL import Image, ImageDraw

    image = Image.new("L", (60, 60), 0)
    ImageDraw.Draw(image).text((10, 5), char, fill=255, font=font)
    return image.tobytes()


@pytest.mark.parametrize("doc_type", [t for t in DocumentType if t is not DocumentType.AUTO])
def test_sample_font_renders_every_accented_character(doc_type):
    # Glifo ausente vira sempre o mesmo "tofu"; acentos do PT-BR não podem cair nele.
    font = load_sample_font(21)
    tofu = _glyph(font, "\U0010fffd")  # plano privado: inexistente na fonte
    text = _TITLES[doc_type] + "".join(_lines(doc_type))
    accented = sorted({c for c in text if not c.isascii()})
    assert accented  # as amostras usam acentos (Ó, ã, ç, Í...)
    assert [c for c in accented if _glyph(font, c) == tofu] == []


@pytest.mark.parametrize("doc_type", [t for t in DocumentType if t is not DocumentType.AUTO])
def test_sample_lines_fit_inside_the_frame(doc_type):
    body, title = load_sample_font(21), load_sample_font(28)
    assert title.getlength(_TITLES[doc_type]) < 1100 - 60
    assert max(body.getlength(line) for line in _lines(doc_type)) < 1100 - 80


@pytest.fixture
def app(monkeypatch) -> AppTest:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    return AppTest.from_file(str(APP_PATH), default_timeout=30).run()


def test_app_renders_with_mock_by_default(app):
    assert not app.exception
    assert app.sidebar.radio[0].value == "Mock determinístico"
    assert app.button[0].label == "Extrair dados"


@pytest.mark.parametrize("doc_type", list(DocumentType))
def test_app_extracts_embedded_example(app, doc_type):
    app.sidebar.selectbox[0].select(doc_type).run()
    app.button[0].click().run()
    assert not app.exception
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Status"] == "✅ Aprovado"
    assert metrics["Confiança"] == "100%"
    assert "Latência" in metrics
    assert metrics["Custo FinOps"] == "—"  # mock não informa custo
    expected_type = DocumentType.INVOICE if doc_type is DocumentType.AUTO else doc_type
    assert app.session_state["envelope"].document_type is expected_type


@pytest.mark.parametrize(
    ("cost", "source", "expected"),
    [
        (0.0012345, CostSource.PROVIDER, "$0.00123"),
        (0.0012345, CostSource.ESTIMATED, "~$0.00123 (estimado)"),
        (None, CostSource.UNKNOWN, "—"),
    ],
)
def test_format_cost_distinguishes_cost_source(cost, source, expected):
    assert format_cost(TokenUsage(estimated_cost_usd=cost, cost_source=source)) == expected


def test_app_hides_previous_result_when_switching_to_auto(app):
    app.sidebar.selectbox[0].select(DocumentType.VEHICLE).run()
    app.button[0].click().run()
    assert app.metric  # resultado do CRLV na tela

    app.sidebar.selectbox[0].select(DocumentType.AUTO).run()
    assert not app.exception
    assert not app.metric  # não reaproveita o resultado de outro tipo

    app.button[0].click().run()
    assert app.metric
    assert app.session_state["envelope"].document_type is DocumentType.INVOICE


def test_app_openrouter_without_key_shows_error(app):
    app.sidebar.radio[0].set_value("OpenRouter · Nemotron VL").run()
    app.button[0].click().run()
    assert not app.exception
    assert "OPENROUTER_API_KEY" in app.error[0].value
