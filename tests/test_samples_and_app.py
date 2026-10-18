from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from doc_agent.app import format_cost
from doc_agent.pipeline import detect_image_mime
from doc_agent.samples import render_sample_image
from doc_agent.schemas import CostSource, DocumentType, TokenUsage

APP_PATH = Path(__file__).parents[1] / "src" / "doc_agent" / "app.py"


@pytest.mark.parametrize("doc_type", list(DocumentType))
def test_sample_images_are_png(doc_type):
    image = render_sample_image(doc_type)
    assert detect_image_mime(image) == "image/png"
    assert image == render_sample_image(doc_type)


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


def test_app_openrouter_without_key_shows_error(app):
    app.sidebar.radio[0].set_value("OpenRouter · Nemotron VL").run()
    app.button[0].click().run()
    assert not app.exception
    assert "OPENROUTER_API_KEY" in app.error[0].value
