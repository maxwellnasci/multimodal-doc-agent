from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from doc_agent.pipeline import detect_image_mime
from doc_agent.samples import render_sample_image
from doc_agent.schemas import DocumentType

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
    assert app.session_state["envelope"].document_type is doc_type


def test_app_openrouter_without_key_shows_error(app):
    app.sidebar.radio[0].set_value("OpenRouter · Nemotron VL").run()
    app.button[0].click().run()
    assert not app.exception
    assert "OPENROUTER_API_KEY" in app.error[0].value
