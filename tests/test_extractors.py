from __future__ import annotations

import base64
import json

import httpx
import pytest

from doc_agent.extractor import (
    DEFAULT_MODEL,
    ExtractorConfigError,
    ExtractorError,
    MockExtractorClient,
    OpenRouterNemotronExtractor,
)
from doc_agent.extractor.prompts import build_user_prompt, infer_document_type, parse_json_payload
from doc_agent.pipeline import DocumentPipeline, ModelPricing
from doc_agent.samples import SAMPLE_PAYLOADS, render_sample_image
from doc_agent.schemas import CostSource, DocumentType

# --- Mock ----------------------------------------------------------------------------------


def test_mock_is_deterministic_and_records_calls(png_bytes):
    mock = MockExtractorClient()
    first = mock.extract(png_bytes, "image/png", DocumentType.VEHICLE)
    second = mock.extract(png_bytes, "image/png", DocumentType.VEHICLE)
    assert first.payload == second.payload == SAMPLE_PAYLOADS[DocumentType.VEHICLE]
    assert first.model == "mock/deterministic"
    assert json.loads(first.raw_text or "") == first.payload
    assert mock.calls == [(DocumentType.VEHICLE, "image/png", len(png_bytes))] * 2


def test_mock_returns_copies(png_bytes):
    mock = MockExtractorClient()
    mock.extract(png_bytes, "image/png", DocumentType.INVOICE).payload["items"].clear()
    assert mock.extract(png_bytes, "image/png", DocumentType.INVOICE).payload["items"]


def test_mock_custom_response_and_error(png_bytes):
    mock = MockExtractorClient({DocumentType.PRESCRIPTION: {"patient_name": "X"}})
    assert mock.extract(png_bytes, "image/png", DocumentType.PRESCRIPTION).payload == {
        "patient_name": "X"
    }
    failing = MockExtractorClient(error=ExtractorError("boom"))
    with pytest.raises(ExtractorError, match="boom"):
        failing.extract(png_bytes, "image/png", DocumentType.PRESCRIPTION)


# --- Prompts / parsing ---------------------------------------------------------------------


@pytest.mark.parametrize("doc_type", list(DocumentType))
def test_user_prompt_embeds_json_schema(doc_type):
    prompt = build_user_prompt(doc_type)
    assert "JSON Schema" in prompt
    assert '"properties"' in prompt


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '<think>raciocínio {"b": 2}</think>\n{"a": 1}',
        'raciocínio sem abertura {"b": 2}</think>\n{"a": 1}',
        '<think>raciocínio truncado {"b": 2} sem fechar tag\n{"a": 1}',
        'Aqui está o resultado: {"a": 1} espero ter ajudado',
    ],
)
def test_parse_json_payload_tolerates_noise(text):
    assert parse_json_payload(text) == {"a": 1}


@pytest.mark.parametrize("text", ["sem json", "[1, 2]", "{quebrado: }", "} {"])
def test_parse_json_payload_rejects_invalid(text):
    with pytest.raises(ExtractorError):
        parse_json_payload(text)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"document_type": "prescription"}, DocumentType.PRESCRIPTION),
        ({"document_type": "VEHICLE"}, DocumentType.VEHICLE),
        ({"od": {"sphere": -1.0}, "patient_name": "Ana"}, DocumentType.PRESCRIPTION),
        ({"plate": "BRA2E19", "renavam": "00639884962"}, DocumentType.VEHICLE),
        ({"access_key": "3526...", "total_amount": "100,00"}, DocumentType.INVOICE),
        ({"foo": "bar"}, None),
        ({}, None),
    ],
)
def test_infer_document_type(payload, expected):
    assert infer_document_type(payload) is expected


# --- OpenRouter ----------------------------------------------------------------------------


def completion(content, model=DEFAULT_MODEL, finish_reason="stop"):
    return {
        "model": model,
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def make_extractor(handler, **kwargs):
    client = httpx.Client(
        base_url="https://openrouter.test/api/v1", transport=httpx.MockTransport(handler)
    )
    return OpenRouterNemotronExtractor(
        api_key="sk-test", client=client, sleep=lambda _: None, **kwargs
    )


def test_openrouter_sends_multimodal_request(png_bytes):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers["authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=completion('{"plate": "ABC1234"}'))

    result = make_extractor(handler).extract(png_bytes, "image/png", DocumentType.VEHICLE)

    assert result.payload == {"plate": "ABC1234"}
    assert result.model == DEFAULT_MODEL == "nvidia/nemotron-3.5-super-vl"
    assert result.usage["prompt_tokens"] == 10
    assert captured["url"].endswith("/chat/completions")
    assert captured["auth"] == "Bearer sk-test"
    body = captured["body"]
    assert body["model"] == DEFAULT_MODEL
    assert body["temperature"] == 0
    assert body["response_format"] == {"type": "json_object"}
    text_part, image_part = body["messages"][1]["content"]
    assert "CRLV" in text_part["text"]
    expected_uri = "data:image/png;base64," + base64.b64encode(png_bytes).decode()
    assert image_part == {"type": "image_url", "image_url": {"url": expected_uri}}


def test_openrouter_model_override_via_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")
    monkeypatch.setenv("DOC_AGENT_MODEL", "nvidia/outro-modelo")
    with OpenRouterNemotronExtractor() as extractor:
        assert extractor.model_name == "nvidia/outro-modelo"


def test_openrouter_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ExtractorConfigError):
        OpenRouterNemotronExtractor()


def test_openrouter_retries_transient_errors(png_bytes):
    responses = iter(
        [
            httpx.Response(429, json={"error": "rate limit"}),
            httpx.Response(503),
            httpx.Response(200, json=completion('{"a": 1}')),
        ]
    )
    sleeps: list[float] = []
    client = httpx.Client(
        base_url="https://x", transport=httpx.MockTransport(lambda r: next(responses))
    )
    extractor = OpenRouterNemotronExtractor(api_key="k", client=client, sleep=sleeps.append)
    assert extractor.extract(png_bytes, "image/png", DocumentType.INVOICE).payload == {"a": 1}
    assert sleeps == [0.5, 1.0]


def test_openrouter_respects_retry_after_header(png_bytes):
    responses = iter(
        [
            httpx.Response(429, headers={"Retry-After": "2.5"}),
            httpx.Response(200, json=completion('{"a": 1}')),
        ]
    )
    sleeps: list[float] = []
    client = httpx.Client(
        base_url="https://x", transport=httpx.MockTransport(lambda r: next(responses))
    )
    extractor = OpenRouterNemotronExtractor(api_key="k", client=client, sleep=sleeps.append)
    assert extractor.extract(png_bytes, "image/png", DocumentType.INVOICE).payload == {"a": 1}
    assert sleeps == [2.5]


def test_openrouter_retries_http_200_upstream_errors(png_bytes):
    responses = iter(
        [
            httpx.Response(200, json={"error": {"message": "Provider overloaded", "code": 502}}),
            httpx.Response(200, json=completion('{"a": 1}', finish_reason="error")),
            httpx.Response(200, json=completion('{"a": 1}')),
        ]
    )
    extractor = make_extractor(lambda r: next(responses))
    assert extractor.extract(png_bytes, "image/png", DocumentType.INVOICE).payload == {"a": 1}


def test_openrouter_does_not_retry_http_200_client_error(png_bytes):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"error": {"message": "Bad request", "code": 400}})

    with pytest.raises(ExtractorError, match="400"):
        make_extractor(handler).extract(png_bytes, "image/png", DocumentType.INVOICE)
    assert len(calls) == 1


def test_openrouter_retries_transport_errors_then_gives_up(png_bytes):
    def handler(request):
        raise httpx.ConnectError("offline", request=request)

    with pytest.raises(ExtractorError, match="3 tentativas"):
        make_extractor(handler).extract(png_bytes, "image/png", DocumentType.INVOICE)


def test_openrouter_does_not_retry_client_errors(png_bytes):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(401, text="invalid key")

    with pytest.raises(ExtractorError, match="401"):
        make_extractor(handler).extract(png_bytes, "image/png", DocumentType.INVOICE)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json=completion("")),
        httpx.Response(200, json=completion(None)),
        httpx.Response(200, text="<html>"),
        httpx.Response(200, json=completion("não é json")),
    ],
)
def test_openrouter_malformed_responses(png_bytes, response):
    with pytest.raises(ExtractorError):
        make_extractor(lambda r: response).extract(png_bytes, "image/png", DocumentType.INVOICE)


def test_openrouter_accepts_content_parts(png_bytes):
    content = [{"type": "text", "text": '{"a":'}, {"type": "text", "text": " 1}"}]
    result = make_extractor(lambda r: httpx.Response(200, json=completion(content))).extract(
        png_bytes, "image/png", DocumentType.INVOICE
    )
    assert result.payload == {"a": 1}


def test_openrouter_cost_reaches_envelope_as_provider_cost():
    body = completion(json.dumps(SAMPLE_PAYLOADS[DocumentType.VEHICLE]))
    # Formato do OpenRouter: usage.cost (em créditos = USD) vem em toda resposta.
    body["usage"] = {
        "prompt_tokens": 1200,
        "completion_tokens": 90,
        "total_tokens": 1290,
        "cost": 0.000417,
    }
    extractor = make_extractor(lambda _: httpx.Response(200, json=body))
    pipeline = DocumentPipeline(extractor, pricing=ModelPricing(99.0, 99.0))  # deve ser ignorado

    usage = pipeline.run(render_sample_image(DocumentType.VEHICLE), DocumentType.VEHICLE).usage
    assert usage.cost_source is CostSource.PROVIDER
    assert usage.estimated_cost_usd == pytest.approx(0.000417)
    assert usage.total_tokens == 1290
