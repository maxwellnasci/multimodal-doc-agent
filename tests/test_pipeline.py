from __future__ import annotations

import io
import json
from itertools import count

import pytest
from PIL import Image

from doc_agent.extractor import ExtractorError, MockExtractorClient
from doc_agent.pipeline import (
    DocumentPipeline,
    ModelPricing,
    build_token_usage,
    completeness,
    detect_image_mime,
    preprocess_image,
    score_confidence,
)
from doc_agent.samples import render_sample_image
from doc_agent.schemas import (
    CostSource,
    DocumentType,
    ExtractionEnvelope,
    ExtractionStatus,
    GuardrailIssue,
    InvoiceData,
    PrescriptionData,
    Severity,
    VehicleDocumentData,
)

EXPECTED_SCHEMA = {
    DocumentType.PRESCRIPTION: PrescriptionData,
    DocumentType.VEHICLE: VehicleDocumentData,
    DocumentType.INVOICE: InvoiceData,
    DocumentType.AUTO: InvoiceData,
}


def fake_clock(step: float = 0.125):
    ticks = count()
    return lambda: next(ticks) * step


@pytest.mark.parametrize("doc_type", list(DocumentType))
def test_happy_path_for_every_document_type(doc_type):
    pipeline = DocumentPipeline(MockExtractorClient(), clock=fake_clock())
    envelope = pipeline.run(render_sample_image(doc_type), doc_type)

    assert envelope.status is ExtractionStatus.OK
    assert envelope.confidence == 1.0
    assert envelope.issues == []
    assert isinstance(envelope.data, EXPECTED_SCHEMA[doc_type])
    assert envelope.latency_ms == 125.0
    assert envelope.model == "mock/deterministic"
    assert envelope.usage.total_tokens == 330
    # Mock não informa custo e nenhum preço foi configurado: nada de custo inventado.
    assert envelope.usage.estimated_cost_usd is None
    assert envelope.usage.cost_source is CostSource.UNKNOWN
    assert envelope.raw_response is not None


@pytest.mark.parametrize(
    ("sample_type", "expected_cls"),
    [
        (DocumentType.PRESCRIPTION, PrescriptionData),
        (DocumentType.VEHICLE, VehicleDocumentData),
        (DocumentType.INVOICE, InvoiceData),
    ],
)
def test_auto_resolves_each_document_type(sample_type, expected_cls):
    pipeline = DocumentPipeline(MockExtractorClient(auto_type=sample_type))
    envelope = pipeline.run(render_sample_image(sample_type), DocumentType.AUTO)
    assert envelope.status is ExtractionStatus.OK
    assert envelope.document_type is sample_type
    assert isinstance(envelope.data, expected_cls)


def test_auto_validates_only_the_data_of_the_declared_type(png_bytes, invoice_payload):
    # Campo de outro tipo dentro de `data` não vaza: o schema da nota ignora `od`.
    response = {"document_type": "invoice", "data": {**invoice_payload, "od": {"sphere": -1}}}
    envelope = DocumentPipeline(MockExtractorClient({DocumentType.AUTO: response})).run(
        png_bytes, DocumentType.AUTO
    )
    assert envelope.status is ExtractionStatus.OK
    assert envelope.document_type is DocumentType.INVOICE
    assert isinstance(envelope.data, InvoiceData)
    assert envelope.raw_response == response


def test_auto_reports_schema_errors_of_the_declared_type(png_bytes):
    response = {"document_type": "vehicle_document", "data": {"manufacture_year": "abc"}}
    envelope = DocumentPipeline(MockExtractorClient({DocumentType.AUTO: response})).run(
        png_bytes, DocumentType.AUTO
    )
    assert envelope.status is ExtractionStatus.FAILED
    assert envelope.document_type is DocumentType.VEHICLE
    assert [i.field for i in envelope.issues] == ["manufacture_year"]


def test_auto_fails_when_document_type_cannot_be_inferred(png_bytes):
    mock = MockExtractorClient({DocumentType.AUTO: {"unknown_field": "valor"}})
    envelope = DocumentPipeline(mock).run(png_bytes, DocumentType.AUTO)
    assert envelope.status is ExtractionStatus.FAILED
    assert [i.code for i in envelope.issues] == ["unknown_document_type"]


def test_preprocess_image_downscales_large_images():
    large = Image.new("RGBA", (2400, 1800), (20, 40, 80, 255))
    buf = io.BytesIO()
    large.save(buf, format="JPEG" if large.mode == "RGB" else "PNG")
    raw_png = buf.getvalue()

    resized_bytes, mime = preprocess_image(raw_png, "image/png", max_dimension=1200)
    assert mime == "image/png"
    with Image.open(io.BytesIO(resized_bytes)) as img:
        assert img.size == (1200, 900)

    # Também converte RGBA -> RGB quando formato é JPEG
    jpeg_bytes, _ = preprocess_image(raw_png, "image/jpeg", max_dimension=1000)
    with Image.open(io.BytesIO(jpeg_bytes)) as img:
        assert img.size == (1000, 750)
        assert img.mode == "RGB"


@pytest.mark.parametrize(("orientation", "expected_size"), [(6, (533, 1600)), (8, (533, 1600))])
def test_preprocess_image_applies_exif_orientation_before_downscale(orientation, expected_size):
    # Foto de celular: pixels em paisagem + tag EXIF mandando girar 90° (retrato).
    photo = Image.new("RGB", (3000, 1000), "white")
    photo.paste((255, 0, 0), (0, 0, 300, 1000))  # faixa vermelha na borda esquerda
    exif = Image.Exif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    photo.save(buf, format="JPEG", exif=exif)

    out, mime = preprocess_image(buf.getvalue(), "image/jpeg")
    assert mime == "image/jpeg"
    with Image.open(io.BytesIO(out)) as img:
        assert img.size == expected_size
        assert img.getexif().get(0x0112) is None  # orientação já aplicada aos pixels
        top_or_bottom = (img.width // 2, 5 if orientation == 6 else img.height - 5)
        red, green, _ = img.convert("RGB").getpixel(top_or_bottom)
        assert red > 200
        assert green < 80


PRICING = ModelPricing(prompt_usd_per_1m=0.30, completion_usd_per_1m=0.80)


def test_build_token_usage_estimates_cost_from_configured_pricing():
    usage = build_token_usage(
        {"prompt_tokens": 1000, "completion_tokens": 500},
        pricing=PRICING,
        image_bytes_original=50000,
        image_bytes_sent=15000,
    )
    assert usage.total_tokens == 1500
    assert usage.estimated_cost_usd == pytest.approx(0.0007)
    assert usage.cost_source is CostSource.ESTIMATED
    assert usage.image_bytes_original == 50000
    assert usage.image_bytes_sent == 15000


def test_build_token_usage_prefers_provider_cost_over_pricing():
    usage = build_token_usage(
        {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.012345}, pricing=PRICING
    )
    assert usage.estimated_cost_usd == pytest.approx(0.012345)
    assert usage.cost_source is CostSource.PROVIDER


@pytest.mark.parametrize("usage", [None, {}, {"prompt_tokens": 1000, "completion_tokens": 500}])
def test_build_token_usage_without_cost_or_pricing_is_unknown(usage):
    result = build_token_usage(usage)
    assert result.estimated_cost_usd is None
    assert result.cost_source is CostSource.UNKNOWN


def test_build_token_usage_without_usage_does_not_estimate_zero():
    result = build_token_usage(None, pricing=PRICING)
    assert result.estimated_cost_usd is None
    assert result.cost_source is CostSource.UNKNOWN


def test_pipeline_applies_configured_pricing():
    pipeline = DocumentPipeline(MockExtractorClient(), pricing=PRICING)
    envelope = pipeline.run(render_sample_image(DocumentType.INVOICE), DocumentType.INVOICE)
    assert envelope.usage.cost_source is CostSource.ESTIMATED
    assert envelope.usage.estimated_cost_usd == pytest.approx((250 * 0.30 + 80 * 0.80) / 1e6)


def test_envelope_round_trips_through_json():
    pipeline = DocumentPipeline(MockExtractorClient())
    envelope = pipeline.run(render_sample_image(DocumentType.INVOICE), DocumentType.INVOICE)
    dumped = json.loads(envelope.model_dump_json())
    assert dumped["data"]["total_amount"] == pytest.approx(1189.90)
    assert dumped["status"] == "ok"
    restored = ExtractionEnvelope.model_validate_json(envelope.model_dump_json())
    assert restored.confidence == envelope.confidence


def test_guardrail_errors_flag_for_review(png_bytes, vehicle_payload):
    vehicle_payload["chassis"] = "9BWZZZ377VTO04251"
    mock = MockExtractorClient({DocumentType.VEHICLE: vehicle_payload})
    envelope = DocumentPipeline(mock).run(png_bytes, DocumentType.VEHICLE)

    assert envelope.status is ExtractionStatus.NEEDS_REVIEW
    assert envelope.confidence == 0.75
    assert [i.code for i in envelope.errors] == ["invalid_chassis"]
    assert envelope.data is not None


def test_low_completeness_flags_for_review(png_bytes):
    mock = MockExtractorClient({DocumentType.PRESCRIPTION: {"patient_name": "Ana", "od": None}})
    envelope = DocumentPipeline(mock).run(png_bytes, DocumentType.PRESCRIPTION)
    assert envelope.status is ExtractionStatus.NEEDS_REVIEW
    assert envelope.errors == []
    assert envelope.confidence == pytest.approx(0.2 - 2 * 0.05)


def test_review_threshold_is_configurable(png_bytes, prescription_payload):
    prescription_payload["patient_name"] = None  # completude 0.8
    mock = MockExtractorClient({DocumentType.PRESCRIPTION: prescription_payload})
    strict = DocumentPipeline(mock, review_threshold=0.9).run(png_bytes, DocumentType.PRESCRIPTION)
    lenient = DocumentPipeline(mock, review_threshold=0.8).run(png_bytes, DocumentType.PRESCRIPTION)
    assert strict.status is ExtractionStatus.NEEDS_REVIEW
    assert lenient.status is ExtractionStatus.OK


def test_schema_violation_fails(png_bytes):
    mock = MockExtractorClient({DocumentType.INVOICE: {"issue_date": "ontem", "items": "x"}})
    envelope = DocumentPipeline(mock).run(png_bytes, DocumentType.INVOICE)
    assert envelope.status is ExtractionStatus.FAILED
    assert envelope.confidence == 0.0
    assert envelope.data is None
    assert {i.field for i in envelope.issues} == {"issue_date", "items"}
    assert all(i.code.startswith("schema_") for i in envelope.issues)
    assert envelope.raw_response == {"issue_date": "ontem", "items": "x"}


def test_extractor_error_fails(png_bytes):
    mock = MockExtractorClient(error=ExtractorError("timeout"))
    envelope = DocumentPipeline(mock).run(png_bytes, DocumentType.VEHICLE)
    assert envelope.status is ExtractionStatus.FAILED
    assert [(i.code, i.message) for i in envelope.issues] == [("extractor_error", "timeout")]


@pytest.mark.parametrize(
    ("image", "code"),
    [
        (b"", "empty_image"),
        (b"%PDF-1.7 ...", "unsupported_image"),
        (b"\x89PNG\r\n\x1a\n" + b"\x00" * 2048, "image_too_large"),
    ],
)
def test_invalid_images_fail_before_calling_extractor(image, code):
    mock = MockExtractorClient()
    envelope = DocumentPipeline(mock, max_image_bytes=1024).run(image, DocumentType.INVOICE)
    assert envelope.status is ExtractionStatus.FAILED
    assert [i.code for i in envelope.issues] == [code]
    assert mock.calls == []


@pytest.mark.parametrize(
    ("header", "mime"),
    [
        (b"\x89PNG\r\n\x1a\n", "image/png"),
        (b"\xff\xd8\xff\xe0", "image/jpeg"),
        (b"GIF89a", "image/gif"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp"),
        (b"RIFF\x00\x00\x00\x00WAVE", None),
        (b"hello", None),
    ],
)
def test_detect_image_mime(header, mime):
    assert detect_image_mime(header) == mime


def test_mime_is_forwarded_to_extractor():
    mock = MockExtractorClient()
    DocumentPipeline(mock).run(b"\xff\xd8\xff\xe0" + b"\x00" * 10, DocumentType.INVOICE)
    assert mock.calls[0][1] == "image/jpeg"


def test_completeness_counts_nested_and_list_fields():
    assert completeness(InvoiceData()) == 0.0
    assert completeness(InvoiceData(issuer_name="A", items=[{"description": "x"}])) == 0.4
    assert completeness(PrescriptionData(od={"sphere": 0})) == 0.2


def test_confidence_is_clamped():
    issue = GuardrailIssue(field="x", code="c", message="m", severity=Severity.ERROR)
    assert score_confidence(InvoiceData(), [issue] * 10) == 0.0
