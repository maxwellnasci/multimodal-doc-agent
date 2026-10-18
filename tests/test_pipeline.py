from __future__ import annotations

import json
from itertools import count

import pytest

from doc_agent.extractor import ExtractorError, MockExtractorClient
from doc_agent.pipeline import DocumentPipeline, completeness, detect_image_mime, score_confidence
from doc_agent.samples import render_sample_image
from doc_agent.schemas import (
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
    assert envelope.raw_response is not None


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
