"""Orquestrador: imagem -> extrator (LLM) -> schema Pydantic -> guardrails -> envelope."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import ValidationError

from doc_agent.extractor import ExtractorClient, ExtractorError
from doc_agent.guardrails import run_guardrails
from doc_agent.schemas import (
    SCHEMA_BY_TYPE,
    DocumentData,
    DocumentType,
    ExtractionEnvelope,
    ExtractionStatus,
    GuardrailIssue,
    Severity,
)

MAX_IMAGE_BYTES = 8 * 1024 * 1024
ERROR_PENALTY = 0.25
WARNING_PENALTY = 0.05
REVIEW_THRESHOLD = 0.8

_MAGIC_NUMBERS: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def detect_image_mime(image: bytes) -> str | None:
    """Detecta o tipo pelo conteúdo (magic bytes), nunca pela extensão do arquivo."""
    for magic, mime in _MAGIC_NUMBERS:
        if image.startswith(magic):
            return mime
    if image[:4] == b"RIFF" and image[8:12] == b"WEBP":
        return "image/webp"
    return None


def completeness(data: DocumentData) -> float:
    """Fração dos campos críticos do schema que foram preenchidos."""
    fields = data.critical_fields
    filled = 0
    for path in fields:
        value: object = data
        for part in path.split("."):
            value = getattr(value, part, None)
            if value is None:
                break
        if value not in (None, "", []):
            filled += 1
    return filled / len(fields)


def score_confidence(data: DocumentData, issues: list[GuardrailIssue]) -> float:
    """Score determinístico: completude dos campos críticos menos penalidades dos guardrails."""
    errors = sum(1 for i in issues if i.severity is Severity.ERROR)
    warnings = len(issues) - errors
    score = completeness(data) - errors * ERROR_PENALTY - warnings * WARNING_PENALTY
    return round(min(1.0, max(0.0, score)), 3)


class DocumentPipeline:
    def __init__(
        self,
        extractor: ExtractorClient,
        *,
        review_threshold: float = REVIEW_THRESHOLD,
        max_image_bytes: int = MAX_IMAGE_BYTES,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._extractor = extractor
        self._review_threshold = review_threshold
        self._max_image_bytes = max_image_bytes
        self._clock = clock

    def run(self, image: bytes, document_type: DocumentType) -> ExtractionEnvelope:
        started = self._clock()

        def envelope(**kwargs: object) -> ExtractionEnvelope:
            return ExtractionEnvelope(
                document_type=document_type,
                model=self._extractor.model_name,
                latency_ms=round((self._clock() - started) * 1000, 2),
                extracted_at=datetime.now(UTC),
                **kwargs,
            )

        def failure(issues: list[GuardrailIssue], **kwargs: object) -> ExtractionEnvelope:
            return envelope(status=ExtractionStatus.FAILED, confidence=0.0, issues=issues, **kwargs)

        input_issue = self._validate_image(image)
        if input_issue is not None:
            return failure([input_issue])
        mime_type = detect_image_mime(image)
        assert mime_type is not None  # garantido por _validate_image

        try:
            result = self._extractor.extract(image, mime_type, document_type)
        except ExtractorError as exc:
            return failure([_issue("image", "extractor_error", str(exc))])

        try:
            data = SCHEMA_BY_TYPE[document_type].model_validate(result.payload)
        except ValidationError as exc:
            issues = [
                _issue(
                    ".".join(str(p) for p in err["loc"]) or "payload",
                    f"schema_{err['type']}",
                    err["msg"],
                )
                for err in exc.errors()
            ]
            return failure(issues, raw_response=result.payload)

        issues = run_guardrails(data)
        confidence = score_confidence(data, issues)
        has_errors = any(i.severity is Severity.ERROR for i in issues)
        status = (
            ExtractionStatus.NEEDS_REVIEW
            if has_errors or confidence < self._review_threshold
            else ExtractionStatus.OK
        )
        return envelope(
            status=status,
            confidence=confidence,
            data=data,
            issues=issues,
            raw_response=result.payload,
        )

    def _validate_image(self, image: bytes) -> GuardrailIssue | None:
        if not image:
            return _issue("image", "empty_image", "imagem vazia")
        if len(image) > self._max_image_bytes:
            limit_mb = self._max_image_bytes / 1024 / 1024
            return _issue("image", "image_too_large", f"imagem excede {limit_mb:.0f} MB")
        if detect_image_mime(image) is None:
            return _issue("image", "unsupported_image", "formato não suportado (PNG/JPEG/WEBP/GIF)")
        return None


def _issue(field: str, code: str, message: str) -> GuardrailIssue:
    return GuardrailIssue(field=field, code=code, message=message, severity=Severity.ERROR)
