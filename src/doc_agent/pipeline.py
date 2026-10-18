"""Orquestrador: imagem -> extrator (LLM) -> schema Pydantic -> guardrails -> envelope."""

from __future__ import annotations

import io
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from PIL import Image, ImageOps
from pydantic import ValidationError

from doc_agent.extractor import ExtractorClient, ExtractorError
from doc_agent.extractor.prompts import resolve_auto_payload
from doc_agent.guardrails import run_guardrails
from doc_agent.schemas import (
    SCHEMA_BY_TYPE,
    CostSource,
    DocumentData,
    DocumentType,
    ExtractionEnvelope,
    ExtractionStatus,
    GuardrailIssue,
    Severity,
    TokenUsage,
)

MAX_IMAGE_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_IMAGE_DIMENSION = 1600
ERROR_PENALTY = 0.25
WARNING_PENALTY = 0.05
REVIEW_THRESHOLD = 0.8

_MAGIC_NUMBERS: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)

_PILLOW_FORMAT_BY_MIME: dict[str, str] = {
    "image/png": "PNG",
    "image/jpeg": "JPEG",
    "image/webp": "WEBP",
    "image/gif": "GIF",
}


def detect_image_mime(image: bytes) -> str | None:
    """Detecta o tipo pelo conteúdo (magic bytes), nunca pela extensão do arquivo."""
    for magic, mime in _MAGIC_NUMBERS:
        if image.startswith(magic):
            return mime
    if image[:4] == b"RIFF" and image[8:12] == b"WEBP":
        return "image/webp"
    return None


def preprocess_image(
    image: bytes, mime_type: str, *, max_dimension: int = DEFAULT_MAX_IMAGE_DIMENSION
) -> tuple[bytes, str]:
    """Reduz imagens acima de `max_dimension` preservando proporção (economia de visual tokens)."""
    if max_dimension <= 0:
        return image, mime_type
    try:
        with Image.open(io.BytesIO(image)) as img:
            width, height = img.size
            if max(width, height) <= max_dimension:
                return image, mime_type
            # Aplica a orientação EXIF nos pixels: o re-encode abaixo descarta o EXIF e
            # fotos de celular chegariam deitadas ao modelo.
            resized = ImageOps.exif_transpose(img)
    except Exception:
        return image, mime_type

    resized.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    fmt = _PILLOW_FORMAT_BY_MIME.get(mime_type, "PNG")
    if fmt == "JPEG" and resized.mode not in ("RGB", "L"):
        resized = resized.convert("RGB")
    buffer = io.BytesIO()
    save_kwargs: dict[str, Any] = {"optimize": True}
    if fmt in ("JPEG", "WEBP"):
        save_kwargs["quality"] = 85
    resized.save(buffer, format=fmt, **save_kwargs)
    return buffer.getvalue(), mime_type


@dataclass(frozen=True)
class ModelPricing:
    """Preço do modelo em USD por 1M de tokens; usado só quando o provedor não informa o custo."""

    prompt_usd_per_1m: float
    completion_usd_per_1m: float

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.prompt_usd_per_1m + completion_tokens * self.completion_usd_per_1m
        ) / 1_000_000


def build_token_usage(
    usage: Mapping[str, Any] | None,
    *,
    pricing: ModelPricing | None = None,
    image_bytes_original: int | None = None,
    image_bytes_sent: int | None = None,
) -> TokenUsage:
    """Tokens e custo em USD: custo do provedor > estimativa por `pricing` > desconhecido."""
    raw = usage or {}
    prompt_tokens = max(0, int(raw.get("prompt_tokens") or 0))
    completion_tokens = max(0, int(raw.get("completion_tokens") or 0))
    total_tokens = max(0, int(raw.get("total_tokens") or (prompt_tokens + completion_tokens)))
    explicit_cost = raw.get("cost") if raw.get("cost") is not None else raw.get("total_cost")
    cost: float | None = None
    source = CostSource.UNKNOWN
    if explicit_cost is not None:
        cost, source = max(0.0, float(explicit_cost)), CostSource.PROVIDER
    elif pricing is not None and raw:
        cost, source = pricing.cost(prompt_tokens, completion_tokens), CostSource.ESTIMATED
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=round(cost, 6) if cost is not None else None,
        cost_source=source,
        image_bytes_original=image_bytes_original,
        image_bytes_sent=image_bytes_sent,
    )


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
        max_image_dimension: int = DEFAULT_MAX_IMAGE_DIMENSION,
        pricing: ModelPricing | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._extractor = extractor
        self._review_threshold = review_threshold
        self._max_image_bytes = max_image_bytes
        self._max_image_dimension = max_image_dimension
        self._pricing = pricing
        self._clock = clock

    def run(self, image: bytes, document_type: DocumentType) -> ExtractionEnvelope:
        started = self._clock()
        resolved_type = document_type

        def envelope(**kwargs: object) -> ExtractionEnvelope:
            return ExtractionEnvelope(
                document_type=resolved_type,
                model=self._extractor.model_name,
                latency_ms=round((self._clock() - started) * 1000, 2),
                extracted_at=datetime.now(UTC),
                **kwargs,
            )

        def failure(issues: list[GuardrailIssue], **kwargs: object) -> ExtractionEnvelope:
            return envelope(status=ExtractionStatus.FAILED, confidence=0.0, issues=issues, **kwargs)

        input_issue = self._validate_image(image)
        if input_issue is not None:
            return failure(
                [input_issue],
                usage=build_token_usage(None, image_bytes_original=len(image), image_bytes_sent=0),
            )
        mime_type = detect_image_mime(image)
        assert mime_type is not None  # garantido por _validate_image

        processed_image, mime_type = preprocess_image(
            image, mime_type, max_dimension=self._max_image_dimension
        )

        try:
            result = self._extractor.extract(processed_image, mime_type, document_type)
        except ExtractorError as exc:
            return failure(
                [_issue("image", "extractor_error", str(exc))],
                usage=build_token_usage(
                    exc.usage,
                    pricing=self._pricing,
                    image_bytes_original=len(image),
                    image_bytes_sent=len(processed_image),
                ),
            )

        token_usage = build_token_usage(
            result.usage,
            pricing=self._pricing,
            image_bytes_original=len(image),
            image_bytes_sent=len(processed_image),
        )

        payload: Mapping[str, Any] = result.payload
        if document_type is DocumentType.AUTO:
            inferred, payload = resolve_auto_payload(result.payload)
            if inferred is None:
                return failure(
                    [
                        _issue(
                            "document_type",
                            "unknown_document_type",
                            "não foi possível identificar o tipo de documento",
                        )
                    ],
                    usage=token_usage,
                    raw_response=result.payload,
                )
            resolved_type = inferred

        try:
            data = SCHEMA_BY_TYPE[resolved_type].model_validate(payload)
        except ValidationError as exc:
            issues = [
                _issue(
                    ".".join(str(p) for p in err["loc"]) or "payload",
                    f"schema_{err['type']}",
                    err["msg"],
                )
                for err in exc.errors()
            ]
            return failure(issues, usage=token_usage, raw_response=result.payload)

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
            usage=token_usage,
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
