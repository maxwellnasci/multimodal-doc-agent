"""Envelope de resposta do pipeline: dados + guardrails + metadados."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from doc_agent.schemas.common import DocumentType
from doc_agent.schemas.invoice import InvoiceData
from doc_agent.schemas.prescription import PrescriptionData
from doc_agent.schemas.vehicle import VehicleDocumentData

DocumentData = PrescriptionData | VehicleDocumentData | InvoiceData


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class GuardrailIssue(BaseModel):
    field: str
    code: str
    message: str
    severity: Severity


class ExtractionStatus(StrEnum):
    OK = "ok"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


class TokenUsage(BaseModel):
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    image_bytes_original: int | None = Field(default=None, ge=0)
    image_bytes_sent: int | None = Field(default=None, ge=0)


class ExtractionEnvelope(BaseModel):
    document_type: DocumentType
    status: ExtractionStatus
    confidence: float = Field(ge=0.0, le=1.0)
    data: DocumentData | None = None
    issues: list[GuardrailIssue] = Field(default_factory=list)
    model: str
    latency_ms: float = Field(ge=0.0)
    extracted_at: datetime
    usage: TokenUsage = Field(default_factory=TokenUsage)
    raw_response: dict[str, Any] | None = None

    @property
    def errors(self) -> list[GuardrailIssue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[GuardrailIssue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]
