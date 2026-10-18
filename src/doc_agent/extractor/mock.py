"""Extrator determinístico para testes e demo sem API key."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import Any

from doc_agent.extractor.base import ExtractionResult, ExtractorClient
from doc_agent.samples import SAMPLE_PAYLOADS
from doc_agent.schemas import DocumentType


class MockExtractorClient(ExtractorClient):
    model_name = "mock/deterministic"

    def __init__(
        self,
        responses: Mapping[DocumentType, dict[str, Any]] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self._responses = {**SAMPLE_PAYLOADS, **(responses or {})}
        self._error = error
        self.calls: list[tuple[DocumentType, str, int]] = []

    def extract(
        self, image: bytes, mime_type: str, document_type: DocumentType
    ) -> ExtractionResult:
        self.calls.append((document_type, mime_type, len(image)))
        if self._error is not None:
            raise self._error
        payload = copy.deepcopy(self._responses[document_type])
        return ExtractionResult(
            payload=payload,
            model=self.model_name,
            raw_text=json.dumps(payload, ensure_ascii=False),
        )
