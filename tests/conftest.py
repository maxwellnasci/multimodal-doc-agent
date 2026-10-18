from __future__ import annotations

import copy
from typing import Any

import pytest

from doc_agent.samples import SAMPLE_PAYLOADS
from doc_agent.schemas import DocumentType

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


@pytest.fixture
def png_bytes() -> bytes:
    return PNG_BYTES


@pytest.fixture
def prescription_payload() -> dict[str, Any]:
    return copy.deepcopy(SAMPLE_PAYLOADS[DocumentType.PRESCRIPTION])


@pytest.fixture
def vehicle_payload() -> dict[str, Any]:
    return copy.deepcopy(SAMPLE_PAYLOADS[DocumentType.VEHICLE])


@pytest.fixture
def invoice_payload() -> dict[str, Any]:
    return copy.deepcopy(SAMPLE_PAYLOADS[DocumentType.INVOICE])
