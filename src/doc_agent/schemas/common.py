"""Tipos base e normalizadores compartilhados pelos schemas de documento."""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, PlainSerializer


class DocumentType(StrEnum):
    PRESCRIPTION = "prescription"
    VEHICLE = "vehicle_document"
    INVOICE = "invoice"


class DocumentModel(BaseModel):
    """Base dos schemas: tolera chaves extras do modelo e limpa espaços."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore")


_DATE_FORMATS = ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y-%m-%d")
_NUMBER_RE = re.compile(r"[^0-9,.\-+]")


def _blank_to_none(value: Any) -> Any:
    if isinstance(value, str) and not value.strip():
        return None
    return value


def parse_br_date(value: Any) -> Any:
    """Aceita datas no padrão brasileiro (dd/mm/aaaa) além de ISO 8601."""
    value = _blank_to_none(value)
    if not isinstance(value, str):
        return value
    text = value.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return text  # deixa o Pydantic acusar o erro


def parse_br_number(value: Any) -> Any:
    """Converte "R$ 1.234,56" / "+1,25" / 3.5 em Decimal."""
    value = _blank_to_none(value)
    if value is None or isinstance(value, Decimal | bool):
        return value
    if isinstance(value, int | float):
        return Decimal(str(value))
    if not isinstance(value, str):
        return value
    text = _NUMBER_RE.sub("", value)
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    try:
        return Decimal(text)
    except InvalidOperation:
        return value


def parse_diopter(value: Any) -> Any:
    """Dioptrias: "plano"/"PL" equivalem a 0; vírgula decimal é aceita."""
    value = _blank_to_none(value)
    if isinstance(value, str) and value.strip().lower() in {"pl", "plano", "neutro"}:
        return 0.0
    parsed = parse_br_number(value)
    return float(parsed) if isinstance(parsed, Decimal) else parsed


def parse_axis(value: Any) -> Any:
    """Eixos costumam vir como "90°" ou "090"."""
    value = _blank_to_none(value)
    if isinstance(value, str):
        return value.strip().rstrip("°º").strip()
    return value


def normalize_document_number(value: Any) -> Any:
    """CPF/CNPJ sem pontuação; maiúsculas para o CNPJ alfanumérico (IN RFB 2.229/2024)."""
    value = _blank_to_none(value)
    if isinstance(value, str):
        return re.sub(r"[^0-9A-Za-z]", "", value).upper()
    return value


def normalize_code(value: Any) -> Any:
    """Placa/chassi: remove separadores e coloca em maiúsculas."""
    value = _blank_to_none(value)
    if isinstance(value, str):
        return re.sub(r"[\s\-.]", "", value).upper()
    return value


def digits_only(value: Any) -> Any:
    value = _blank_to_none(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return re.sub(r"\D", "", value)
    return value


OptionalStr = Annotated[str | None, BeforeValidator(_blank_to_none)]
BRDate = Annotated[date | None, BeforeValidator(parse_br_date)]
Money = Annotated[
    Decimal | None,
    BeforeValidator(parse_br_number),
    PlainSerializer(float, return_type=float, when_used="json-unless-none"),
]
Diopter = Annotated[float | None, BeforeValidator(parse_diopter)]
Axis = Annotated[int | None, BeforeValidator(parse_axis)]
DocumentNumber = Annotated[str | None, BeforeValidator(normalize_document_number)]
Code = Annotated[str | None, BeforeValidator(normalize_code)]
Digits = Annotated[str | None, BeforeValidator(digits_only)]
