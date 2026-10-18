"""Validadores determinísticos (regex + dígitos verificadores), sem dependência de LLM."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from enum import StrEnum

MERCOSUL_PLATE_RE = re.compile(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$")
LEGACY_PLATE_RE = re.compile(r"^[A-Z]{3}-?[0-9]{4}$")
# ISO 3779: 17 caracteres alfanuméricos, sem I, O e Q (confundíveis com 1 e 0).
VIN_RE = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")
RENAVAM_RE = re.compile(r"^[0-9]{11}$")
CPF_RE = re.compile(r"^[0-9]{11}$")
# CNPJ alfanumérico (vigente desde jul/2026): 12 posições [0-9A-Z] + 2 DVs numéricos.
CNPJ_RE = re.compile(r"^[0-9A-Z]{12}[0-9]{2}$")
NFE_KEY_RE = re.compile(r"^[0-9]{44}$")

DIOPTER_STEP = Decimal("0.25")
AXIS_MIN, AXIS_MAX = 0, 180


class PlateFormat(StrEnum):
    MERCOSUL = "mercosul"
    LEGACY = "legacy"


def _clean(value: str) -> str:
    return re.sub(r"[\s.\-/]", "", value).upper()


def plate_format(plate: str | None) -> PlateFormat | None:
    if not plate:
        return None
    normalized = plate.strip().upper()
    if MERCOSUL_PLATE_RE.fullmatch(normalized):
        return PlateFormat.MERCOSUL
    if LEGACY_PLATE_RE.fullmatch(normalized):
        return PlateFormat.LEGACY
    return None


def is_valid_plate(plate: str | None) -> bool:
    return plate_format(plate) is not None


def is_valid_vin(vin: str | None) -> bool:
    return bool(vin) and VIN_RE.fullmatch(vin.strip().upper()) is not None


def is_valid_renavam(renavam: str | None) -> bool:
    """RENAVAM: 11 dígitos (os de 9 dígitos legados são completados com zeros à esquerda)."""
    if not renavam:
        return False
    digits = _clean(renavam)
    if len(digits) == 9 and digits.isdigit():
        digits = digits.zfill(11)
    if not RENAVAM_RE.fullmatch(digits):
        return False
    weights = (2, 3, 4, 5, 6, 7, 8, 9, 2, 3)
    total = sum(int(d) * w for d, w in zip(reversed(digits[:10]), weights, strict=True))
    check = (total * 10) % 11
    return (0 if check == 10 else check) == int(digits[10])


def is_valid_cpf(cpf: str | None) -> bool:
    if not cpf:
        return False
    digits = _clean(cpf)
    if not CPF_RE.fullmatch(digits) or len(set(digits)) == 1:
        return False
    for size in (9, 10):
        total = sum(int(d) * w for d, w in zip(digits[:size], range(size + 1, 1, -1), strict=True))
        check = (total * 10) % 11
        if (0 if check == 10 else check) != int(digits[size]):
            return False
    return True


def is_valid_cnpj(cnpj: str | None) -> bool:
    """Valida CNPJ numérico e alfanumérico (valor do caractere = código ASCII - 48)."""
    if not cnpj:
        return False
    chars = _clean(cnpj)
    if not CNPJ_RE.fullmatch(chars) or len(set(chars)) == 1:
        return False
    base_weights = (5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2)
    for size, weights in ((12, base_weights), (13, (6, *base_weights))):
        total = sum((ord(c) - 48) * w for c, w in zip(chars[:size], weights, strict=True))
        remainder = total % 11
        if (0 if remainder < 2 else 11 - remainder) != int(chars[size]):
            return False
    return True


def is_valid_tax_id(value: str | None) -> bool:
    """CPF (11) ou CNPJ (14), decidido pelo tamanho."""
    if not value:
        return False
    cleaned = _clean(value)
    return is_valid_cpf(cleaned) if len(cleaned) == 11 else is_valid_cnpj(cleaned)


def is_valid_nfe_access_key(key: str | None) -> bool:
    """Chave de acesso NF-e: 44 dígitos, DV módulo 11 com pesos 2..9 da direita para a esquerda."""
    if not key:
        return False
    digits = _clean(key)
    if not NFE_KEY_RE.fullmatch(digits):
        return False
    total = sum(int(d) * (2 + i % 8) for i, d in enumerate(reversed(digits[:43])))
    remainder = total % 11
    return (0 if remainder < 2 else 11 - remainder) == int(digits[43])


def is_valid_diopter_step(value: float | Decimal | None, step: Decimal = DIOPTER_STEP) -> bool:
    """Lentes são fabricadas em passos clínicos de 0,25 D."""
    if value is None:
        return False
    try:
        quotient = Decimal(str(value)) / step
    except InvalidOperation:
        return False
    return quotient == quotient.to_integral_value()


def is_valid_axis(axis: int | None) -> bool:
    return isinstance(axis, int) and not isinstance(axis, bool) and AXIS_MIN <= axis <= AXIS_MAX
