"""Regras de negócio por tipo de documento, construídas sobre os validadores."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from doc_agent.guardrails import validators as v
from doc_agent.schemas import (
    DocumentData,
    EyePrescription,
    GuardrailIssue,
    InvoiceData,
    PrescriptionData,
    Severity,
    VehicleDocumentData,
)

SPHERE_RANGE = (-30.0, 30.0)
CYLINDER_RANGE = (-10.0, 10.0)
ADDITION_RANGE = (0.75, 4.0)
MONEY_TOLERANCE = Decimal("0.02")


def _error(field: str, code: str, message: str) -> GuardrailIssue:
    return GuardrailIssue(field=field, code=code, message=message, severity=Severity.ERROR)


def _warning(field: str, code: str, message: str) -> GuardrailIssue:
    return GuardrailIssue(field=field, code=code, message=message, severity=Severity.WARNING)


def _check_eye(prefix: str, eye: EyePrescription) -> list[GuardrailIssue]:
    issues: list[GuardrailIssue] = []
    ranges = {"sphere": SPHERE_RANGE, "cylinder": CYLINDER_RANGE, "addition": ADDITION_RANGE}
    for attr, (low, high) in ranges.items():
        value = getattr(eye, attr)
        if value is None:
            continue
        field = f"{prefix}.{attr}"
        if not v.is_valid_diopter_step(value):
            issues.append(
                _error(field, "invalid_diopter_step", f"{value} D não é múltiplo de 0,25")
            )
        if not low <= value <= high:
            issues.append(_error(field, "out_of_range", f"{value} D fora de [{low}, {high}]"))

    has_cylinder = eye.cylinder not in (None, 0)
    if eye.axis is not None and not v.is_valid_axis(eye.axis):
        issues.append(_error(f"{prefix}.axis", "invalid_axis", f"eixo {eye.axis}° fora de 0-180"))
    if has_cylinder and eye.axis is None:
        issues.append(_error(f"{prefix}.axis", "axis_required", "cilindro informado sem eixo"))
    if not has_cylinder and eye.axis is not None:
        issues.append(_warning(f"{prefix}.axis", "axis_without_cylinder", "eixo sem cilindro"))
    return issues


def _is_empty_eye(eye: EyePrescription | None) -> bool:
    return eye is None or all(
        getattr(eye, attr) is None for attr in ("sphere", "cylinder", "axis", "addition")
    )


def check_prescription(data: PrescriptionData) -> list[GuardrailIssue]:
    issues: list[GuardrailIssue] = []
    for prefix, eye in (("od", data.od), ("oe", data.oe)):
        if _is_empty_eye(eye):
            issues.append(_warning(prefix, "missing_eye", f"grau do {prefix.upper()} ausente"))
        else:
            assert eye is not None
            issues.extend(_check_eye(prefix, eye))
    if data.issue_date and data.issue_date > date.today():
        issues.append(_warning("issue_date", "future_date", "data de emissão no futuro"))
    return issues


def check_vehicle(data: VehicleDocumentData) -> list[GuardrailIssue]:
    issues: list[GuardrailIssue] = []
    required = (
        ("plate", v.is_valid_plate, "placa fora dos padrões Mercosul/antigo"),
        ("chassis", v.is_valid_vin, "chassi não segue a ISO 3779 (17 caracteres, sem I/O/Q)"),
        ("renavam", v.is_valid_renavam, "RENAVAM com dígito verificador inválido"),
    )
    for field, validator, message in required:
        value = getattr(data, field)
        if value is None:
            issues.append(_error(field, "missing_field", f"{field} não encontrado"))
        elif not validator(value):
            issues.append(_error(field, f"invalid_{field}", message))

    if data.owner_document and not v.is_valid_tax_id(data.owner_document):
        issues.append(_error("owner_document", "invalid_tax_id", "CPF/CNPJ inválido"))
    years = (data.manufacture_year, data.model_year)
    if None not in years and not 0 <= years[1] - years[0] <= 1:
        issues.append(
            _warning(
                "model_year",
                "inconsistent_years",
                "ano modelo deve ser igual ou 1 ano após o de fabricação",
            )
        )
    return issues


def _check_nfe_key_consistency(data: InvoiceData) -> list[GuardrailIssue]:
    key_info = v.parse_nfe_access_key(data.access_key)
    if key_info is None:
        return []
    issues: list[GuardrailIssue] = []
    issuer = data.issuer_document
    # Emitente pessoa física (ex.: produtor rural): a chave traz "000" + CPF no campo do CNPJ.
    expected_key_doc = f"000{issuer}" if issuer and len(issuer) == 11 else issuer
    if issuer and expected_key_doc != key_info.issuer_cnpj:
        issues.append(
            _error(
                "access_key",
                "access_key_cnpj_mismatch",
                f"CNPJ/CPF na chave ({key_info.issuer_cnpj}) != emitente ({issuer})",
            )
        )
    if data.issue_date is not None:
        expected_yymm = f"{data.issue_date.year % 100:02d}{data.issue_date.month:02d}"
        if expected_yymm != key_info.year_month:
            issues.append(
                _error(
                    "access_key",
                    "access_key_date_mismatch",
                    f"AAMM na chave ({key_info.year_month}) difere da emissão ({expected_yymm})",
                )
            )
    if data.invoice_number and int(data.invoice_number) != key_info.invoice_number:
        issues.append(
            _error(
                "access_key",
                "access_key_number_mismatch",
                f"nNF na chave ({key_info.invoice_number}) != nota ({int(data.invoice_number)})",
            )
        )
    if data.series and int(data.series) != key_info.series:
        issues.append(
            _warning(
                "access_key",
                "access_key_series_mismatch",
                f"série na chave ({key_info.series}) difere da nota ({int(data.series)})",
            )
        )
    return issues


def check_invoice(data: InvoiceData) -> list[GuardrailIssue]:
    issues: list[GuardrailIssue] = []
    if data.issuer_document and not v.is_valid_tax_id(data.issuer_document):
        issues.append(_error("issuer_document", "invalid_tax_id", "CNPJ/CPF do emitente inválido"))
    if data.recipient_document and not v.is_valid_tax_id(data.recipient_document):
        issues.append(
            _error("recipient_document", "invalid_tax_id", "CPF/CNPJ do destinatário inválido")
        )
    if data.access_key:
        if not v.is_valid_nfe_access_key(data.access_key):
            issues.append(_error("access_key", "invalid_access_key", "chave de acesso inválida"))
        else:
            issues.extend(_check_nfe_key_consistency(data))

    for idx, item in enumerate(data.items):
        if None in (item.quantity, item.unit_price, item.total):
            continue
        expected = item.quantity * item.unit_price
        if abs(expected - item.total) > MONEY_TOLERANCE:
            issues.append(
                _warning(
                    f"items.{idx}.total",
                    "item_total_mismatch",
                    f"{item.quantity} x {item.unit_price} = {expected:.2f}, nota diz {item.total}",
                )
            )

    if data.total_amount is None:
        issues.append(_error("total_amount", "missing_field", "valor total não encontrado"))
    elif not data.items:
        issues.append(_warning("items", "missing_items", "nenhum item extraído"))
    elif all(item.total is not None for item in data.items):
        items_sum = sum((item.total for item in data.items), Decimal("0"))
        if abs(items_sum - data.total_amount) > MONEY_TOLERANCE:
            issues.append(
                _error(
                    "total_amount",
                    "total_mismatch",
                    f"soma dos itens ({items_sum:.2f}) difere do total ({data.total_amount})",
                )
            )
    return issues


def run_guardrails(data: DocumentData) -> list[GuardrailIssue]:
    if isinstance(data, PrescriptionData):
        return check_prescription(data)
    if isinstance(data, VehicleDocumentData):
        return check_vehicle(data)
    if isinstance(data, InvoiceData):
        return check_invoice(data)
    raise TypeError(f"tipo de documento não suportado: {type(data).__name__}")
