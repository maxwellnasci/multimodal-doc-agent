from __future__ import annotations

from datetime import date, timedelta

import pytest

from doc_agent.guardrails import run_guardrails
from doc_agent.schemas import InvoiceData, PrescriptionData, Severity, VehicleDocumentData


def codes(issues):
    return {(i.field, i.code) for i in issues}


@pytest.mark.parametrize(
    ("schema", "fixture"),
    [
        (PrescriptionData, "prescription_payload"),
        (VehicleDocumentData, "vehicle_payload"),
        (InvoiceData, "invoice_payload"),
    ],
)
def test_samples_pass_all_guardrails(schema, fixture, request):
    assert run_guardrails(schema.model_validate(request.getfixturevalue(fixture))) == []


def test_run_guardrails_rejects_unknown_type():
    with pytest.raises(TypeError):
        run_guardrails(object())  # type: ignore[arg-type]


# --- Receita -------------------------------------------------------------------------------


def test_prescription_invalid_step_and_axis(prescription_payload):
    prescription_payload["od"].update(sphere="-2,20", axis=190)
    issues = run_guardrails(PrescriptionData.model_validate(prescription_payload))
    assert codes(issues) == {("od.sphere", "invalid_diopter_step"), ("od.axis", "invalid_axis")}
    assert all(i.severity is Severity.ERROR for i in issues)


def test_prescription_out_of_range(prescription_payload):
    prescription_payload["oe"]["cylinder"] = "-12,00"
    issues = run_guardrails(PrescriptionData.model_validate(prescription_payload))
    assert ("oe.cylinder", "out_of_range") in codes(issues)


def test_prescription_cylinder_requires_axis(prescription_payload):
    prescription_payload["od"]["axis"] = None
    issues = run_guardrails(PrescriptionData.model_validate(prescription_payload))
    assert codes(issues) == {("od.axis", "axis_required")}


def test_prescription_axis_without_cylinder_is_warning(prescription_payload):
    prescription_payload["oe"]["cylinder"] = "plano"
    issues = run_guardrails(PrescriptionData.model_validate(prescription_payload))
    assert codes(issues) == {("oe.axis", "axis_without_cylinder")}
    assert issues[0].severity is Severity.WARNING


def test_prescription_missing_eye_and_future_date(prescription_payload):
    prescription_payload["oe"] = None
    prescription_payload["issue_date"] = (date.today() + timedelta(days=3)).isoformat()
    issues = run_guardrails(PrescriptionData.model_validate(prescription_payload))
    assert codes(issues) == {("oe", "missing_eye"), ("issue_date", "future_date")}


def test_prescription_empty_eye_object_is_flagged_as_missing(prescription_payload):
    prescription_payload["od"] = {"sphere": None, "cylinder": None, "axis": None, "addition": None}
    issues = run_guardrails(PrescriptionData.model_validate(prescription_payload))
    assert codes(issues) == {("od", "missing_eye")}


# --- CRLV ----------------------------------------------------------------------------------


def test_vehicle_invalid_identifiers(vehicle_payload):
    vehicle_payload.update(
        plate="BR42E19", chassis="9BWZZZ377VTO04251", renavam="00639884961", owner_document="1"
    )
    issues = run_guardrails(VehicleDocumentData.model_validate(vehicle_payload))
    assert codes(issues) == {
        ("plate", "invalid_plate"),
        ("chassis", "invalid_chassis"),
        ("renavam", "invalid_renavam"),
        ("owner_document", "invalid_tax_id"),
    }


def test_vehicle_missing_required_fields():
    issues = run_guardrails(VehicleDocumentData())
    assert codes(issues) == {
        ("plate", "missing_field"),
        ("chassis", "missing_field"),
        ("renavam", "missing_field"),
    }


def test_vehicle_accepts_company_owner(vehicle_payload):
    vehicle_payload["owner_document"] = "12.ABC.345/01DE-35"
    assert run_guardrails(VehicleDocumentData.model_validate(vehicle_payload)) == []


@pytest.mark.parametrize(("fab", "mod"), [(2022, 2021), (2020, 2022)])
def test_vehicle_inconsistent_years(vehicle_payload, fab, mod):
    vehicle_payload.update(manufacture_year=fab, model_year=mod)
    issues = run_guardrails(VehicleDocumentData.model_validate(vehicle_payload))
    assert codes(issues) == {("model_year", "inconsistent_years")}


# --- Nota fiscal ---------------------------------------------------------------------------


def test_invoice_total_mismatch(invoice_payload):
    invoice_payload["total_amount"] = "R$ 1.200,00"
    issues = run_guardrails(InvoiceData.model_validate(invoice_payload))
    assert codes(issues) == {("total_amount", "total_mismatch")}


def test_invoice_total_within_tolerance(invoice_payload):
    invoice_payload["total_amount"] = "1189,91"
    assert run_guardrails(InvoiceData.model_validate(invoice_payload)) == []


def test_invoice_item_math_mismatch_is_warning(invoice_payload):
    invoice_payload["items"][0]["unit_price"] = "400,00"
    issues = run_guardrails(InvoiceData.model_validate(invoice_payload))
    assert codes(issues) == {("items.0.total", "item_total_mismatch")}
    assert issues[0].severity is Severity.WARNING


def test_invoice_skips_incomplete_items(invoice_payload):
    invoice_payload["items"][0]["quantity"] = None
    invoice_payload["items"][1]["total"] = None
    assert run_guardrails(InvoiceData.model_validate(invoice_payload)) == []


def test_invoice_invalid_documents_and_key(invoice_payload):
    invoice_payload.update(
        issuer_document="11.222.333/0001-80",
        recipient_document="529.982.247-24",
        access_key="1234",
    )
    issues = run_guardrails(InvoiceData.model_validate(invoice_payload))
    assert codes(issues) == {
        ("issuer_document", "invalid_tax_id"),
        ("recipient_document", "invalid_tax_id"),
        ("access_key", "invalid_access_key"),
    }


def test_invoice_access_key_cross_validation_mismatches(invoice_payload):
    invoice_payload.update(
        issuer_document="12.ABC.345/01DE-35",  # válido, mas difere da chave (11222333000181)
        issue_date="10/11/2026",  # AAMM 2611 != 2610 na chave
        invoice_number="99999",  # != 12345 na chave
        series="2",  # != 1 na chave
    )
    issues = run_guardrails(InvoiceData.model_validate(invoice_payload))
    assert codes(issues) == {
        ("access_key", "access_key_cnpj_mismatch"),
        ("access_key", "access_key_date_mismatch"),
        ("access_key", "access_key_number_mismatch"),
        ("access_key", "access_key_series_mismatch"),
    }


def _nfe_key(issuer_field: str) -> str:
    """Chave NF-e com DV válido: SP, 10/2026, modelo 55, série 1, nNF 12345."""
    base = f"352610{issuer_field}55001000012345100012345"
    total = sum((ord(c) - 48) * (2 + i % 8) for i, c in enumerate(reversed(base)))
    remainder = total % 11
    return base + str(0 if remainder < 2 else 11 - remainder)


def test_invoice_rural_producer_cpf_matches_zero_padded_key(invoice_payload):
    invoice_payload.update(issuer_document="529.982.247-25", access_key=_nfe_key("00052998224725"))
    assert run_guardrails(InvoiceData.model_validate(invoice_payload)) == []


def test_invoice_rural_producer_cpf_mismatch_is_flagged(invoice_payload):
    invoice_payload.update(issuer_document="529.982.247-25", access_key=_nfe_key("00011144477735"))
    issues = run_guardrails(InvoiceData.model_validate(invoice_payload))
    assert codes(issues) == {("access_key", "access_key_cnpj_mismatch")}


def test_invoice_alphanumeric_cnpj_key_matches_issuer(invoice_payload):
    invoice_payload.update(
        issuer_document="12.ABC.345/01DE-35",
        access_key="3526 1012AB C345 01DE 3555 0010 0001 2345 1000 1234 51",
    )
    data = InvoiceData.model_validate(invoice_payload)
    assert data.access_key == _nfe_key("12ABC34501DE35")
    assert run_guardrails(data) == []


def test_invoice_alphanumeric_cnpj_key_mismatch_is_flagged(invoice_payload):
    invoice_payload.update(access_key=_nfe_key("12ABC34501DE35"))  # emitente: 11222333000181
    issues = run_guardrails(InvoiceData.model_validate(invoice_payload))
    assert codes(issues) == {("access_key", "access_key_cnpj_mismatch")}


@pytest.mark.parametrize("issuer_field", ["11222333000180", "12ABC34501DE36", "00052998224724"])
def test_invoice_key_with_invalid_issuer_is_flagged_without_extracted_issuer(
    invoice_payload, issuer_field
):
    invoice_payload.update(issuer_document=None, access_key=_nfe_key(issuer_field))
    issues = run_guardrails(InvoiceData.model_validate(invoice_payload))
    assert ("access_key", "access_key_invalid_issuer") in codes(issues)
    assert ("access_key", "access_key_cnpj_mismatch") not in codes(issues)


def test_invoice_missing_total_and_items():
    assert codes(run_guardrails(InvoiceData())) == {("total_amount", "missing_field")}
    assert codes(run_guardrails(InvoiceData(total_amount="10"))) == {("items", "missing_items")}
