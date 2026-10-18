from __future__ import annotations

import json
from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from doc_agent.schemas import (
    SCHEMA_BY_TYPE,
    DocumentType,
    EyePrescription,
    InvoiceData,
    InvoiceItem,
    PrescriptionData,
    VehicleDocumentData,
)


def test_prescription_normalizes_brazilian_formats(prescription_payload):
    data = PrescriptionData.model_validate(prescription_payload)
    assert data.issue_date == date(2026, 9, 15)
    assert data.od == EyePrescription(sphere=-2.25, cylinder=-0.75, axis=180, addition=1.5)
    assert data.oe is not None
    assert data.oe.axis == 170


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("plano", 0.0), ("PL", 0.0), ("+1,25", 1.25), ("-0.50", -0.5), (2, 2.0), ("", None)],
)
def test_diopter_parsing(raw, expected):
    assert EyePrescription(sphere=raw).sphere == expected


@pytest.mark.parametrize("raw", ["90°", "090", 90, " 90º "])
def test_axis_parsing(raw):
    assert EyePrescription(axis=raw).axis == 90


def test_invalid_axis_type_raises():
    with pytest.raises(ValidationError):
        EyePrescription(axis="noventa")


def test_vehicle_normalizes_codes(vehicle_payload):
    vehicle_payload["chassis"] = "9bw zzz377vt004251"
    data = VehicleDocumentData.model_validate(vehicle_payload)
    assert data.plate == "BRA2E19"
    assert data.chassis == "9BWZZZ377VT004251"
    assert data.owner_document == "52998224725"
    assert data.renavam == "00639884962"


def test_renavam_accepts_integer():
    assert VehicleDocumentData(renavam=639884962).renavam == "639884962"


def test_invoice_parses_money_and_items(invoice_payload):
    data = InvoiceData.model_validate(invoice_payload)
    assert data.total_amount == Decimal("1189.90")
    assert data.items[0].unit_price == Decimal("450.00")
    assert data.items[1].quantity == Decimal("1")
    assert data.issuer_document == "11222333000181"
    assert len(data.access_key or "") == 44


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("R$ 1.234,56", "1234.56"),
        ("R$ 1.200", "1200"),
        ("1.000.000", "1000000"),
        ("10,5", "10.5"),
        ("99.90", "99.90"),
        (12.3, "12.3"),
        (7, "7"),
        (Decimal("42.50"), "42.50"),
    ],
)
def test_money_parsing(raw, expected):
    assert InvoiceData(total_amount=raw).total_amount == Decimal(expected)


def test_access_key_keeps_cnpj_letters_and_drops_separators():
    data = InvoiceData(access_key="3526.1012.abc3.4501.de35-5500 1000 0123 4510 0012 3451")
    assert data.access_key == "35261012ABC34501DE35550010000123451000123451"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1.500", "1.500"),  # qCom com 3 casas, não milhar
        ("0.250", "0.250"),
        ("1,5", "1.5"),
        ("1.500,0000", "1500.0000"),
        ("2", "2"),
        (3.75, "3.75"),
    ],
)
def test_quantity_parsing_treats_bare_dot_as_decimal(raw, expected):
    assert InvoiceItem(quantity=raw).quantity == Decimal(expected)


def test_money_keeps_bare_thousands_while_quantity_does_not():
    item = InvoiceItem(quantity="1.200", unit_price="R$ 1.200", total="1.440")
    assert item.quantity == Decimal("1.2")
    assert item.unit_price == Decimal("1200")
    assert item.total == Decimal("1440")


@pytest.mark.parametrize("invalid_qty", [float("nan"), "Infinity", Decimal("-Inf")])
def test_non_finite_quantity_rejected(invalid_qty):
    with pytest.raises(ValidationError):
        InvoiceItem(quantity=invalid_qty)


@pytest.mark.parametrize(
    "invalid_num",
    [float("nan"), float("inf"), float("-inf"), "NaN", "Infinity", Decimal("NaN"), Decimal("Inf")],
)
def test_non_finite_numbers_rejected(invalid_num):
    with pytest.raises(ValidationError):
        InvoiceData(total_amount=invalid_num)
    with pytest.raises(ValidationError):
        EyePrescription(sphere=invalid_num)


def test_money_serializes_as_number_in_json(invoice_payload):
    dumped = json.loads(InvoiceData.model_validate(invoice_payload).model_dump_json())
    assert dumped["total_amount"] == pytest.approx(1189.90)
    assert InvoiceData().model_dump(mode="json")["total_amount"] is None


@pytest.mark.parametrize("raw", ["2026-13-45", "ontem"])
def test_invalid_date_raises(raw):
    with pytest.raises(ValidationError):
        InvoiceData(issue_date=raw)


@pytest.mark.parametrize("raw", ["18/10/2026", "18-10-2026", "18.10.2026", "2026-10-18"])
def test_date_formats(raw):
    assert InvoiceData(issue_date=raw).issue_date == date(2026, 10, 18)


def test_blank_and_punctuation_only_strings_become_none():
    data = PrescriptionData(patient_name="   ", issue_date="", od={"axis": "°"})
    assert data.patient_name is None
    assert data.issue_date is None
    assert data.od is not None
    assert data.od.axis is None

    vehicle = VehicleDocumentData(plate="---", owner_document="../-", renavam="N/A")
    assert vehicle.plate is None
    assert vehicle.owner_document is None
    assert vehicle.renavam is None


def test_extra_keys_are_ignored():
    data = VehicleDocumentData.model_validate({"plate": "ABC1234", "confidence": 0.9})
    assert not hasattr(data, "confidence")


@pytest.mark.parametrize("doc_type", list(SCHEMA_BY_TYPE))
def test_every_type_has_schema_and_json_schema(doc_type):
    schema = SCHEMA_BY_TYPE[doc_type]
    json_schema = schema.model_json_schema()
    assert json_schema["type"] == "object"
    for path in schema.critical_fields:
        assert path.split(".")[0] in json_schema["properties"]


def test_document_type_includes_auto():
    assert DocumentType.AUTO == "auto"
