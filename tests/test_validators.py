from __future__ import annotations

from decimal import Decimal

import pytest

from doc_agent.guardrails import validators as v


@pytest.mark.parametrize(
    ("plate", "expected"),
    [
        ("BRA2E19", v.PlateFormat.MERCOSUL),
        ("BRA-2E19", v.PlateFormat.MERCOSUL),
        ("abc1d23", v.PlateFormat.MERCOSUL),
        ("ABC1234", v.PlateFormat.LEGACY),
        ("ABC-1234", v.PlateFormat.LEGACY),
        ("AB12345", None),
        ("ABCD123", None),
        ("ABC12D3", None),
        ("", None),
        (None, None),
    ],
)
def test_plate_format(plate, expected):
    assert v.plate_format(plate) == expected
    assert v.is_valid_plate(plate) is (expected is not None)


@pytest.mark.parametrize(
    ("vin", "valid"),
    [
        ("9BWZZZ377VT004251", True),
        ("1HGCM82633A004352", True),
        ("9BWZZZ377VT00425", False),  # 16 caracteres
        ("9BWZZZ377VT0042511", False),  # 18 caracteres
        ("9BWZZZ377VTO04251", False),  # contém O
        ("9BWZZZ377VTI04251", False),  # contém I
        ("9BWZZZ377VTQ04251", False),  # contém Q
        ("AAAAAAAAAAAAAAAAA", False),  # caractere único repetido
        ("11111111111111111", False),  # dígito único repetido
        ("9BWZZZ377VT00425-", False),
        (None, False),
    ],
)
def test_vin_iso_3779(vin, valid):
    assert v.is_valid_vin(vin) is valid


@pytest.mark.parametrize(
    ("renavam", "valid"),
    [
        ("00639884962", True),
        ("639884962", True),  # 9 dígitos legado, completado com zeros
        ("00639884961", False),
        ("0063988496", False),
        ("0063988496A", False),
        (None, False),
    ],
)
def test_renavam(renavam, valid):
    assert v.is_valid_renavam(renavam) is valid


@pytest.mark.parametrize(
    ("cpf", "valid"),
    [
        ("529.982.247-25", True),
        ("52998224725", True),
        ("529.982.247-24", False),
        ("111.111.111-11", False),
        ("5299822472", False),
        ("", False),
    ],
)
def test_cpf(cpf, valid):
    assert v.is_valid_cpf(cpf) is valid


@pytest.mark.parametrize(
    ("cnpj", "valid"),
    [
        ("11.222.333/0001-81", True),
        ("11222333000181", True),
        ("12.ABC.345/01DE-35", True),  # exemplo oficial de CNPJ alfanumérico
        ("12abc34501de35", True),
        ("11.222.333/0001-80", False),
        ("12.ABC.345/01DE-36", False),
        ("00.000.000/0000-00", False),
        ("AAAAAAAAAAAA45", False),  # raiz repetida alfanumérica
        ("12.IBC.345/01DE-00", False),  # letra proibida I (IN RFB 2.229/2024)
        ("12.OBC.345/01DE-00", False),  # letra proibida O
        ("12.UBC.345/01DE-00", False),  # letra proibida U
        ("12.QBC.345/01DE-00", False),  # letra proibida Q
        ("12.FBC.345/01DE-00", False),  # letra proibida F
        ("12ABC34501DEAB", False),  # DVs devem ser numéricos
        (None, False),
    ],
)
def test_cnpj(cnpj, valid):
    assert v.is_valid_cnpj(cnpj) is valid


@pytest.mark.parametrize(
    ("value", "valid"),
    [("529.982.247-25", True), ("11.222.333/0001-81", True), ("123", False), (None, False)],
)
def test_tax_id_dispatches_by_length(value, valid):
    assert v.is_valid_tax_id(value) is valid


def test_nfe_access_key():
    key = "35261011222333000181550010000123451000123450"
    assert v.is_valid_nfe_access_key(key)
    assert v.is_valid_nfe_access_key(" ".join(key[i : i + 4] for i in range(0, 44, 4)))
    assert not v.is_valid_nfe_access_key("0" * 44)
    assert not v.is_valid_nfe_access_key(key[:-1] + "1")
    assert not v.is_valid_nfe_access_key(key[:-1])
    assert not v.is_valid_nfe_access_key(None)

    parsed = v.parse_nfe_access_key(key)
    assert parsed == v.NFeAccessKeyInfo(
        uf_code="35",
        year_month="2610",
        issuer_cnpj="11222333000181",
        model="55",
        series=1,
        invoice_number=12345,
    )
    assert v.parse_nfe_access_key("0" * 44) is None


@pytest.mark.parametrize(
    ("value", "valid"),
    [
        (0.0, True),
        (-2.25, True),
        (1.5, True),
        (Decimal("-0.75"), True),
        (10.0, True),
        (-2.2, False),
        (0.1, False),
        (1.33, False),
        (None, False),
        (float("nan"), False),
        (float("inf"), False),
        (Decimal("Infinity"), False),
    ],
)
def test_diopter_step(value, valid):
    assert v.is_valid_diopter_step(value) is valid


@pytest.mark.parametrize(
    ("axis", "valid"),
    [(0, True), (90, True), (180, True), (-1, False), (181, False), (None, False), (True, False)],
)
def test_axis(axis, valid):
    assert v.is_valid_axis(axis) is valid
