"""Exemplos embutidos: payloads válidos (fictícios) e imagens sintéticas para a demo."""

from __future__ import annotations

import io
from functools import cache
from typing import Any

from doc_agent.schemas import DocumentType

# Todos os identificadores são fictícios, mas com dígitos verificadores válidos.
SAMPLE_PAYLOADS: dict[DocumentType, dict[str, Any]] = {
    DocumentType.PRESCRIPTION: {
        "patient_name": "Maria Aparecida Souza",
        "prescriber_name": "Dr. João Pereira",
        "prescriber_registry": "CRM-SP 123456",
        "issue_date": "15/09/2026",
        "od": {"sphere": "-2,25", "cylinder": "-0,75", "axis": "180°", "addition": "+1,50"},
        "oe": {"sphere": "-2,00", "cylinder": "-0,50", "axis": "170°", "addition": "+1,50"},
        "notes": "Lentes multifocais com antirreflexo",
    },
    DocumentType.VEHICLE: {
        "plate": "BRA-2E19",
        "chassis": "9BWZZZ377VT004251",
        "renavam": "00639884962",
        "owner_name": "Carlos Eduardo Lima",
        "owner_document": "529.982.247-25",
        "brand_model": "VW/GOL 1.0",
        "manufacture_year": 2021,
        "model_year": 2022,
        "color": "Prata",
        "fuel": "Álcool/Gasolina",
        "licensing_year": 2026,
    },
    DocumentType.INVOICE: {
        "invoice_number": "000.012.345",
        "series": "1",
        "access_key": "3526 1011 2223 3300 0181 5500 1000 0123 4510 0012 3450",
        "issue_date": "10/10/2026",
        "issuer_name": "Ótica Visão Clara LTDA",
        "issuer_document": "11.222.333/0001-81",
        "recipient_name": "Maria Aparecida Souza",
        "recipient_document": "529.982.247-25",
        "items": [
            {
                "description": "Lente multifocal 1.67",
                "quantity": "2",
                "unit_price": "R$ 450,00",
                "total": "R$ 900,00",
            },
            {
                "description": "Armação acetato",
                "quantity": "1",
                "unit_price": "R$ 289,90",
                "total": "R$ 289,90",
            },
        ],
        "total_amount": "R$ 1.189,90",
    },
}

_TITLES = {
    DocumentType.PRESCRIPTION: "RECEITA OFTALMOLÓGICA",
    DocumentType.VEHICLE: "CRLV - CERTIFICADO DE REGISTRO E LICENCIAMENTO",
    DocumentType.INVOICE: "DANFE - NOTA FISCAL ELETRÔNICA",
}


def _lines(document_type: DocumentType) -> list[str]:
    p = SAMPLE_PAYLOADS[document_type]
    if document_type is DocumentType.PRESCRIPTION:
        eye = "{label}: ESF {sphere}  CIL {cylinder}  EIXO {axis}  ADD {addition}"
        return [
            f"Paciente: {p['patient_name']}",
            f"Data: {p['issue_date']}",
            "",
            eye.format(label="OD", **p["od"]),
            eye.format(label="OE", **p["oe"]),
            "",
            f"Obs.: {p['notes']}",
            "",
            f"{p['prescriber_name']} - {p['prescriber_registry']}",
        ]
    if document_type is DocumentType.VEHICLE:
        return [
            f"PLACA: {p['plate']}        EXERCÍCIO: {p['licensing_year']}",
            f"RENAVAM: {p['renavam']}",
            f"CHASSI: {p['chassis']}",
            f"MARCA/MODELO: {p['brand_model']}",
            f"ANO FAB: {p['manufacture_year']}   ANO MOD: {p['model_year']}",
            f"COR: {p['color']}   COMBUSTÍVEL: {p['fuel']}",
            "",
            f"NOME: {p['owner_name']}",
            f"CPF/CNPJ: {p['owner_document']}",
        ]
    items = [
        f"  {i['description']:<26} {i['quantity']:>3} x {i['unit_price']:>10} = {i['total']:>10}"
        for i in p["items"]
    ]
    return [
        f"NF-e Nº {p['invoice_number']}  Série {p['series']}  Emissão {p['issue_date']}",
        f"Chave: {p['access_key']}",
        "",
        f"EMITENTE: {p['issuer_name']}  CNPJ {p['issuer_document']}",
        f"DESTINATÁRIO: {p['recipient_name']}  CPF {p['recipient_document']}",
        "",
        "PRODUTOS",
        *items,
        "",
        f"VALOR TOTAL DA NOTA: {p['total_amount']}",
    ]


@cache
def render_sample_image(document_type: DocumentType) -> bytes:
    """Gera um PNG sintético do documento de exemplo (determinístico)."""
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1100, 620
    image = Image.new("RGB", (width, height), "#fbfaf6")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default(size=28)
    body_font = ImageFont.load_default(size=21)

    draw.rectangle((0, 0, width, 70), fill="#1f3a5f")
    draw.text((30, 20), _TITLES[document_type], fill="white", font=title_font)
    draw.rectangle((15, 85, width - 15, height - 15), outline="#9aa5b1", width=2)
    y = 110
    for line in _lines(document_type):
        draw.text((40, y), line, fill="#1b1b1b", font=body_font)
        y += 40 if line else 18

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
