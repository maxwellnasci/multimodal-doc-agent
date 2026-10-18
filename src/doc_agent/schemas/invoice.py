"""Nota fiscal (NF-e / DANFE) com itens e total."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from doc_agent.schemas.common import (
    AccessKey,
    BRDate,
    Digits,
    DocumentModel,
    DocumentNumber,
    Money,
    OptionalStr,
    Quantity,
)


class InvoiceItem(DocumentModel):
    description: OptionalStr = Field(default=None, description="Descrição do produto/serviço")
    quantity: Quantity = Field(default=None, description="Quantidade")
    unit_price: Money = Field(default=None, description="Valor unitário em R$")
    total: Money = Field(default=None, description="Valor total do item em R$")


class InvoiceData(DocumentModel):
    critical_fields: ClassVar[tuple[str, ...]] = (
        "issuer_name",
        "issuer_document",
        "issue_date",
        "items",
        "total_amount",
    )

    invoice_number: Digits = Field(default=None, description="Número da nota")
    series: Digits = Field(default=None, description="Série")
    access_key: AccessKey = Field(
        default=None,
        description="Chave de acesso NF-e, 44 caracteres (o CNPJ nela pode conter letras)",
    )
    issue_date: BRDate = Field(default=None, description="Data de emissão")
    issuer_name: OptionalStr = Field(default=None, description="Razão social do emitente")
    issuer_document: DocumentNumber = Field(default=None, description="CNPJ do emitente")
    recipient_name: OptionalStr = Field(default=None, description="Nome do destinatário")
    recipient_document: DocumentNumber = Field(
        default=None, description="CPF ou CNPJ do destinatário"
    )
    items: list[InvoiceItem] = Field(default_factory=list, description="Itens da nota")
    total_amount: Money = Field(default=None, description="Valor total da nota em R$")
