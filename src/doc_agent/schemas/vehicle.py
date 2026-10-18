"""CRLV — Certificado de Registro e Licenciamento de Veículo."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from doc_agent.schemas.common import Code, Digits, DocumentModel, DocumentNumber, OptionalStr


class VehicleDocumentData(DocumentModel):
    critical_fields: ClassVar[tuple[str, ...]] = (
        "plate",
        "chassis",
        "renavam",
        "owner_name",
        "owner_document",
        "brand_model",
        "model_year",
    )

    plate: Code = Field(default=None, description="Placa (Mercosul ABC1D23 ou antiga ABC1234)")
    chassis: Code = Field(default=None, description="Número do chassi/VIN, 17 caracteres")
    renavam: Digits = Field(default=None, description="Código RENAVAM, 11 dígitos")
    owner_name: OptionalStr = Field(default=None, description="Nome do proprietário")
    owner_document: DocumentNumber = Field(default=None, description="CPF ou CNPJ do proprietário")
    brand_model: OptionalStr = Field(default=None, description="Marca/modelo")
    manufacture_year: int | None = Field(default=None, description="Ano de fabricação")
    model_year: int | None = Field(default=None, description="Ano do modelo")
    color: OptionalStr = Field(default=None, description="Cor predominante")
    fuel: OptionalStr = Field(default=None, description="Combustível")
    licensing_year: int | None = Field(default=None, description="Exercício do licenciamento")
