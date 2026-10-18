"""Receita oftalmológica: OD (olho direito) e OE (olho esquerdo)."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from doc_agent.schemas.common import Axis, BRDate, Diopter, DocumentModel, OptionalStr


class EyePrescription(DocumentModel):
    sphere: Diopter = Field(default=None, description="Grau esférico em dioptrias (ex.: -2.25)")
    cylinder: Diopter = Field(default=None, description="Grau cilíndrico em dioptrias (ex.: -0.75)")
    axis: Axis = Field(default=None, description="Eixo do cilindro em graus, de 0 a 180")
    addition: Diopter = Field(default=None, description="Adição para perto em dioptrias")


class PrescriptionData(DocumentModel):
    critical_fields: ClassVar[tuple[str, ...]] = (
        "patient_name",
        "prescriber_name",
        "issue_date",
        "od.sphere",
        "oe.sphere",
    )

    patient_name: OptionalStr = Field(default=None, description="Nome do paciente")
    prescriber_name: OptionalStr = Field(default=None, description="Nome do médico")
    prescriber_registry: OptionalStr = Field(default=None, description="CRM do médico, com UF")
    issue_date: BRDate = Field(default=None, description="Data de emissão")
    od: EyePrescription | None = Field(default=None, description="Olho direito (OD)")
    oe: EyePrescription | None = Field(default=None, description="Olho esquerdo (OE)")
    notes: OptionalStr = Field(default=None, description="Observações (tipo de lente etc.)")
