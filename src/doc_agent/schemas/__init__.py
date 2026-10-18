from doc_agent.schemas.common import DocumentType
from doc_agent.schemas.envelope import (
    DocumentData,
    ExtractionEnvelope,
    ExtractionStatus,
    GuardrailIssue,
    Severity,
)
from doc_agent.schemas.invoice import InvoiceData, InvoiceItem
from doc_agent.schemas.prescription import EyePrescription, PrescriptionData
from doc_agent.schemas.vehicle import VehicleDocumentData

SCHEMA_BY_TYPE: dict[DocumentType, type[DocumentData]] = {
    DocumentType.PRESCRIPTION: PrescriptionData,
    DocumentType.VEHICLE: VehicleDocumentData,
    DocumentType.INVOICE: InvoiceData,
}

__all__ = [
    "SCHEMA_BY_TYPE",
    "DocumentData",
    "DocumentType",
    "ExtractionEnvelope",
    "ExtractionStatus",
    "EyePrescription",
    "GuardrailIssue",
    "InvoiceData",
    "InvoiceItem",
    "PrescriptionData",
    "Severity",
    "VehicleDocumentData",
]
