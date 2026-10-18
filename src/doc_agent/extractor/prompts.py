"""Prompts de extração: o schema Pydantic é a fonte da verdade do formato de saída."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from doc_agent.extractor.base import ExtractorError
from doc_agent.schemas import SCHEMA_BY_TYPE, DocumentType

DOCUMENT_HINTS: dict[DocumentType, str] = {
    DocumentType.PRESCRIPTION: (
        "Receita oftalmológica brasileira. OD = olho direito, OE = olho esquerdo. "
        "Graus em dioptrias com sinal (ex.: -2.25); 'plano' equivale a 0. Eixo em graus."
    ),
    DocumentType.VEHICLE: (
        "CRLV brasileiro. Placa Mercosul (ABC1D23) ou antiga (ABC1234), chassi/VIN de "
        "17 caracteres, RENAVAM de 11 dígitos, CPF/CNPJ do proprietário."
    ),
    DocumentType.INVOICE: (
        "Nota fiscal brasileira (DANFE/NF-e). Extraia emitente, destinatário, chave de acesso "
        "de 44 caracteres (copie letras do CNPJ alfanumérico como estão), cada item com "
        "quantidade, valor unitário e total, e o valor total da nota."
    ),
    DocumentType.AUTO: (
        "Identifique automaticamente o tipo de documento brasileiro na imagem "
        "('prescription', 'vehicle_document' ou 'invoice'), inclua a chave 'document_type' "
        "e extraia os campos correspondentes ao schema desse documento."
    ),
}

_SIGNATURE_FIELDS: dict[DocumentType, frozenset[str]] = {
    DocumentType.PRESCRIPTION: frozenset(
        {"od", "oe", "patient_name", "prescriber_name", "prescriber_registry"}
    ),
    DocumentType.VEHICLE: frozenset(
        {
            "plate",
            "chassis",
            "renavam",
            "brand_model",
            "manufacture_year",
            "model_year",
            "licensing_year",
            "owner_name",
            "owner_document",
        }
    ),
    DocumentType.INVOICE: frozenset(
        {
            "invoice_number",
            "series",
            "access_key",
            "issuer_name",
            "issuer_document",
            "recipient_name",
            "recipient_document",
            "items",
            "total_amount",
        }
    ),
}

SYSTEM_PROMPT = (
    "Você é um extrator de dados de documentos. Responda APENAS com um objeto JSON válido "
    "que siga o JSON Schema fornecido. Transcreva os valores exatamente como aparecem; "
    "use null para campos ilegíveis ou ausentes e nunca invente valores."
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CLOSE_THINK_RE = re.compile(r"^.*</think>", re.DOTALL | re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r"<think>", re.IGNORECASE)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def build_user_prompt(document_type: DocumentType) -> str:
    if document_type is DocumentType.AUTO:
        properties: dict[str, Any] = {
            "document_type": {
                "type": "string",
                "enum": [t.value for t in SCHEMA_BY_TYPE],
            }
        }
        defs: dict[str, Any] = {}
        for doc_schema in SCHEMA_BY_TYPE.values():
            sub = doc_schema.model_json_schema()
            properties.update(sub.get("properties", {}))
            defs.update(sub.get("$defs", {}))
        schema: dict[str, Any] = {"type": "object", "properties": properties}
        if defs:
            schema["$defs"] = defs
    else:
        schema = SCHEMA_BY_TYPE[document_type].model_json_schema()
    return (
        f"{DOCUMENT_HINTS[document_type]}\n\n"
        f"JSON Schema de saída:\n{json.dumps(schema, ensure_ascii=False)}"
    )


def infer_document_type(payload: Mapping[str, Any]) -> DocumentType | None:
    """Detecta o DocumentType concreto a partir de `document_type` explícito ou chaves típicas."""
    explicit = payload.get("document_type")
    if isinstance(explicit, str):
        normalized = explicit.strip().lower()
        for doc_type in SCHEMA_BY_TYPE:
            if normalized in (doc_type.value, doc_type.name.lower()):
                return doc_type

    present_keys = {k for k, val in payload.items() if val not in (None, "", [])}
    best_type: DocumentType | None = None
    best_score = 0
    for doc_type, sig_keys in _SIGNATURE_FIELDS.items():
        score = len(present_keys & sig_keys)
        if score > best_score:
            best_score = score
            best_type = doc_type
    return best_type


def _strip_thinking(text: str) -> str:
    cleaned = _THINK_RE.sub("", text)
    cleaned = _CLOSE_THINK_RE.sub("", cleaned)
    return _FENCE_RE.sub("", cleaned.strip()).strip()


def parse_json_payload(text: str) -> dict[str, Any]:
    """Extrai o JSON da resposta, tolerando <think> (mesmo truncado), cercas e texto extra."""
    cleaned = _strip_thinking(text)
    has_unclosed_think = _OPEN_THINK_RE.search(cleaned) is not None

    if not has_unclosed_think:
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            payload = None
        if payload is not None:
            if not isinstance(payload, dict):
                raise ExtractorError("a resposta do modelo deve ser um objeto JSON")
            return payload

    decoder = json.JSONDecoder()
    positions = [idx for idx, ch in enumerate(cleaned) if ch == "{"]
    if not positions:
        raise ExtractorError("resposta do modelo não contém JSON")

    # Com <think> truncado sem </think>, prioriza o último objeto JSON válido.
    ordered_positions = reversed(positions) if has_unclosed_think else positions
    last_exc: json.JSONDecodeError | None = None
    for pos in ordered_positions:
        try:
            candidate, _ = decoder.raw_decode(cleaned, pos)
        except json.JSONDecodeError as exc:
            last_exc = exc
            continue
        if isinstance(candidate, dict):
            return candidate
    raise ExtractorError(f"JSON inválido na resposta do modelo: {last_exc}") from last_exc
