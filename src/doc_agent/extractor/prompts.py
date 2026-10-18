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
}
DOCUMENT_HINTS[DocumentType.AUTO] = (
    "Identifique o tipo do documento brasileiro na imagem e responda no formato "
    '{"document_type": "<tipo>", "data": {<campos do schema desse tipo>}}, usando só os '
    "campos do tipo escolhido. Tipos:\n"
    + "\n".join(f"- {t.value}: {DOCUMENT_HINTS[t]}" for t in SCHEMA_BY_TYPE)
)

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


def build_auto_schema() -> dict[str, Any]:
    """Union discriminada: cada tipo mantém seu schema intacto sob `data`, sem mesclar campos."""
    branches: list[dict[str, Any]] = []
    defs: dict[str, Any] = {}
    for doc_type, doc_schema in SCHEMA_BY_TYPE.items():
        schema = doc_schema.model_json_schema()
        # $refs apontam para "#/$defs/..." da raiz, então os $defs sobem para o schema externo.
        for name, definition in schema.pop("$defs", {}).items():
            if defs.setdefault(name, definition) != definition:
                raise ValueError(f"$defs conflitante entre schemas: {name}")
        branches.append(
            {
                "type": "object",
                "properties": {"document_type": {"const": doc_type.value}, "data": schema},
                "required": ["document_type", "data"],
            }
        )
    auto: dict[str, Any] = {"anyOf": branches}
    if defs:
        auto["$defs"] = defs
    return auto


def build_user_prompt(document_type: DocumentType) -> str:
    if document_type is DocumentType.AUTO:
        schema = build_auto_schema()
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


def resolve_auto_payload(
    payload: Mapping[str, Any],
) -> tuple[DocumentType | None, Mapping[str, Any]]:
    """Separa tipo e dados da resposta AUTO; tolera o modelo devolver os campos sem `data`."""
    data = payload.get("data")
    if isinstance(data, Mapping):
        doc_type = infer_document_type({"document_type": payload.get("document_type")})
        return doc_type or infer_document_type(data), data
    return infer_document_type(payload), payload


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
