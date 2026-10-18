"""Prompts de extração: o schema Pydantic é a fonte da verdade do formato de saída."""

from __future__ import annotations

import json
import re
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
        "de 44 dígitos, cada item com quantidade, valor unitário e total, e o valor total da nota."
    ),
}

SYSTEM_PROMPT = (
    "Você é um extrator de dados de documentos. Responda APENAS com um objeto JSON válido "
    "que siga o JSON Schema fornecido. Transcreva os valores exatamente como aparecem; "
    "use null para campos ilegíveis ou ausentes e nunca invente valores."
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def build_user_prompt(document_type: DocumentType) -> str:
    schema = SCHEMA_BY_TYPE[document_type].model_json_schema()
    return (
        f"{DOCUMENT_HINTS[document_type]}\n\n"
        f"JSON Schema de saída:\n{json.dumps(schema, ensure_ascii=False)}"
    )


def parse_json_payload(text: str) -> dict[str, Any]:
    """Extrai o objeto JSON da resposta, tolerando <think>, cercas markdown e texto ao redor."""
    cleaned = _FENCE_RE.sub("", _THINK_RE.sub("", text).strip()).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ExtractorError("resposta do modelo não contém JSON") from None
        try:
            payload = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ExtractorError(f"JSON inválido na resposta do modelo: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExtractorError("a resposta do modelo deve ser um objeto JSON")
    return payload
