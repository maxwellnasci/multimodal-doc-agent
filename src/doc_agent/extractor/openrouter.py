"""Adaptador OpenRouter para o NVIDIA Nemotron VL (API compatível com OpenAI)."""

from __future__ import annotations

import base64
import os
import time
from collections.abc import Callable
from typing import Any

import httpx

from doc_agent.extractor.base import (
    ExtractionResult,
    ExtractorClient,
    ExtractorConfigError,
    ExtractorError,
)
from doc_agent.extractor.prompts import SYSTEM_PROMPT, build_user_prompt, parse_json_payload
from doc_agent.schemas import DocumentType

DEFAULT_MODEL = "nvidia/nemotron-3.5-super-vl"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class OpenRouterNemotronExtractor(ExtractorClient):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        *,
        base_url: str = OPENROUTER_BASE_URL,
        timeout: float = 90.0,
        max_retries: int = 2,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self._api_key:
            raise ExtractorConfigError("defina OPENROUTER_API_KEY ou informe api_key")
        self.model_name = model or os.environ.get("DOC_AGENT_MODEL") or DEFAULT_MODEL
        self._max_retries = max_retries
        self._sleep = sleep
        self._owns_client = client is None
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout)

    def __enter__(self) -> OpenRouterNemotronExtractor:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def build_request(
        self, image: bytes, mime_type: str, document_type: DocumentType
    ) -> dict[str, Any]:
        data_uri = f"data:{mime_type};base64,{base64.b64encode(image).decode()}"
        return {
            "model": self.model_name,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_user_prompt(document_type)},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                },
            ],
        }

    def extract(
        self, image: bytes, mime_type: str, document_type: DocumentType
    ) -> ExtractionResult:
        body = self._post(self.build_request(image, mime_type, document_type))
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ExtractorError(f"resposta inesperada do OpenRouter: {body!r:.300}") from exc
        if isinstance(content, list):  # alguns provedores devolvem partes
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if not isinstance(content, str) or not content.strip():
            raise ExtractorError("o modelo retornou conteúdo vazio")
        return ExtractionResult(
            payload=parse_json_payload(content),
            model=body.get("model", self.model_name),
            raw_text=content,
            usage=body.get("usage") or {},
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "X-Title": "multimodal-doc-agent",
        }
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            if attempt:
                self._sleep(0.5 * 2 ** (attempt - 1))
            try:
                response = self._client.post("/chat/completions", json=payload, headers=headers)
            except httpx.TransportError as exc:
                last_error = exc
                continue
            if response.status_code in RETRYABLE_STATUS:
                last_error = ExtractorError(f"OpenRouter HTTP {response.status_code}")
                continue
            if response.is_error:
                raise ExtractorError(
                    f"OpenRouter HTTP {response.status_code}: {response.text[:300]}"
                )
            try:
                return response.json()
            except ValueError as exc:
                raise ExtractorError("OpenRouter retornou corpo não-JSON") from exc
        raise ExtractorError(f"falha após {self._max_retries + 1} tentativas: {last_error}")
