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
# Campos de `usage` somados entre tentativas: toda resposta HTTP 200 pode ter sido cobrada.
_SUMMED_USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens", "cost")


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
        body, usage = self._post(self.build_request(image, mime_type, document_type))
        try:
            return self._build_result(body, usage)
        except ExtractorError as exc:
            exc.usage = usage  # a resposta chegou (e foi cobrada) mesmo sem JSON utilizável
            raise

    def _build_result(self, body: dict[str, Any], usage: dict[str, Any]) -> ExtractionResult:
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
            usage=usage,
        )

    def _post(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Envia com retry; devolve o corpo final e o `usage` somado de todas as tentativas."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "X-Title": "multimodal-doc-agent",
        }
        last_error: Exception | None = None
        retry_after_delay: float | None = None
        usage: dict[str, Any] = {}
        for attempt in range(self._max_retries + 1):
            if attempt:
                delay = (
                    retry_after_delay if retry_after_delay is not None else 0.5 * 2 ** (attempt - 1)
                )
                retry_after_delay = None
                self._sleep(delay)
            try:
                response = self._client.post("/chat/completions", json=payload, headers=headers)
            except httpx.TransportError as exc:
                last_error = exc
                continue
            if response.status_code in RETRYABLE_STATUS:
                retry_after_delay = _parse_retry_after(response.headers.get("retry-after"))
                last_error = ExtractorError(f"OpenRouter HTTP {response.status_code}")
                continue
            if response.is_error:
                raise ExtractorError(
                    f"OpenRouter HTTP {response.status_code}: {response.text[:300]}", usage=usage
                )
            try:
                body = response.json()
            except ValueError as exc:
                raise ExtractorError("OpenRouter retornou corpo não-JSON", usage=usage) from exc
            _add_usage(usage, body)
            if isinstance(body, dict) and "error" in body:
                err = body["error"]
                code = err.get("code") if isinstance(err, dict) else None
                msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                try:
                    code_int = int(code) if code is not None else None
                except (TypeError, ValueError):
                    code_int = None
                if code_int is None or code_int in RETRYABLE_STATUS or code_int >= 500:
                    last_error = ExtractorError(f"OpenRouter erro upstream ({code}): {msg}")
                    continue
                raise ExtractorError(f"OpenRouter HTTP {code_int}: {msg}", usage=usage)
            if _has_finish_reason_error(body):
                last_error = ExtractorError("OpenRouter retornou finish_reason=error")
                continue
            return body, usage
        raise ExtractorError(
            f"falha após {self._max_retries + 1} tentativas: {last_error}", usage=usage
        )


def _add_usage(total: dict[str, Any], body: Any) -> None:
    usage = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(usage, dict):
        return
    for key in _SUMMED_USAGE_KEYS:
        value = usage.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            total[key] = total.get(key, 0) + value


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return min(max(parsed, 0.0), 30.0)


def _has_finish_reason_error(body: Any) -> bool:
    if not isinstance(body, dict):
        return False
    choices = body.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0].get("finish_reason") == "error"
    return False
