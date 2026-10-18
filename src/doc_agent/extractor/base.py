"""Porta (interface) de extração: qualquer provedor multimodal implementa ExtractorClient."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from doc_agent.schemas import DocumentType


class ExtractorError(RuntimeError):
    """Falha ao obter ou interpretar a resposta do modelo.

    `usage` traz os tokens/custo já cobrados até a falha (ex.: tentativas anteriores).
    """

    def __init__(self, message: str, *, usage: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.usage: dict[str, Any] = usage or {}


class ExtractorConfigError(ExtractorError):
    """Configuração ausente/inválida (ex.: sem API key)."""


@dataclass(frozen=True)
class ExtractionResult:
    payload: dict[str, Any]
    model: str
    raw_text: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


class ExtractorClient(ABC):
    model_name: str

    @abstractmethod
    def extract(
        self, image: bytes, mime_type: str, document_type: DocumentType
    ) -> ExtractionResult:
        """Extrai um dicionário compatível com o schema do `document_type`."""

    def close(self) -> None:  # noqa: B027 - hook opcional para liberar recursos
        """Libera recursos (conexões HTTP etc.). Padrão: nada a fazer."""
