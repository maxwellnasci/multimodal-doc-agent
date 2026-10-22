"""Configuração do ambiente, lida uma vez e validada no boot (fail-closed).

Execute `python -m doc_agent.settings` no entrypoint para abortar antes de subir o servidor.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from doc_agent.extractor import DEFAULT_MODEL

DEVELOPMENT = "development"
PRODUCTION = "production"
_ENVS = frozenset({DEVELOPMENT, PRODUCTION})


class ConfigError(RuntimeError):
    """Configuração ausente, inválida ou insegura: o app não deve subir."""


@dataclass(frozen=True)
class Settings:
    env: str
    api_key: str | None
    allowed_models: tuple[str, ...]
    default_model: str

    @property
    def production(self) -> bool:
        return self.env == PRODUCTION

    @property
    def mock_enabled(self) -> bool:
        # O mock devolve dados fixos como "Aprovado": só com opt-in explícito de desenvolvimento.
        return self.env == DEVELOPMENT


def load_settings() -> Settings:
    env = os.environ.get("DOC_AGENT_ENV", PRODUCTION).strip().lower()  # padrão seguro
    if env not in _ENVS:
        raise ConfigError(f"DOC_AGENT_ENV={env!r} inválido; use {' ou '.join(sorted(_ENVS))}")

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip() or None
    if env == PRODUCTION and api_key is None:
        raise ConfigError(
            "OPENROUTER_API_KEY é obrigatória em produção; para rodar localmente sem chave "
            "(mock), exporte DOC_AGENT_ENV=development"
        )

    allowed = tuple(
        dict.fromkeys(
            m.strip()
            for m in os.environ.get("DOC_AGENT_ALLOWED_MODELS", "").split(",")
            if m.strip()
        )
    ) or (DEFAULT_MODEL,)
    default_model = os.environ.get("DOC_AGENT_MODEL", "").strip() or allowed[0]
    if default_model not in allowed:
        raise ConfigError(
            f"DOC_AGENT_MODEL={default_model!r} fora de DOC_AGENT_ALLOWED_MODELS "
            f"({', '.join(allowed)})"
        )

    return Settings(env=env, api_key=api_key, allowed_models=allowed, default_model=default_model)


if __name__ == "__main__":
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"configuração inválida: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    print(f"configuração ok (DOC_AGENT_ENV={settings.env})")
