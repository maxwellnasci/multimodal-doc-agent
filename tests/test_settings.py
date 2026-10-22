from __future__ import annotations

import pytest

from doc_agent.extractor import DEFAULT_MODEL
from doc_agent.settings import ConfigError, load_settings

_VARS = ("DOC_AGENT_ENV", "OPENROUTER_API_KEY", "DOC_AGENT_MODEL", "DOC_AGENT_ALLOWED_MODELS")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in _VARS:
        monkeypatch.delenv(name, raising=False)


def test_defaults_to_production_and_requires_key():
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        load_settings()


@pytest.mark.parametrize("key", ["", "   "])
def test_blank_key_counts_as_missing_in_production(monkeypatch, key):
    monkeypatch.setenv("OPENROUTER_API_KEY", key)
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        load_settings()


def test_production_with_key_disables_mock(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")
    settings = load_settings()
    assert settings.production
    assert not settings.mock_enabled
    assert settings.api_key == "sk-env"


def test_development_must_be_explicit_and_allows_mock_without_key(monkeypatch):
    monkeypatch.setenv("DOC_AGENT_ENV", " Development ")
    settings = load_settings()
    assert not settings.production
    assert settings.mock_enabled
    assert settings.api_key is None


@pytest.mark.parametrize("env", ["prod", "dev", "staging", ""])
def test_unknown_env_is_rejected(monkeypatch, env):
    monkeypatch.setenv("DOC_AGENT_ENV", env)
    with pytest.raises(ConfigError, match="DOC_AGENT_ENV"):
        load_settings()


def test_model_defaults_to_allowlist(monkeypatch):
    monkeypatch.setenv("DOC_AGENT_ENV", "development")
    settings = load_settings()
    assert settings.allowed_models == (DEFAULT_MODEL,)
    assert settings.default_model == DEFAULT_MODEL


def test_model_outside_allowlist_is_rejected(monkeypatch):
    monkeypatch.setenv("DOC_AGENT_ENV", "development")
    monkeypatch.setenv("DOC_AGENT_MODEL", "openai/caro")
    with pytest.raises(ConfigError, match="DOC_AGENT_ALLOWED_MODELS"):
        load_settings()


def test_allowlist_is_parsed_and_deduplicated(monkeypatch):
    monkeypatch.setenv("DOC_AGENT_ENV", "development")
    monkeypatch.setenv("DOC_AGENT_ALLOWED_MODELS", " a/x , b/y,,a/x ")
    monkeypatch.setenv("DOC_AGENT_MODEL", "b/y")
    settings = load_settings()
    assert settings.allowed_models == ("a/x", "b/y")
    assert settings.default_model == "b/y"
