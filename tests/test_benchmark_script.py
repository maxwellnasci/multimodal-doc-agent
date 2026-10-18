"""Testa a lógica do benchmark ao vivo sem rede, usando o mock no lugar do OpenRouter."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from doc_agent.extractor import MockExtractorClient
from doc_agent.samples import SAMPLE_PAYLOADS
from doc_agent.schemas import DocumentType

_SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_live.py"
_spec = importlib.util.spec_from_file_location("benchmark_live", _SCRIPT)
assert _spec is not None
assert _spec.loader is not None
benchmark = importlib.util.module_from_spec(_spec)
sys.modules["benchmark_live"] = benchmark  # dataclasses resolvem o módulo pelo nome
_spec.loader.exec_module(benchmark)


@pytest.mark.parametrize("doc_type", list(benchmark.CONCRETE_TYPES))
def test_specific_mode_with_mock_scores_every_field(doc_type):
    (record,) = benchmark.run_benchmark(MockExtractorClient(), modes=["specific"], types=[doc_type])
    assert record.detected == doc_type.value
    assert record.fields_correct == record.fields_total > 0
    assert record.mismatches == []
    assert not record.discriminated


def test_auto_mode_records_detected_type_and_discriminated_format():
    (record,) = benchmark.run_benchmark(
        MockExtractorClient(auto_type=DocumentType.VEHICLE),
        modes=["auto"],
        types=[DocumentType.VEHICLE],
    )
    assert record.detected == "vehicle_document"
    assert record.discriminated
    assert record.fields_correct == record.fields_total


def test_wrong_values_are_reported_as_mismatches():
    wrong = {**SAMPLE_PAYLOADS[DocumentType.VEHICLE], "plate": "XYZ9Z99"}
    (record,) = benchmark.run_benchmark(
        MockExtractorClient({DocumentType.VEHICLE: wrong}),
        modes=["specific"],
        types=[DocumentType.VEHICLE],
    )
    assert record.fields_correct == record.fields_total - 1
    assert [m[0] for m in record.mismatches] == ["plate"]


def test_markdown_table_and_total_cost():
    records = benchmark.run_benchmark(MockExtractorClient(), modes=["specific"])
    table = benchmark.format_markdown(records)
    assert table.count("\n| ") == len(records) + 1  # separador + uma linha por amostra
    assert "13/13" in table
    assert "Custo total: US$ 0.000000 em 3 chamadas." in table  # mock não informa custo


def test_main_without_api_key_exits_before_any_call(monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert benchmark.main([]) == 2
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err
