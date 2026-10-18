"""Benchmark ao vivo: modo AUTO x tipo específico contra o modelo real (OpenRouter).

Usa as amostras sintéticas de `doc_agent.samples` e compara cada campo extraído com o
gabarito (`SAMPLE_PAYLOADS`). Faz chamadas pagas: não roda no pytest nem no CI.

    export OPENROUTER_API_KEY=sk-or-...
    python scripts/benchmark_live.py                     # 6 chamadas (3 tipos x 2 modos)
    python scripts/benchmark_live.py --modes auto --output resultados.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from doc_agent.extractor import ExtractorClient, OpenRouterNemotronExtractor
from doc_agent.pipeline import DocumentPipeline
from doc_agent.samples import SAMPLE_PAYLOADS, render_sample_image
from doc_agent.schemas import SCHEMA_BY_TYPE, DocumentType

MODES = ("auto", "specific")
CONCRETE_TYPES = tuple(SCHEMA_BY_TYPE)


@dataclass
class BenchmarkRecord:
    mode: str
    sample: str
    detected: str
    status: str
    confidence: float
    discriminated: bool  # resposta veio como {"document_type", "data"}
    fields_total: int
    fields_correct: int
    mismatches: list[tuple[str, Any, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None
    cost_source: str = "unknown"
    latency_ms: float = 0.0
    model: str = ""
    raw_response: dict[str, Any] | None = None


def _flatten(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _flatten(item, f"{prefix}{key}.")
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            yield from _flatten(item, f"{prefix}{idx}.")
    else:
        yield prefix[:-1], value


def compare_with_ground_truth(
    document_type: DocumentType, data: dict[str, Any] | None
) -> tuple[int, list[tuple[str, Any, Any]]]:
    """Compara campo a campo (já normalizado pelo schema) com o gabarito da amostra."""
    truth_model = SCHEMA_BY_TYPE[document_type].model_validate(SAMPLE_PAYLOADS[document_type])
    truth = dict(_flatten(truth_model.model_dump(mode="json")))
    got = dict(_flatten(data or {}))
    mismatches = [(key, truth[key], got.get(key)) for key in truth if truth[key] != got.get(key)]
    return len(truth), mismatches


def run_benchmark(
    extractor: ExtractorClient,
    *,
    modes: Iterable[str] = MODES,
    types: Iterable[DocumentType] = CONCRETE_TYPES,
) -> list[BenchmarkRecord]:
    pipeline = DocumentPipeline(extractor)
    records: list[BenchmarkRecord] = []
    for mode in modes:
        for doc_type in types:
            requested = DocumentType.AUTO if mode == "auto" else doc_type
            envelope = pipeline.run(render_sample_image(doc_type), requested)
            data = envelope.data.model_dump(mode="json") if envelope.data else None
            total, mismatches = compare_with_ground_truth(doc_type, data)
            raw = envelope.raw_response or {}
            records.append(
                BenchmarkRecord(
                    mode=mode,
                    sample=doc_type.value,
                    detected=envelope.document_type.value,
                    status=envelope.status.value,
                    confidence=envelope.confidence,
                    discriminated=isinstance(raw.get("data"), dict),
                    fields_total=total,
                    fields_correct=total - len(mismatches),
                    mismatches=mismatches,
                    prompt_tokens=envelope.usage.prompt_tokens,
                    completion_tokens=envelope.usage.completion_tokens,
                    cost_usd=envelope.usage.estimated_cost_usd,
                    cost_source=envelope.usage.cost_source.value,
                    latency_ms=envelope.latency_ms,
                    model=envelope.model,
                    raw_response=envelope.raw_response,
                )
            )
    return records


def format_markdown(records: Sequence[BenchmarkRecord]) -> str:
    lines = [
        "| Amostra | Modo | Tipo detectado | Campos certos | Tokens (entrada + saída) "
        "| Custo (USD) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in records:
        cost = f"{r.cost_usd:.6f}" if r.cost_usd is not None else "—"
        lines.append(
            f"| {r.sample} | {r.mode} | {r.detected} | {r.fields_correct}/{r.fields_total} "
            f"| {r.prompt_tokens:,} + {r.completion_tokens:,} | {cost} ({r.cost_source}) |"
        )
    total = sum(r.cost_usd or 0.0 for r in records)
    lines.append(f"\nCusto total: US$ {total:.6f} em {len(records)} chamadas.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument(
        "--types",
        nargs="+",
        choices=[t.value for t in CONCRETE_TYPES],
        default=[t.value for t in CONCRETE_TYPES],
    )
    parser.add_argument("--model", help="sobrescreve DOC_AGENT_MODEL / modelo padrão")
    parser.add_argument(
        "--output", help="salva os registros completos (com resposta bruta) em JSON"
    )
    args = parser.parse_args(argv)

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("defina OPENROUTER_API_KEY para rodar o benchmark ao vivo", file=sys.stderr)
        return 2

    with OpenRouterNemotronExtractor(model=args.model, timeout=120) as extractor:
        records = run_benchmark(
            extractor, modes=args.modes, types=[DocumentType(t) for t in args.types]
        )
    print(f"Modelo: {records[0].model if records else extractor.model_name}\n")
    print(format_markdown(records))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump([asdict(r) for r in records], fh, ensure_ascii=False, indent=2, default=str)
    return 0 if all(r.fields_correct == r.fields_total for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
