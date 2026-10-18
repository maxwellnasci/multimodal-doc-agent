# multimodal-doc-agent

Agente de extração estruturada de documentos brasileiros — **receitas oftalmológicas**, **CRLV** e
**notas fiscais** — usando visão multimodal (NVIDIA Nemotron VL via OpenRouter), **Pydantic v2** e
**guardrails determinísticos**.

## Arquitetura (Ports & Adapters)

```
imagem ─► DocumentPipeline ─► ExtractorClient (porta)
                │                 ├─ OpenRouterNemotronExtractor  (adapter HTTP)
                │                 └─ MockExtractorClient          (adapter determinístico)
                ├─► schema Pydantic  (normaliza datas, R$, dioptrias, placas, CPF/CNPJ)
                ├─► guardrails       (regex + dígitos verificadores, regras clínicas e fiscais)
                └─► ExtractionEnvelope (dados + issues + confiança + latência + status)
```

| Módulo | Responsabilidade |
| --- | --- |
| `schemas/` | `PrescriptionData` (OD/OE), `VehicleDocumentData`, `InvoiceData`, `ExtractionEnvelope` |
| `guardrails/validators.py` | Placa Mercosul/antiga, VIN ISO 3779, RENAVAM, CPF, CNPJ (inclusive alfanumérico), chave NF-e, passo 0,25 D, eixo 0–180 |
| `guardrails/checks.py` | Regras por documento: cilindro exige eixo, soma dos itens = total, ano modelo × fabricação… |
| `extractor/` | Porta `ExtractorClient`, adapter OpenRouter (retries, parsing robusto de JSON) e mock |
| `pipeline.py` | Orquestração, validação da imagem por magic bytes e score de confiança |
| `app.py` | Interface Streamlit com upload, exemplos embutidos, métricas e download do JSON |

**Score de confiança** = completude dos campos críticos − 0,25 por erro − 0,05 por aviso.
Status `ok` exige zero erros e confiança ≥ 0,8; senão `needs_review`. Falha de schema/extrator → `failed`.

## Uso

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

streamlit run src/doc_agent/app.py        # demo (mock por padrão)
export OPENROUTER_API_KEY=sk-or-...       # habilita o Nemotron real
```

```python
from doc_agent.extractor import OpenRouterNemotronExtractor
from doc_agent.pipeline import DocumentPipeline
from doc_agent.schemas import DocumentType

with OpenRouterNemotronExtractor() as extractor:
    envelope = DocumentPipeline(extractor).run(open("crlv.jpg", "rb").read(), DocumentType.VEHICLE)
print(envelope.model_dump_json(indent=2))
```

O modelo padrão é `nvidia/nemotron-3.5-super-vl` e pode ser trocado via `DOC_AGENT_MODEL`.

## Qualidade

```bash
pytest --cov=doc_agent
ruff check src tests && ruff format --check src tests
```

Todos os dados de exemplo são fictícios (com dígitos verificadores válidos).
