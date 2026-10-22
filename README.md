# multimodal-doc-agent

![Python 3.11 | 3.12 | 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white)
[![CI](https://github.com/maxwellnasci/multimodal-doc-agent/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/maxwellnasci/multimodal-doc-agent/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![NVIDIA Nemotron 3.5 Super VL](https://img.shields.io/badge/NVIDIA-Nemotron%203.5%20Super%20VL-76B900?logo=nvidia&logoColor=white)
[![Licença MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Transforma a **foto de um documento brasileiro** — receita oftalmológica, CRLV ou nota fiscal —
em **JSON validado**, usando visão multimodal (NVIDIA Nemotron VL via OpenRouter), **Pydantic v2** e
**guardrails determinísticos** que barram o que o modelo leu errado antes de o dado seguir adiante.

**Destaques**

- **41/41 campos corretos** no benchmark com o modelo real, a **frações de centavo de dólar por
  documento** (US$ 0,0005–0,0043, custo informado pelo OpenRouter) — [ver benchmark e limites](#benchmark-real-nemotron-35-super-vl-via-openrouter).
- **Detecção automática do tipo** de documento em uma única chamada (union discriminada).
- **Validação fiscal de verdade:** dígitos verificadores, CNPJ alfanumérico (NT 2025.001) e
  validação cruzada da chave NF-e contra os campos do DANFE.
- **Nada de custo ou dado inventado:** o envelope diz de onde veio o custo e por que um documento
  precisa de revisão humana.
- **292 testes sem rede**, CI em Python 3.11–3.13, `ruff` limpo.

## Arquitetura (Ports & Adapters)

```
imagem ─► DocumentPipeline
            ├─► validação da imagem   (magic bytes, tamanho) + downscale com orientação EXIF
            ├─► ExtractorClient (porta)
            │     ├─ OpenRouterNemotronExtractor  (adapter HTTP: retries, Retry-After, custo somado)
            │     └─ MockExtractorClient          (adapter determinístico, testes e demo)
            ├─► roteamento AUTO       ({"document_type", "data"} → schema do tipo detectado)
            ├─► schema Pydantic       (normaliza datas, R$, quantidades, dioptrias, placas, CPF/CNPJ)
            ├─► guardrails            (dígitos verificadores, regras clínicas e fiscais, chave NF-e)
            └─► ExtractionEnvelope    (dados + issues + confiança + status + tokens/custo)
```

| Módulo | Responsabilidade |
| --- | --- |
| `schemas/` | `PrescriptionData` (OD/OE), `VehicleDocumentData`, `InvoiceData`, `ExtractionEnvelope` |
| `guardrails/validators.py` | Placa Mercosul/antiga, VIN ISO 3779, RENAVAM, CPF, CNPJ (inclusive alfanumérico), chave NF-e, passo 0,25 D, eixo 0–180 |
| `guardrails/checks.py` | Regras por documento: cilindro exige eixo, soma dos itens = total, ano modelo × fabricação… |
| `extractor/` | Porta `ExtractorClient`, adapter OpenRouter (retries, parsing robusto de JSON) e mock |
| `pipeline.py` | Orquestração, validação da imagem por magic bytes e score de confiança |
| `app.py` | Interface Streamlit com upload, exemplos embutidos, métricas e download do JSON |
| `scripts/benchmark_live.py` | Benchmark pago contra o modelo real: AUTO × específico, campo a campo contra o gabarito |

## Defesa anti-alucinação em 3 camadas

Um modelo de visão pode devolver um JSON bem formado com valores inventados ou mal lidos. Nenhum
valor chega ao envelope como `ok` sem passar por três camadas determinísticas:

1. **Schema Pydantic v2** — tipos estritos e normalização de formatos brasileiros (datas
   `dd/mm/aaaa`, `R$ 1.234,56`, dioptrias com vírgula, placas, CPF/CNPJ). `NaN`/`Infinity` são
   rejeitados, e quantidades (`1.500` = 1,5) são tratadas separadamente de valores em reais
   (`R$ 1.500` = 1500). Campo com tipo errado → `failed`, com o caminho do campo no erro.
2. **Guardrails de domínio** — dígitos verificadores (CPF, CNPJ numérico e alfanumérico, RENAVAM,
   chave NF-e), formatos (placa Mercosul/antiga, VIN ISO 3779), regras clínicas (passo de 0,25 D,
   faixas de grau, eixo 0–180°, cilindro exige eixo) e fiscais (soma dos itens = total,
   quantidade × unitário = total do item), além de coerência temporal (datas futuras, ano modelo
   × fabricação).
3. **Score de confiança e revisão humana** — completude dos campos críticos − 0,25 por erro −
   0,05 por aviso. Status `ok` exige zero erros e confiança ≥ 0,8; senão o documento vai para
   `needs_review` em vez de ser aceito. Falha de schema ou do extrator → `failed`.

As camadas pegam valores **impossíveis ou inconsistentes**. Um valor plausível e bem formado,
mas errado (um nome trocado, por exemplo), passa por elas; por isso o status `needs_review` e a
resposta bruta no envelope existem para auditoria.

### Exemplo: um dígito lido errado

O modelo leu o RENAVAM `00639884962` como `00639884961` — um JSON perfeitamente válido. O
guardrail confere o dígito verificador e o documento vai para revisão em vez de ser aceito.
Saída do pipeline, resumida, com a leitura errada simulada pelo extrator mock:

```json
{
  "document_type": "vehicle_document",
  "status": "needs_review",
  "confidence": 0.75,
  "data": { "plate": "BRA2E19", "renavam": "00639884961", "model_year": 2022, "...": "..." },
  "issues": [
    {
      "field": "renavam",
      "code": "invalid_renavam",
      "message": "RENAVAM com dígito verificador inválido",
      "severity": "error"
    }
  ]
}
```

### Validação cruzada da NF-e

A chave de acesso de 44 posições é decomposta (cUF, AAMM, CNPJ/CPF do emitente, modelo, série,
número) e conferida contra os campos lidos no DANFE:

- **DV da chave** pelo módulo 11 sobre `ASCII − 48`, como manda a NT 2025.001, aceitando o
  **CNPJ alfanumérico** só nas 12 posições da raiz;
- **emitente** da chave = emitente extraído (inclusive `000` + CPF de produtor rural), e o
  CNPJ/CPF da chave precisa ser válido mesmo quando o emitente não foi extraído;
- **AAMM** da chave = mês/ano de emissão; **número** e **série** da chave = os da nota.

Qualquer divergência vira erro ou aviso com o valor da chave e o valor lido lado a lado.

## Decisões técnicas

| Decisão | Alternativa descartada | Por quê |
| --- | --- | --- |
| **Ports & Adapters** (`ExtractorClient`) | Chamar a API direto no pipeline | Trocar de modelo/provedor sem tocar em schema e guardrails; testes rodam com o mock e com `httpx.MockTransport`, sem rede. |
| **AUTO como union discriminada** em 1 chamada | Classificar e depois extrair (2 chamadas) | Medido: o schema extra custa ~1.450 tokens de entrada (~US$ 0,00015); a 2ª chamada reenviaria a imagem e dobraria a latência. |
| **Custo vindo do provedor** (`usage.cost`), com fonte declarada | Tabela de preços fixa no código | A tabela erra quando o modelo ou o roteamento muda — no benchmark, uma chamada custou 3× o preço de tabela. Sem dado, o custo é `None`, não um chute. |
| **Guardrails determinísticos** | Pedir ao LLM que confira a própria saída | Dígito verificador e soma de itens são regras exatas: mais baratas, reprodutíveis e testáveis que uma segunda opinião do modelo. |
| **Tokens de novas tentativas somados** | Contar só a resposta final | Tentativas com `finish_reason=error` também são cobradas; o envelope de falha mostra o custo real. |

## Uso

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

# Local, sem chave de API (mock determinístico):
export DOC_AGENT_ENV=development
streamlit run src/doc_agent/app.py

# Com o modelo real:
cp .env.example .env                       # preencha OPENROUTER_API_KEY
set -a && source .env && set +a            # o projeto lê variáveis de ambiente, não o arquivo
python -m doc_agent.settings && streamlit run src/doc_agent/app.py
```

**Fail-closed:** sem `DOC_AGENT_ENV`, o app assume `production`. Nesse modo, ele se recusa a
subir sem `OPENROUTER_API_KEY` e o mock fica desabilitado, para que um deploy com o secret mal
configurado não exiba dados fictícios como "Aprovado". O mock exige
`DOC_AGENT_ENV=development` explícito. `python -m doc_agent.settings` valida a configuração e
sai com código 1 se algo estiver errado; use no entrypoint do container.

```python
from doc_agent.extractor import OpenRouterNemotronExtractor
from doc_agent.pipeline import DocumentPipeline
from doc_agent.schemas import DocumentType

with OpenRouterNemotronExtractor() as extractor:
    envelope = DocumentPipeline(extractor).run(open("crlv.jpg", "rb").read(), DocumentType.VEHICLE)
print(envelope.model_dump_json(indent=2))
```

O modelo padrão é `nvidia/nemotron-3.5-super-vl` e pode ser trocado via `DOC_AGENT_MODEL`.
Na UI, só são aceitos os modelos de `DOC_AGENT_ALLOWED_MODELS` (lista separada por vírgula;
padrão: apenas o modelo padrão). Assim, um visitante não escolhe um modelo caro pago pela sua chave.

**Custo (FinOps):** `envelope.usage.estimated_cost_usd` vem do `usage.cost` que o OpenRouter
devolve em toda resposta (`cost_source="provider"`). Para provedores que não informam custo,
passe os preços do modelo: `DocumentPipeline(extractor, pricing=ModelPricing(0.10, 0.40))`
(USD por 1M de tokens de entrada/saída, `cost_source="estimated"`). Sem nenhum dos dois, o
custo fica `None` (`cost_source="unknown"`) em vez de um valor inventado.

## Benchmark real (Nemotron 3.5 Super VL via OpenRouter)

Execução de 18/10/2026 com `nvidia/nemotron-3.5-super-vl` sobre as três amostras sintéticas do
projeto. Cada campo extraído foi comparado com o gabarito (`SAMPLE_PAYLOADS`). Os custos são os
informados pelo próprio OpenRouter em `usage.cost` de cada resposta.

| Amostra | Modo | Tipo detectado | Campos certos | Tokens (entrada + saída) | Custo (USD) |
| --- | --- | --- | --- | --- | --- |
| Receita oftalmológica | AUTO | ✅ prescription | **13/13** | 2.932 + 1.291 | 0,00081 |
| CRLV | AUTO | ✅ vehicle_document | **11/11** | 2.932 + 587 | 0,00053 |
| Nota fiscal (DANFE) | AUTO | ✅ invoice | **17/17** | 2.933 + 2.524 | 0,00130 |
| Receita oftalmológica | Específico | — | **13/13** | 1.411 + 1.023 | 0,00072 |
| CRLV | Específico | — | **11/11** | 1.368 + 1.118 | 0,00058 |
| Nota fiscal (DANFE) | Específico | — | **17/17** | 1.554 + 3.132 | 0,00430¹ |

- **41/41 campos corretos nos dois modos**; o modo AUTO identificou o tipo e respondeu no formato
  discriminado `{"document_type", "data"}` nas três amostras.
- O modo AUTO usa ~1.450 tokens de entrada a mais (o schema dos três tipos vai no prompt),
  cerca de US$ 0,00015 por chamada. O custo é dominado pelos tokens de saída, que variam bastante
  entre execuções da mesma imagem.
- ¹ Custo ~3× acima do preço de tabela para os mesmos tokens; provavelmente o OpenRouter
  roteou essa chamada para outro provedor. É o valor cobrado, por isso é ele que aparece.

**Limites:** uma chamada por caso, em imagens sintéticas e limpas. O resultado não vale como
estimativa de precisão para fotos reais (inclinação, sombra, reflexo, baixa resolução).

Para reproduzir (faz chamadas pagas, cerca de US$ 0,01 no total):

```bash
export OPENROUTER_API_KEY=sk-or-...
python scripts/benchmark_live.py --output resultados.json
```

O script sai com código 1 se algum campo divergir do gabarito e 2 se não houver chave.

## Qualidade

```bash
pytest --cov=doc_agent
ruff check . && ruff format --check .
```

A suíte roda sem rede (o OpenRouter é simulado com `httpx.MockTransport`). O
[CI](.github/workflows/ci.yml) roda `ruff` e os testes (com `-W error` e cobertura) no Python 3.11,
3.12 e 3.13 a cada push na `main` e em todo pull request. O benchmark pago não roda no CI.

Todos os dados de exemplo são fictícios (com dígitos verificadores válidos).

## Limitações e próximos passos

- **Fotos reais:** o benchmark usa imagens sintéticas e limpas. O próximo passo é um conjunto de
  fotos reais anonimizadas (inclinação, sombra, reflexo) para medir precisão por campo.
- **Variância de custo:** os tokens de saída variam muito entre execuções da mesma imagem; vale
  testar o controle de raciocínio do modelo (`reasoning` no OpenRouter) contra precisão e latência.
- **Valores plausíveis e errados** (um nome trocado) passam pelos guardrails; para esses casos o
  caminho é revisão humana guiada pelo status `needs_review`.

## Licença

Código sob a [licença MIT](LICENSE). A fonte DejaVu Sans, usada para renderizar as amostras,
é distribuída sob a licença própria em
[`src/doc_agent/assets/fonts/LICENSE-DejaVu.txt`](src/doc_agent/assets/fonts/LICENSE-DejaVu.txt).
