"""Interface Streamlit. Execute com: streamlit run src/doc_agent/app.py"""

from __future__ import annotations

import os

import streamlit as st

from doc_agent.extractor import (
    DEFAULT_MODEL,
    ExtractorClient,
    ExtractorError,
    MockExtractorClient,
    OpenRouterNemotronExtractor,
)
from doc_agent.pipeline import DocumentPipeline
from doc_agent.samples import render_sample_image
from doc_agent.schemas import (
    CostSource,
    DocumentType,
    ExtractionEnvelope,
    ExtractionStatus,
    TokenUsage,
)

DOC_LABELS = {
    DocumentType.PRESCRIPTION: "👓 Receita oftalmológica",
    DocumentType.VEHICLE: "🚗 CRLV (documento de veículo)",
    DocumentType.INVOICE: "🧾 Nota fiscal",
    DocumentType.AUTO: "🔍 Detectar automaticamente (AUTO)",
}
STATUS_LABELS = {
    ExtractionStatus.OK: "✅ Aprovado",
    ExtractionStatus.NEEDS_REVIEW: "⚠️ Revisar",
    ExtractionStatus.FAILED: "❌ Falhou",
}
PROVIDER_MOCK = "Mock determinístico"
PROVIDER_OPENROUTER = "OpenRouter · Nemotron VL"

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero h1 {font-size: 2.1rem; margin-bottom: .2rem;}
.hero p {color: #6b7280; margin-top: 0;}
div[data-testid="stMetric"] {
  border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: .75rem 1rem;
}
</style>
"""


def build_extractor(provider: str, api_key: str, model: str) -> ExtractorClient:
    if provider == PROVIDER_OPENROUTER:
        return OpenRouterNemotronExtractor(api_key=api_key or None, model=model or None)
    return MockExtractorClient()


def render_sidebar() -> tuple[str, str, str, DocumentType]:
    with st.sidebar:
        st.header("Configuração")
        has_env_key = bool(os.environ.get("OPENROUTER_API_KEY"))
        provider = st.radio(
            "Provedor",
            [PROVIDER_MOCK, PROVIDER_OPENROUTER],
            index=1 if has_env_key else 0,
            help="O mock devolve dados fixos — ideal para demo sem chave de API.",
        )
        api_key, model = "", DEFAULT_MODEL
        if provider == PROVIDER_OPENROUTER:
            api_key = st.text_input(
                "OPENROUTER_API_KEY",
                type="password",
                placeholder="usando variável de ambiente" if has_env_key else "sk-or-...",
            )
            model = st.text_input("Modelo", value=os.environ.get("DOC_AGENT_MODEL", DEFAULT_MODEL))
        document_type = st.selectbox(
            "Tipo de documento", list(DocumentType), format_func=DOC_LABELS.__getitem__
        )
        st.caption("Guardrails: placa, chassi ISO 3779, RENAVAM, CPF/CNPJ, passo 0,25 D e eixo.")
    return provider, api_key, model, document_type


def format_cost(usage: TokenUsage) -> str:
    if usage.estimated_cost_usd is None:
        return "—"
    cost = f"${usage.estimated_cost_usd:.5f}"
    return f"~{cost} (estimado)" if usage.cost_source is CostSource.ESTIMATED else cost


def render_result(envelope: ExtractionEnvelope) -> None:
    cols = st.columns(5)
    cols[0].metric("Status", STATUS_LABELS[envelope.status])
    cols[1].metric("Confiança", f"{envelope.confidence:.0%}")
    cols[2].metric("Latência", f"{envelope.latency_ms:,.0f} ms")
    cols[3].metric("Alertas", f"{len(envelope.errors)} erros · {len(envelope.warnings)} avisos")
    usage = envelope.usage
    cols[4].metric("Custo FinOps", format_cost(usage), f"{usage.total_tokens} tok")

    data_tab, issues_tab, raw_tab = st.tabs(["Dados extraídos", "Guardrails", "Resposta bruta"])
    with data_tab:
        if envelope.data is not None:
            st.json(envelope.data.model_dump(mode="json"))
        else:
            st.info("Nenhum dado válido extraído.")
    with issues_tab:
        if envelope.issues:
            st.dataframe([issue.model_dump(mode="json") for issue in envelope.issues])
        else:
            st.success("Todas as validações determinísticas passaram.")
    with raw_tab:
        st.json(envelope.raw_response or {})

    st.download_button(
        "⬇️ Baixar JSON",
        data=envelope.model_dump_json(indent=2),
        file_name=f"{envelope.document_type.value}_extraction.json",
        mime="application/json",
    )
    st.caption(f"Modelo: `{envelope.model}` · Tipo: `{envelope.document_type.value}`")


def main() -> None:
    st.set_page_config(page_title="Multimodal Doc Agent", page_icon="🧾", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="hero"><h1>Multimodal Doc Agent</h1>'
        "<p>Extração estruturada de receitas, CRLVs e notas fiscais com visão multimodal, "
        "Pydantic v2 e guardrails determinísticos.</p></div>",
        unsafe_allow_html=True,
    )
    provider, api_key, model, document_type = render_sidebar()

    source = st.radio("Origem da imagem", ["Exemplo embutido", "Upload"], horizontal=True)
    image: bytes | None = None
    if source == "Upload":
        uploaded = st.file_uploader("Imagem do documento", type=["png", "jpg", "jpeg", "webp"])
        if uploaded is not None:
            image = uploaded.getvalue()
        if provider == PROVIDER_MOCK:
            st.warning("O mock ignora a imagem e devolve sempre o exemplo do tipo escolhido.")
    else:
        image = render_sample_image(document_type)

    left, right = st.columns([1, 1], gap="large")
    with left:
        if image:
            st.image(image, caption=DOC_LABELS[document_type])
        else:
            st.info("Envie uma imagem para começar.")
    with right:
        if st.button("Extrair dados", type="primary", disabled=not image):
            try:
                extractor = build_extractor(provider, api_key, model)
            except ExtractorError as exc:
                st.error(str(exc))
            else:
                with st.spinner("Extraindo..."):
                    try:
                        st.session_state["envelope"] = DocumentPipeline(extractor).run(
                            image, document_type
                        )
                    finally:
                        extractor.close()
        envelope = st.session_state.get("envelope")
        if envelope is not None and document_type in (envelope.document_type, DocumentType.AUTO):
            render_result(envelope)


main()
