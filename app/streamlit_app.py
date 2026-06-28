from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import streamlit as st
except ModuleNotFoundError as exc:
    raise SystemExit("Install UI dependencies with: pip install -r requirements.txt") from exc

from evidenceiq.agents import InvestigationAgent
from evidenceiq.case import InvestigationCase
from evidenceiq.entities import enrich_entities
from evidenceiq.llm import DEFAULT_GROQ_MODEL, SAFE_GROQ_MODELS, GroqLLMClient
from evidenceiq.parsing import deduplicate, parse_uploaded_bytes
from evidenceiq.pipeline import build_case_from_folder
from evidenceiq.search import EvidenceSearch
from evidenceiq.storage import EvidenceStore


@st.cache_resource
def load_case():
    return build_case_from_folder(ROOT / "data" / "sample_case")


st.set_page_config(page_title="EvidenceIQ", page_icon="EI", layout="wide")
st.title("EvidenceIQ")
st.caption("Agentic AI Investigation Command Center")

if "evidence_items" not in st.session_state:
    st.session_state["evidence_items"] = load_case().items

with st.sidebar:
    configured_model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    default_model = configured_model if configured_model in SAFE_GROQ_MODELS else DEFAULT_GROQ_MODEL
    selected_model = st.selectbox(
        "Groq model",
        SAFE_GROQ_MODELS,
        index=SAFE_GROQ_MODELS.index(default_model),
    )
    groq_api_key = st.secrets.get("GROQ_API_KEY", None) if hasattr(st, "secrets") else None
    llm_client = GroqLLMClient(api_key=groq_api_key, model=selected_model)
    st.caption(f"LLM mode: {'Groq enabled' if llm_client.is_configured else 'Local fallback'}")

case = InvestigationCase(st.session_state["evidence_items"])
agent = InvestigationAgent(case, llm_client if llm_client.is_configured else None)
search = EvidenceSearch(case.items)

tab_overview, tab_search, tab_graph, tab_timeline, tab_qa, tab_memo = st.tabs(
    ["Command Center", "Evidence Vault", "Entity Graph", "Timeline", "Investigator Q&A", "Memo"]
)

with tab_overview:
    risks = case.risk_signals()
    known, unknown = case.timeline()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Evidence", len(case.items))
    c2.metric("Entities", len(case.entity_index))
    c3.metric("Risk Signals", len(risks))
    c4.metric("Unknown Dates", len(unknown))
    st.subheader("Top Risk Signals")
    for signal in risks[:5]:
        st.warning(f"{signal.label} | {signal.severity} | {signal.score}/100\n\n{signal.reason}")

with tab_search:
    uploaded = st.file_uploader("Add evidence", type=["txt", "eml", "md", "csv", "pdf"], accept_multiple_files=True)
    if uploaded:
        parsed = []
        errors = []
        for file in uploaded:
            try:
                parsed.extend(parse_uploaded_bytes(file.name, file.getvalue()))
            except ValueError as exc:
                errors.append(f"{file.name}: {exc}")
        if parsed:
            st.session_state["evidence_items"] = enrich_entities(deduplicate(case.items + parsed))
            store = EvidenceStore(ROOT / "data" / "local" / "evidence.sqlite")
            inserted = store.upsert_many(st.session_state["evidence_items"])
            st.success(f"Loaded {len(parsed)} uploaded evidence items. SQLite inserted {inserted} new records.")
            st.rerun()
        for error in errors:
            st.error(error)
    query = st.text_input("Search evidence", "Northstar Energy personal email")
    for result in search.search(query, limit=8):
        with st.expander(f"{result.evidence.title} | score {result.score:.2f}"):
            st.write(result.excerpt)
            st.caption(f"Citation: {result.evidence.id} | {result.evidence.source}")

with tab_graph:
    entity = st.text_input("Entity profile", "Maya Rao")
    profile = case.entity_profile(entity)
    st.write(profile)
    st.subheader("Relationship path")
    target = st.text_input("Connect to", "Northstar Energy")
    path = case.relationship_path(entity, target)
    st.write(" -> ".join(path) if path else "No relationship path found.")

with tab_timeline:
    query = st.text_input("Timeline filter", "")
    known, unknown = case.timeline(query or None)
    for item in known:
        st.write(f"**{item.timestamp.date()}** - {item.title} `{item.id}`")
    if unknown:
        st.subheader("Unknown-date evidence")
        for item in unknown:
            st.write(f"{item.title} `{item.id}`")

with tab_qa:
    question = st.text_area("Ask an investigation question", "What connects Maya Rao to Northstar Energy?")
    if st.button("Ask EvidenceIQ"):
        answer = agent.answer(question)
        st.markdown(answer.answer)
        st.caption(f"Confidence: {answer.confidence}")
        for citation in answer.citations:
            st.info(f"{citation.evidence_id}: {citation.excerpt}")

with tab_memo:
    memo = agent.memo("Aster Bridge")
    if st.button("Generate enhanced memo"):
        memo = agent.generate_llm_memo("Aster Bridge")
    st.download_button("Download memo", memo, file_name="evidenceiq_memo.md")
    st.markdown(memo)
