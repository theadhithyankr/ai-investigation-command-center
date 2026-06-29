from __future__ import annotations

import os
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import networkx as nx
    import plotly.graph_objects as go
    import streamlit as st
except ModuleNotFoundError as exc:
    raise SystemExit("Install UI dependencies with: pip install -r requirements.txt") from exc

from evidenceiq.agents import InvestigationAgent
from evidenceiq.case import InvestigationCase
from evidenceiq.entities import enrich_entities
from evidenceiq.llm import DEFAULT_GROQ_MODEL, SAFE_GROQ_MODELS, GroqLLMClient
from evidenceiq.models import CaseRecord, EvidenceItem, MemoResult
from evidenceiq.parsing import (
    create_manual_evidence,
    deduplicate,
    parse_date,
    parse_manual_note_payload,
    parse_manual_note_payloads,
    parse_uploaded_bytes,
)
from evidenceiq.search import EvidenceSearch
from evidenceiq.spatial import extract_map_pins
from evidenceiq.storage import SAMPLE_CASE_ID, EvidenceStore
from evidenceiq.wall import build_investigation_wall


STORE_PATH = ROOT / "data" / "local" / "evidence.sqlite"
SAMPLE_PATH = ROOT / "data" / "sample_case"
EVIDENCE_TYPES = (
    "witness_note",
    "scene_note",
    "phone_log",
    "forensic_note",
    "interview",
    "report",
    "contradiction_note",
    "unknown_date_note",
    "timeline_note",
    "security_log",
    "financial_note",
    "email",
    "document",
    "other",
)


@st.cache_resource
def get_store() -> EvidenceStore:
    store = EvidenceStore(STORE_PATH)
    store.seed_sample_case(SAMPLE_PATH)
    return store


def streamlit_secret(name: str) -> str | None:
    try:
        return st.secrets.get(name, None)
    except Exception:
        return None


def load_selected_case(store: EvidenceStore) -> CaseRecord:
    cases = store.list_cases()
    if not cases:
        return store.seed_sample_case(SAMPLE_PATH)
    selected = st.session_state.get("selected_case_id")
    if not selected or not store.get_case(selected):
        selected = cases[0].id
        st.session_state["selected_case_id"] = selected
    return store.get_case(selected) or cases[0]


def build_agent(case: InvestigationCase, llm_client: GroqLLMClient | None) -> InvestigationAgent:
    if not llm_client or not llm_client.is_configured:
        return InvestigationAgent(case)
    return InvestigationAgent(case, llm_client)


def case_summary(items: list[EvidenceItem]) -> str:
    if not items:
        return "No evidence has been added to this case yet."
    known = [item for item in items if item.timestamp]
    types = sorted({item.source_type for item in items})
    return f"{len(items)} evidence items across {len(types)} evidence types. {len(known)} items have known dates."


def status_badge(label: str, tone: str = "neutral") -> str:
    return f'<span class="status-pill {tone}">{escape(label)}</span>'


def fallback_label(reason: str | None) -> str:
    labels = {
        "groq_not_configured_or_disabled": "Groq unavailable or disabled for this case",
        "groq_unavailable_or_error": "Groq unavailable or returned an error",
        "groq_invalid_citations": "Groq response failed citation validation",
        "legal_conclusion_refusal": "Legal conclusion refused before analysis",
        "unsupported_question": "No selected-case evidence supported the question",
    }
    return labels.get(reason or "", reason or "None")


def format_indian_date(value) -> str:
    return value.strftime("%d/%m/%Y")


def apply_manual_note_import(payload: dict[str, object]) -> str:
    evidence_type = str(payload.get("evidence_type") or "other")
    if evidence_type not in EVIDENCE_TYPES:
        evidence_type = "other"
    date_value = str(payload.get("date") or "")
    parsed_date = parse_date(date_value)
    payload_marked_unknown = bool(payload.get("unknown_date"))
    unknown_date = payload_marked_unknown or parsed_date is None
    st.session_state["manual_note_title"] = str(payload.get("title") or "")
    st.session_state["manual_evidence_type"] = evidence_type
    st.session_state["manual_unknown_date"] = unknown_date
    st.session_state["manual_note_date"] = format_indian_date(parsed_date) if parsed_date else ""
    st.session_state["manual_source_person"] = str(payload.get("source_person") or "")
    st.session_state["manual_tags"] = ", ".join(str(tag) for tag in payload.get("tags", []) if str(tag).strip())
    st.session_state["manual_location"] = str(payload.get("location") or "")
    st.session_state["manual_latitude"] = str(payload.get("latitude") or "")
    st.session_state["manual_longitude"] = str(payload.get("longitude") or "")
    st.session_state["manual_body"] = str(payload.get("body") or "")
    if date_value and not parsed_date and not payload_marked_unknown:
        return f"Imported note, but could not parse date: {date_value}"
    return "Imported note into the manual evidence fields."


def evidence_from_payload(payload: dict[str, object]) -> EvidenceItem:
    tags = [str(tag).strip() for tag in payload.get("tags", []) if str(tag).strip()]
    date_value = None if payload.get("unknown_date") else str(payload.get("date") or "")
    return create_manual_evidence(
        str(payload.get("title") or ""),
        str(payload.get("evidence_type") or "other"),
        str(payload.get("body") or ""),
        date_value=date_value,
        source_person=str(payload.get("source_person") or ""),
        tags=tags,
        location=str(payload.get("location") or ""),
        latitude=str(payload.get("latitude") or ""),
        longitude=str(payload.get("longitude") or ""),
    )


def render_evidence_map(items: list[EvidenceItem]) -> None:
    pins = extract_map_pins(items)
    if not pins:
        st.info("No explicit coordinates found. Add lines like `Location: Warehouse A` and `Coordinates: 12.9716, 77.5946` to map evidence.")
        return
    hover_text = []
    for pin in pins:
        date_label = pin.timestamp.date().isoformat() if pin.timestamp else "Unknown date"
        people = ", ".join(pin.people) if pin.people else "No people extracted"
        hover_text.append(
            "<br>".join(
                [
                    f"<b>{escape(pin.title)}</b>",
                    f"Date: {escape(date_label)}",
                    f"Evidence: {escape(pin.evidence_id)}",
                    f"Source: {escape(pin.source)}",
                    f"Location: {escape(pin.location_label)}",
                    f"People: {escape(people)}",
                ]
            )
        )
    fig = go.Figure(
        go.Scattergeo(
            lat=[pin.latitude for pin in pins],
            lon=[pin.longitude for pin in pins],
            text=hover_text,
            hoverinfo="text",
            mode="markers",
            marker={"size": 12, "color": "#d6b25e", "line": {"width": 1, "color": "#0d0f12"}},
        )
    )
    fig.update_geos(
        projection_type="natural earth",
        showland=True,
        landcolor="#202832",
        showocean=True,
        oceancolor="#101820",
        showcountries=True,
        countrycolor="#4a5663",
        fitbounds="locations",
    )
    fig.update_layout(
        height=560,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="#0d0f12",
        plot_bgcolor="#0d0f12",
        font={"color": "#e8edf2"},
    )
    st.plotly_chart(fig, use_container_width=True)
    rows = [
        {
            "evidence_id": pin.evidence_id,
            "title": pin.title,
            "date": pin.timestamp.date().isoformat() if pin.timestamp else "Unknown date",
            "location": pin.location_label,
            "latitude": pin.latitude,
            "longitude": pin.longitude,
            "people": ", ".join(pin.people),
        }
        for pin in pins
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_investigation_wall(case: InvestigationCase) -> None:
    nodes, edges = build_investigation_wall(case)
    if not nodes:
        st.info("No risk or entity connections are available yet. Add evidence with risk terms or extracted entities.")
        return
    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node.id)
    for edge in edges:
        graph.add_edge(edge.source, edge.target)
    positions = nx.spring_layout(graph, seed=42, k=0.9)
    node_by_id = {node.id: node for node in nodes}
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for edge in edges:
        x0, y0 = positions[edge.source]
        x1, y1 = positions[edge.target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line={"width": 1, "color": "#4a5663"},
        hoverinfo="none",
        mode="lines",
    )
    color_map = {
        "risk": "#d96b5f",
        "evidence": "#d6b25e",
        "person": "#74b28d",
        "organization": "#8fb3d9",
        "email": "#b59bd8",
        "location": "#df9f73",
    }
    size_map = {"risk": 22, "evidence": 18, "person": 14, "organization": 14, "email": 12, "location": 12}
    node_trace = go.Scatter(
        x=[positions[node.id][0] for node in nodes],
        y=[positions[node.id][1] for node in nodes],
        mode="markers+text",
        text=[node.label[:28] for node in nodes],
        textposition="top center",
        hovertext=[_wall_hover(node_by_id[node.id]) for node in nodes],
        hoverinfo="text",
        marker={
            "size": [size_map.get(node.node_type, 12) for node in nodes],
            "color": [color_map.get(node.node_type, "#91a0ad") for node in nodes],
            "line": {"width": 1, "color": "#0d0f12"},
        },
    )
    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(
        height=620,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        paper_bgcolor="#0d0f12",
        plot_bgcolor="#0d0f12",
        font={"color": "#e8edf2", "size": 11},
        xaxis={"showgrid": False, "zeroline": False, "visible": False},
        yaxis={"showgrid": False, "zeroline": False, "visible": False},
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{len([node for node in nodes if node.node_type == 'risk'])} risk node(s), {len(nodes)} total node(s), {len(edges)} connection(s).")


def _wall_hover(node) -> str:
    lines = [f"<b>{escape(node.label)}</b>", f"Type: {escape(node.node_type)}"]
    if node.severity:
        lines.append(f"Severity: {escape(node.severity)}")
    for key, value in node.metadata.items():
        lines.append(f"{escape(key.replace('_', ' ').title())}: {escape(value)}")
    return "<br>".join(lines)


st.set_page_config(page_title="EvidenceIQ", page_icon="EI", layout="wide")
st.markdown(
    """
    <style>
    :root {
        --ei-bg: #0d0f12;
        --ei-panel: #151a20;
        --ei-panel-2: #1b222b;
        --ei-line: #2d3742;
        --ei-text: #e8edf2;
        --ei-muted: #91a0ad;
        --ei-accent: #d6b25e;
        --ei-danger: #d96b5f;
        --ei-ok: #74b28d;
    }
    .stApp { background: var(--ei-bg); color: var(--ei-text); }
    .block-container { padding-top: 1.5rem; max-width: 1440px; }
    h1, h2, h3 { letter-spacing: 0; }
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, var(--ei-panel), #11161c);
        border: 1px solid var(--ei-line);
        border-radius: 6px;
        padding: 12px 14px;
    }
    div[data-testid="stMetricLabel"] { color: var(--ei-muted); }
    .ei-header {
        border: 1px solid var(--ei-line);
        background: linear-gradient(135deg, #151a20, #101418);
        padding: 18px 20px;
        border-radius: 6px;
        margin-bottom: 14px;
    }
    .ei-title { font-size: 28px; font-weight: 720; margin: 0; }
    .ei-subtitle { color: var(--ei-muted); margin-top: 4px; font-size: 14px; }
    .panel {
        border: 1px solid var(--ei-line);
        background: var(--ei-panel);
        border-radius: 6px;
        padding: 14px;
        min-height: 100%;
    }
    .panel h3 { margin-top: 0; font-size: 16px; }
    .status-pill {
        display: inline-block;
        border: 1px solid var(--ei-line);
        border-radius: 999px;
        color: var(--ei-text);
        background: var(--ei-panel-2);
        padding: 3px 9px;
        margin: 2px 4px 2px 0;
        font-size: 12px;
        line-height: 1.4;
    }
    .status-pill.ok { border-color: #496f5a; color: #bce0ca; }
    .status-pill.warn { border-color: #7d6840; color: #ebd08b; }
    .status-pill.danger { border-color: #7a4945; color: #ecaaa2; }
    .source {
        border-left: 3px solid var(--ei-accent);
        padding-left: 10px;
        color: var(--ei-muted);
        font-size: 13px;
    }
    .lead-row {
        border: 1px solid var(--ei-line);
        background: #121820;
        border-radius: 6px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }
    .lead-name { font-size: 18px; font-weight: 700; margin-bottom: 3px; }
    .lead-score-track {
        height: 8px;
        background: #232c36;
        border-radius: 999px;
        overflow: hidden;
        margin: 8px 0 4px 0;
    }
    .lead-score-fill {
        height: 8px;
        background: linear-gradient(90deg, var(--ei-accent), var(--ei-danger));
    }
    .small-muted { color: var(--ei-muted); font-size: 13px; }
    </style>
    """,
    unsafe_allow_html=True,
)

store = get_store()
selected_case = load_selected_case(store)

with st.sidebar:
    st.header("Case Desk")
    cases = store.list_cases()
    case_names = {case.id: f"{case.name} ({case.case_type})" for case in cases}
    selected_index = max(0, [case.id for case in cases].index(selected_case.id)) if selected_case.id in case_names else 0
    chosen_case_id = st.selectbox(
        "Active case",
        [case.id for case in cases],
        format_func=lambda case_id: case_names.get(case_id, case_id),
        index=selected_index,
    )
    if chosen_case_id != selected_case.id:
        st.session_state["selected_case_id"] = chosen_case_id
        st.rerun()

    with st.expander("New case", expanded=False):
        with st.form("new_case_form"):
            new_name = st.text_input("Case name")
            new_type = st.text_input("Case type", "custom")
            new_description = st.text_area("Description", height=80)
            create_clicked = st.form_submit_button("Create case")
        if create_clicked:
            created = store.create_case(new_name, new_type, new_description, llm_enabled=False)
            st.session_state["selected_case_id"] = created.id
            st.rerun()

    with st.expander("Case settings", expanded=False):
        with st.form("case_settings_form"):
            edit_name = st.text_input("Name", selected_case.name)
            edit_type = st.text_input("Type", selected_case.case_type)
            edit_description = st.text_area("Description", selected_case.description, height=90)
            llm_enabled = st.checkbox("Enable Groq enhanced analysis for this case", selected_case.llm_enabled)
            saved = st.form_submit_button("Save settings")
        if saved:
            store.update_case(
                selected_case.id,
                name=edit_name,
                case_type=edit_type,
                description=edit_description,
                llm_enabled=llm_enabled,
            )
            st.rerun()
        if selected_case.id != SAMPLE_CASE_ID and st.button("Archive case"):
            store.delete_case(selected_case.id)
            st.session_state["selected_case_id"] = SAMPLE_CASE_ID
            st.rerun()

    configured_model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    default_model = configured_model if configured_model in SAFE_GROQ_MODELS else DEFAULT_GROQ_MODEL
    selected_model = st.selectbox("Groq model", SAFE_GROQ_MODELS, index=SAFE_GROQ_MODELS.index(default_model))
    groq_api_key = streamlit_secret("GROQ_API_KEY")
    candidate_llm = GroqLLMClient(api_key=groq_api_key, model=selected_model)
    if candidate_llm.is_configured and selected_case.llm_enabled:
        llm_status = "Groq enabled for this case"
        llm_status_tone = "ok"
        active_llm = candidate_llm
    elif candidate_llm.is_configured:
        llm_status = "Groq available, disabled for this case"
        llm_status_tone = "warn"
        active_llm = None
    else:
        llm_status = "Local fallback, no Groq key"
        llm_status_tone = "neutral"
        active_llm = None
    st.markdown(status_badge(llm_status, llm_status_tone), unsafe_allow_html=True)

    if active_llm and st.button("Re-extract entities with Groq"):
        existing_items = store.all(selected_case.id)
        updated = store.update_many(selected_case.id, enrich_entities(existing_items, active_llm))
        st.success(f"Re-extracted entities for {updated} evidence item(s).")
        st.rerun()

items = store.all(selected_case.id)
case = InvestigationCase(items)
agent = build_agent(case, active_llm)
search = EvidenceSearch(case.items)
risks = case.risk_signals()
leads = case.leadboard()
victims = case.victims()
known, unknown = case.timeline()

st.markdown(
    f"""
    <div class="ei-header">
      <p class="ei-title">{escape(selected_case.name)}</p>
      <div class="ei-subtitle">{escape(selected_case.case_type)} command center | {escape(selected_case.description or 'No description recorded.')}</div>
      <div style="margin-top:10px;">
        {status_badge(f'{len(items)} evidence items')}
        {status_badge(f'{len(case.entity_index)} entities')}
        {status_badge(f'{len(victims)} victim records', 'neutral')}
        {status_badge(f'{len(risks)} risk signals', 'danger' if risks else 'ok')}
        {status_badge(f'{len(leads)} investigative leads', 'warn' if leads else 'neutral')}
        {status_badge(llm_status, llm_status_tone)}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_overview, tab_vault, tab_leadboard, tab_map, tab_wall, tab_timeline, tab_copilot, tab_memo = st.tabs(
    ["Overview", "Evidence Vault", "Leadboard", "Map", "Investigation Wall", "Timeline", "Copilot", "Memo"]
)

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Evidence", len(items))
    c2.metric("Entities", len(case.entity_index))
    c3.metric("Risk Signals", len(risks))
    c4.metric("Unknown Dates", len(unknown))

    left, middle, right = st.columns([1.1, 1, 1])
    with left:
        st.markdown('<div class="panel"><h3>Case Summary</h3>', unsafe_allow_html=True)
        st.write(case_summary(items))
        st.markdown("</div>", unsafe_allow_html=True)
    with middle:
        st.markdown('<div class="panel"><h3>Latest Evidence</h3>', unsafe_allow_html=True)
        latest = sorted(items, key=lambda item: item.timestamp or selected_case.updated_at, reverse=True)[:5]
        for item in latest:
            date_label = item.timestamp.date().isoformat() if item.timestamp else "Unknown date"
            st.write(f"**{item.title}**")
            st.caption(f"{date_label} | {item.source_type} | {item.id}")
        if not latest:
            st.caption("Add evidence in the Evidence Vault.")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><h3>Entity Highlights</h3>', unsafe_allow_html=True)
        for entity, docs in sorted(case.entity_index.items(), key=lambda pair: len(pair[1]), reverse=True)[:8]:
            st.write(f"{entity}")
            st.caption(f"{len({doc.id for doc in docs})} linked item(s)")
        if not case.entity_index:
            st.caption("Entities appear after evidence is added.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Priority Lead Preview")
    st.caption("Evidence-backed triage only. Scores do not estimate guilt, culpability, or legal liability.")
    if not leads:
        st.info("No person entities found yet. Add witness notes, interviews, reports, or documents with named people.")
    for lead in leads[:3]:
        tone = "danger" if lead.priority == "high" else "warn" if lead.priority == "medium" else "neutral"
        st.markdown(
            f"""
            <div class="lead-row">
              <div class="lead-name">{escape(lead.name)}</div>
              {status_badge(f'{lead.priority.upper()} priority', tone)}
              {status_badge(f'{lead.score}/100 triage score')}
              <div class="lead-score-track"><div class="lead-score-fill" style="width:{lead.score}%"></div></div>
              <div class="small-muted">{escape(lead.reasons[0])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Victims")
    if not victims:
        st.caption("No explicit victim/deceased entities detected.")
    for victim in victims:
        with st.expander(f"{victim.name} | {victim.evidence_count} linked item(s)", expanded=False):
            for reason in victim.reasons:
                st.write(f"- {reason}")
            for citation in victim.citations:
                st.markdown(
                    f'<div class="source">{escape(citation.evidence_id)} | {escape(citation.title)}: {escape(citation.excerpt)}</div>',
                    unsafe_allow_html=True,
                )

    st.subheader("Risk Cards")
    if not risks:
        st.info("No configured risk signals found.")
    for signal in risks[:6]:
        with st.expander(f"{signal.label} | {signal.severity} | {signal.score}/100", expanded=False):
            st.write(signal.reason)
            for citation in signal.citations:
                st.markdown(
                    f'<div class="source">{escape(citation.evidence_id)}: {escape(citation.excerpt)}</div>',
                    unsafe_allow_html=True,
                )

with tab_vault:
    upload_col, manual_col = st.columns([1, 1])
    with upload_col:
        st.subheader("Upload Evidence")
        uploaded = st.file_uploader(
            "Add .txt, .eml, .md, .csv, or .pdf",
            type=["txt", "eml", "md", "csv", "pdf"],
            accept_multiple_files=True,
        )
        if uploaded and st.button("Ingest uploads"):
            parsed = []
            errors = []
            for file in uploaded:
                try:
                    parsed.extend(parse_uploaded_bytes(file.name, file.getvalue()))
                except ValueError as exc:
                    errors.append(f"{file.name}: {exc}")
            if parsed:
                current_hashes = {item.content_hash for item in items}
                new_items = [item for item in deduplicate(parsed) if item.content_hash not in current_hashes]
                inserted = store.upsert_many(selected_case.id, enrich_entities(new_items, active_llm))
                st.success(f"Parsed {len(parsed)} item(s). Inserted {inserted} new selected-case record(s).")
                st.rerun()
            for error in errors:
                st.error(error)
    with manual_col:
        st.subheader("Manual Note")
        with st.expander("Bulk add notes", expanded=False):
            st.caption("Paste a JSON array or multiple labeled notes separated by --- EVIDENCE ITEM ---.")
            bulk_payload = st.text_area("Bulk note payload", height=220, key="bulk_note_import_payload")
            if st.button("Import all notes"):
                try:
                    payloads = parse_manual_note_payloads(bulk_payload)
                    missing_body = [index for index, payload in enumerate(payloads, start=1) if not str(payload.get("body") or "").strip()]
                    bad_dates = [
                        index
                        for index, payload in enumerate(payloads, start=1)
                        if not payload.get("unknown_date") and str(payload.get("date") or "").strip() and not parse_date(str(payload.get("date")))
                    ]
                    if missing_body:
                        st.error(f"Evidence item(s) missing body: {', '.join(map(str, missing_body))}")
                    elif bad_dates:
                        st.error(f"Evidence item(s) have invalid dates. Use DD/MM/YYYY: {', '.join(map(str, bad_dates))}")
                    else:
                        new_items = [evidence_from_payload(payload) for payload in payloads]
                        current_hashes = {item.content_hash for item in items}
                        unique_items = [item for item in deduplicate(new_items) if item.content_hash not in current_hashes]
                        inserted = store.upsert_many(selected_case.id, enrich_entities(unique_items, active_llm))
                        st.success(f"Imported {inserted} new evidence item(s) from {len(payloads)} parsed note(s).")
                        st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        with st.expander("Auto-fill from JSON or labeled note", expanded=False):
            st.caption("Paste either a JSON object or a labeled note block with Title, Evidence type, Date, Source/person, Tags, and Body.")
            imported_note = st.text_area("Paste note payload", height=180, key="manual_note_import_payload")
            if st.button("Auto-fill note fields"):
                try:
                    import_result = apply_manual_note_import(parse_manual_note_payload(imported_note))
                    st.session_state["manual_note_import_status"] = import_result
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
            if st.session_state.get("manual_note_import_status"):
                st.success(st.session_state["manual_note_import_status"])
        with st.form("manual_evidence_form"):
            note_title = st.text_input("Title", key="manual_note_title")
            evidence_type = st.selectbox("Evidence type", EVIDENCE_TYPES, key="manual_evidence_type")
            unknown_date = st.checkbox("Unknown date", True, key="manual_unknown_date")
            note_date = st.text_input("Evidence date", placeholder="DD/MM/YYYY", key="manual_note_date")
            st.caption("Use Indian date format DD/MM/YYYY. If Unknown date is checked, this date will not be saved.")
            source_person = st.text_input("Source/person", key="manual_source_person")
            tags = st.text_input("Tags", placeholder="comma separated", key="manual_tags")
            location = st.text_input("Location", placeholder="Warehouse A", key="manual_location")
            coord_cols = st.columns(2)
            latitude = coord_cols[0].text_input("Latitude", placeholder="12.9716", key="manual_latitude")
            longitude = coord_cols[1].text_input("Longitude", placeholder="77.5946", key="manual_longitude")
            body = st.text_area("Body", height=160, key="manual_body")
            save_note = st.form_submit_button("Save note")
        if save_note:
            if not body.strip():
                st.error("Manual evidence needs a body.")
            elif not unknown_date and not parse_date(note_date):
                st.error("Enter the evidence date as DD/MM/YYYY, for example 17/10/2023.")
            elif (latitude.strip() and not longitude.strip()) or (longitude.strip() and not latitude.strip()):
                st.error("Enter both latitude and longitude, or leave both blank.")
            else:
                tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
                item = create_manual_evidence(
                    note_title,
                    evidence_type,
                    body,
                    date_value=None if unknown_date else note_date,
                    source_person=source_person,
                    tags=tag_list,
                    location=location,
                    latitude=latitude,
                    longitude=longitude,
                )
                inserted = store.upsert_many(selected_case.id, enrich_entities([item], active_llm))
                if inserted:
                    st.success("Manual evidence saved.")
                else:
                    st.info("That evidence already exists in this case.")
                st.rerun()

    st.subheader("Search Vault")
    query = st.text_input("Search selected-case evidence", "")
    results = search.search(query, limit=12) if query else []
    if query and not results:
        st.info("No selected-case evidence matched that search.")
    for result in results:
        with st.expander(f"{result.evidence.title} | score {result.score:.2f}"):
            st.write(result.excerpt)
            st.caption(f"Citation: {result.evidence.id} | {result.evidence.source}")

    table_rows = [
        {
            "id": item.id,
            "title": item.title,
            "type": item.source_type,
            "date": item.timestamp.date().isoformat() if item.timestamp else "Unknown",
            "source": item.source,
        }
        for item in items
    ]
    st.dataframe(table_rows, use_container_width=True, hide_index=True)

with tab_leadboard:
    st.subheader("Person of Interest Priority Board")
    st.caption(
        "Ranks people by cited investigative signals inside the selected case. This is not a culprit prediction or legal conclusion."
    )
    if not leads:
        st.info("No person entities are available for ranking yet.")
    for rank, lead in enumerate(leads, start=1):
        tone = "danger" if lead.priority == "high" else "warn" if lead.priority == "medium" else "neutral"
        st.markdown(
            f"""
            <div class="lead-row">
              <div class="lead-name">#{rank} {escape(lead.name)}</div>
              {status_badge(f'{lead.priority.upper()} priority', tone)}
              {status_badge(f'{lead.score}/100 triage score')}
              <div class="lead-score-track"><div class="lead-score-fill" style="width:{lead.score}%"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        metric_cols = st.columns(6)
        metric_cols[0].metric("Evidence", lead.evidence_count)
        metric_cols[1].metric("Risk Proximity", lead.risk_proximity)
        metric_cols[2].metric("Timeline", lead.timeline_proximity)
        metric_cols[3].metric("Scene Adjacent", lead.scene_proximity)
        metric_cols[4].metric("Relations", lead.relationship_density)
        metric_cols[5].metric("Unresolved", lead.unresolved_items)
        with st.expander(f"Why {lead.name} is ranked here"):
            for reason in lead.reasons:
                st.write(f"- {reason}")
            st.markdown("**Citations**")
            for citation in lead.citations:
                st.markdown(
                    f'<div class="source">{escape(citation.evidence_id)} | {escape(citation.title)}: {escape(citation.excerpt)}</div>',
                    unsafe_allow_html=True,
                )

with tab_map:
    st.subheader("Explicit Coordinate Map")
    st.caption("Evidence appears here only when it contains explicit latitude and longitude. EvidenceIQ does not geocode or guess locations.")
    render_evidence_map(items)

with tab_wall:
    st.subheader("Risk-First Investigation Wall")
    wall_col, tools_col = st.columns([1.45, 0.85])
    with wall_col:
        render_investigation_wall(case)
    with tools_col:
        st.markdown("**Entity Profile**")
        entity_names = sorted(case.entity_index)
        entity = st.text_input("Entity profile", entity_names[0] if entity_names else "")
        if entity:
            profile = case.entity_profile(entity)
            st.write(profile)
        st.markdown("**Relationship Path**")
        target = st.text_input("Relationship path target", entity_names[1] if len(entity_names) > 1 else "")
        if entity and target:
            path = case.relationship_path(entity, target)
            st.write(" -> ".join(path) if path else "No relationship path found.")

with tab_timeline:
    timeline_filter = st.text_input("Timeline filter", "")
    known_filtered, unknown_filtered = case.timeline(timeline_filter or None)
    st.subheader("Known-Date Evidence")
    for item in known_filtered:
        st.write(f"**{item.timestamp.date()}** - {item.title} `{item.id}`")
        st.caption(item.source)
    if not known_filtered:
        st.caption("No known-date evidence matched.")
    st.subheader("Unknown-Date Evidence")
    for item in unknown_filtered:
        st.write(f"{item.title} `{item.id}`")
        st.caption(item.source)
    if not unknown_filtered:
        st.caption("No unknown-date evidence matched.")

with tab_copilot:
    suggestions = [
        "What are the strongest risk signals in this case?",
        "Which people or organizations appear most often?",
        "What evidence has unknown dates?",
        "What gaps should an investigator review next?",
    ]
    st.caption("Suggested investigation questions")
    st.write(" | ".join(suggestions))
    question = st.text_area("Ask about selected-case evidence", suggestions[0], height=90)
    if st.button("Ask EvidenceIQ"):
        answer = agent.answer(question)
        st.markdown(answer.answer)
        st.caption(f"Confidence: {answer.confidence} | Mode: {answer.mode} | Fallback: {fallback_label(answer.fallback_reason)}")
        for citation in answer.citations:
            with st.expander(f"{citation.evidence_id}: {citation.title}"):
                st.write(citation.excerpt)
                st.caption(citation.source)
        if answer.fallback_reason == "unsupported_question":
            st.info("Gap: no retrieved selected-case evidence supported that question.")

with tab_memo:
    memo_mode = st.radio("Memo mode", ["Local", "Enhanced if enabled"], horizontal=True)
    if memo_mode == "Enhanced if enabled":
        memo_result = agent.generate_memo_result(selected_case.name)
    else:
        memo_result = MemoResult(agent.memo(selected_case.name), "local", None)
    st.caption(f"Mode: {memo_result.mode} | Fallback: {fallback_label(memo_result.fallback_reason)}")
    st.download_button("Download memo", memo_result.memo, file_name=f"{selected_case.name.lower().replace(' ', '-')}-memo.md")
    st.markdown(memo_result.memo)
