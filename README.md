# EvidenceIQ: Agentic AI Investigation Command Center

EvidenceIQ turns raw evidence into searchable intelligence, relationship graphs,
risk signals, timelines, and citation-backed investigation memos.

This is built as a zero-budget portfolio product. The core engine runs on the
Python standard library. The Streamlit interface is optional but recommended for
the product demo.

## What It Does

- Ingests email-like text, reports, notes, and CSV files.
- Extracts people, organizations, emails, money amounts, dates, and risk terms.
- Deduplicates evidence with content hashes.
- Searches evidence with keyword and lightweight semantic scoring.
- Builds entity profiles and relationship paths.
- Generates timelines with unknown-date handling.
- Produces citation-backed answers and investigation memos.
- Refuses unsupported claims instead of guessing.

## Quick Start

```powershell
cd C:\Users\aswin\documents\evidenceiq
python -m evidenceiq.demo
```

Optional UI:

```powershell
pip install -r requirements.txt
streamlit run app\streamlit_app.py
```

Recommended local UI with an isolated environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\scripts\start_evidenceiq.ps1
```

Then open http://localhost:8501.

## Project Structure

```text
app/                    Streamlit product UI
data/sample_case/        Safe synthetic demo evidence
docs/                    Architecture, evaluation, portfolio notes
evidenceiq/              Core product engine
tests/                   Standard-library unit tests
```

## Safety Boundary

EvidenceIQ is an investigation triage assistant. It does not declare guilt,
fraud, criminality, or legal conclusions. It only surfaces risk signals and
links them to source evidence.

## Portfolio Positioning

**EvidenceIQ: Agentic AI Investigation Command Center**

Built a zero-cost AI investigation platform that transforms raw evidence into
searchable intelligence, relationship graphs, timelines, risk signals, and
citation-backed investigation memos using local NLP, retrieval, graph analytics,
and tool-using agent workflows.
