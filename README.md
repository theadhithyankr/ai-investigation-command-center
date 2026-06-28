# EvidenceIQ: AI Investigation Command Center

<p align="center">
  <img src="https://raw.githubusercontent.com/theadhithyankr/ai-investigation-command-center/main/assets/evidenceiq-logo.svg" alt="EvidenceIQ logo" width="720">
</p>

EvidenceIQ is a local-first AI investigation workspace that turns messy evidence
into searchable intelligence, relationship graphs, timelines, risk signals, and
citation-backed investigation memos.

It is built to be understood quickly by recruiters, hiring managers, and
technical reviewers: upload evidence, ask investigation questions, inspect the
supporting sources, and export a professional memo.

## Product Snapshot

- **Evidence Vault**: ingest email-like files, reports, notes, CSVs, and PDFs.
- **Investigator Q&A**: answer questions only from retrieved evidence.
- **Entity Graph**: connect people, companies, emails, transactions, and files.
- **Timeline Builder**: separate dated evidence from unknown-date evidence.
- **Risk Console**: surface explainable investigation leads.
- **Memo Generator**: export a citation-backed investigation brief.

<p align="center">
  <img src="https://raw.githubusercontent.com/theadhithyankr/ai-investigation-command-center/main/assets/evidenceiq-architecture.svg" alt="EvidenceIQ architecture diagram" width="860">
</p>

Direct asset links: [logo SVG](assets/evidenceiq-logo.svg) · [architecture SVG](assets/evidenceiq-architecture.svg)

## Why This Project Stands Out

Most AI demos stop at chat. EvidenceIQ behaves like an investigation product:

- Every answer needs source citations.
- Unsupported questions are refused.
- Legal conclusions are blocked.
- Risk scores show their ingredients.
- Duplicate evidence is removed with content hashes.
- Unknown dates are kept out of exact timelines.
- The core engine runs without paid APIs.

## Demo Workflow

1. Open the Command Center and review evidence/risk counts.
2. Search for `Northstar Energy personal email`.
3. Inspect the relationship path from `Maya Rao` to `Northstar Energy`.
4. Open the timeline and verify the unknown-date evidence is separated.
5. Ask: `What connects Maya Rao to Northstar Energy?`
6. Download the generated investigation memo.

## Quick Start

```powershell
cd C:\Users\aswin\documents\evidenceiq
python -m evidenceiq.demo
```

## Run The Product UI

```powershell
cd C:\Users\aswin\documents\evidenceiq
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\scripts\start_evidenceiq.ps1
```

Then open:

```text
http://localhost:8501
```

## Tech Stack

- Python
- Streamlit
- SQLite
- Local retrieval/search
- Rule-based NLP/entity extraction
- Graph traversal
- Standard-library unit tests

The implementation is intentionally zero-budget. Optional heavier NLP/LLM
components can be added later behind the same tool interfaces.

## Project Structure

```text
app/                     Streamlit product UI
assets/                  README SVGs and visual assets
data/sample_case/         Safe synthetic demo evidence
docs/                    Architecture, evaluation, portfolio notes
evidenceiq/               Core investigation engine
scripts/                  Local app launcher
tests/                    Unit tests
```

## Safety Boundary

EvidenceIQ is an investigation triage assistant. It does not declare guilt,
fraud, criminality, or legal liability. It only surfaces cited risk signals and
investigation leads.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Current coverage includes parsing, entity extraction, deduplication, retrieval,
unsupported-question refusal, legal-conclusion refusal, relationship paths,
timeline handling, risk signals, and SQLite round trips.

## Portfolio Positioning

**EvidenceIQ: AI Investigation Command Center**

Built a zero-cost AI investigation platform that transforms raw evidence into
searchable intelligence, relationship graphs, timelines, risk signals, and
citation-backed investigation memos using local NLP, retrieval, graph analytics,
SQLite, and tool-using agent workflows.
