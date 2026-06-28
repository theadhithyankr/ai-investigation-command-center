from __future__ import annotations

from pathlib import Path

from evidenceiq.agents import InvestigationAgent
from evidenceiq.pipeline import build_case_from_folder


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    case = build_case_from_folder(root / "data" / "sample_case")
    agent = InvestigationAgent(case)
    print(agent.memo("Aster Bridge"))
    print()
    answer = agent.answer("What connects Maya Rao to Northstar Energy?")
    print("## Investigator Q&A")
    print(answer.answer)
    for citation in answer.citations:
        print(f"- {citation.evidence_id}: {citation.title}")


if __name__ == "__main__":
    main()
