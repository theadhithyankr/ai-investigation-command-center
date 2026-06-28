import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from evidenceiq.agents import InvestigationAgent
from evidenceiq.entities import extract_entities
from evidenceiq.llm import build_answer_prompt
from evidenceiq.parsing import deduplicate, parse_date, parse_text_file
from evidenceiq.pipeline import build_case_from_folder
from evidenceiq.search import EvidenceSearch
from evidenceiq.storage import EvidenceStore


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample_case"


class SpyLLM:
    model = "test-model"

    def __init__(self, answer_text="Enhanced answer [ev-001]"):
        self.answer_text = answer_text
        self.answer_calls = []
        self.memo_calls = []

    def answer(self, question, evidence):
        self.answer_calls.append((question, evidence))
        evidence_id = evidence[0].evidence.id if evidence else "ev-001"
        return self.answer_text.replace("ev-001", evidence_id)

    def memo(self, case_name, evidence, risks, timeline):
        self.memo_calls.append((case_name, evidence, risks, timeline))
        citation_id = evidence[0].evidence_id if evidence else "ev-001"
        return f"# Enhanced Memo\n\n## Key Findings\n- Finding supported by evidence [{citation_id}]"


class EvidenceIQTests(unittest.TestCase):
    def test_parse_date_handles_known_and_unknown_dates(self):
        self.assertEqual(parse_date("2026-02-03").year, 2026)
        self.assertIsNone(parse_date("not a date"))

    def test_email_parser_handles_headers(self):
        item = parse_text_file(SAMPLE / "01_email.txt")
        self.assertEqual(item.sender, "Maya Rao <maya.rao@asterbridge.com>")
        self.assertIn("Leo Grant", item.body)
        self.assertIsNotNone(item.timestamp)

    def test_entity_extraction_captures_core_types(self):
        text = "Maya Rao sent $250000 to Northstar Energy on 2026-02-06 using maya@example.com. Keep confidential."
        entities = extract_entities(text)
        self.assertIn("Maya Rao", entities["people"])
        self.assertIn("Northstar Energy", entities["organizations"])
        self.assertIn("maya@example.com", entities["emails"])
        self.assertIn("$250000", entities["money"])
        self.assertIn("confidential", entities["risk_terms"])

    def test_deduplicate_removes_duplicate_content(self):
        one = parse_text_file(SAMPLE / "01_email.txt")
        two = parse_text_file(SAMPLE / "01_email.txt")
        self.assertEqual(len(deduplicate([one, two])), 1)

    def test_search_returns_cited_evidence(self):
        case = build_case_from_folder(SAMPLE)
        results = EvidenceSearch(case.items).search("Northstar Energy side letter")
        self.assertTrue(results)
        self.assertIn("Northstar", results[0].excerpt)

    def test_agent_refuses_unsupported_claims(self):
        case = build_case_from_folder(SAMPLE)
        answer = InvestigationAgent(case).answer("Evidence about a spaceship vendor")
        self.assertEqual(answer.confidence, "none")
        self.assertFalse(answer.citations)

    def test_agent_uses_deterministic_fallback_without_llm(self):
        case = build_case_from_folder(SAMPLE)
        answer = InvestigationAgent(case).answer("What connects Maya Rao to Northstar Energy?")
        self.assertIn("Based only on retrieved evidence", answer.answer)
        self.assertTrue(answer.citations)

    def test_agent_refuses_legal_conclusions(self):
        case = build_case_from_folder(SAMPLE)
        answer = InvestigationAgent(case).answer("Did EvidenceIQ find proof of fraud?")
        self.assertEqual(answer.confidence, "none")
        self.assertIn("cannot determine", answer.answer)
        self.assertFalse(answer.citations)

    def test_refusals_happen_before_llm_calls(self):
        case = build_case_from_folder(SAMPLE)
        llm = SpyLLM()
        InvestigationAgent(case, llm).answer("Did EvidenceIQ find proof of fraud?")
        InvestigationAgent(case, llm).answer("Evidence about a spaceship vendor")
        self.assertFalse(llm.answer_calls)

    def test_groq_prompt_contains_only_retrieved_excerpts_and_citation_ids(self):
        case = build_case_from_folder(SAMPLE)
        results = EvidenceSearch(case.items).search("Northstar Energy side letter", limit=2)
        payload = build_answer_prompt("What happened?", results)
        for result in results:
            self.assertIn(f"[{result.evidence.id}]", payload.user)
            self.assertIn(result.excerpt, payload.user)
            self.assertNotIn(result.evidence.body, payload.user)

    def test_llm_answer_missing_citation_ids_is_downgraded_to_fallback(self):
        case = build_case_from_folder(SAMPLE)
        llm = SpyLLM("Enhanced answer without citations")
        answer = InvestigationAgent(case, llm).answer("What connects Maya Rao to Northstar Energy?")
        self.assertIn("Based only on retrieved evidence", answer.answer)
        self.assertTrue(llm.answer_calls)

    def test_llm_answer_with_unknown_citation_id_is_downgraded_to_fallback(self):
        case = build_case_from_folder(SAMPLE)
        llm = SpyLLM("Enhanced answer [ev-not-real]")
        answer = InvestigationAgent(case, llm).answer("What connects Maya Rao to Northstar Energy?")
        self.assertIn("Based only on retrieved evidence", answer.answer)
        self.assertTrue(llm.answer_calls)

    def test_memo_fallback_and_enhanced_generation(self):
        case = build_case_from_folder(SAMPLE)
        fallback = InvestigationAgent(case).generate_llm_memo("Aster Bridge")
        self.assertIn("# Aster Bridge Investigation Memo", fallback)
        llm = SpyLLM()
        enhanced = InvestigationAgent(case, llm).generate_llm_memo("Aster Bridge")
        self.assertIn("# Enhanced Memo", enhanced)
        self.assertTrue(llm.memo_calls)

    def test_timeline_separates_unknown_dates(self):
        case = build_case_from_folder(SAMPLE)
        known, unknown = case.timeline()
        self.assertTrue(known)
        self.assertTrue(any(item.timestamp is None for item in unknown))

    def test_relationship_path_connects_entities(self):
        case = build_case_from_folder(SAMPLE)
        path = case.relationship_path("Maya Rao", "Northstar Energy")
        self.assertTrue(path)

    def test_risk_signals_are_explainable(self):
        case = build_case_from_folder(SAMPLE)
        signals = case.risk_signals()
        self.assertTrue(signals)
        self.assertTrue(signals[0].reason)
        self.assertTrue(signals[0].citations)

    def test_sqlite_store_round_trips_evidence(self):
        case = build_case_from_folder(SAMPLE)
        with TemporaryDirectory() as temp_dir:
            store = EvidenceStore(Path(temp_dir) / "test_evidence.sqlite")
            inserted = store.upsert_many(case.items)
            loaded = store.all()
        self.assertEqual(inserted, len(case.items))
        self.assertEqual(len(loaded), len(case.items))
        self.assertTrue(loaded[0].entities)


if __name__ == "__main__":
    unittest.main()
