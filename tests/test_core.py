import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from evidenceiq.agents import InvestigationAgent
from evidenceiq.case import InvestigationCase
from evidenceiq.entities import enrich_entities, extract_entities
from evidenceiq.llm import build_answer_prompt
from evidenceiq.parsing import create_manual_evidence, deduplicate, parse_date, parse_text_file
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
            record = store.create_case("Round Trip")
            inserted = store.upsert_many(record.id, case.items)
            loaded = store.all(record.id)
        self.assertEqual(inserted, len(case.items))
        self.assertEqual(len(loaded), len(case.items))
        self.assertTrue(loaded[0].entities)

    def test_store_isolates_evidence_by_case_id(self):
        item = parse_text_file(SAMPLE / "01_email.txt")
        with TemporaryDirectory() as temp_dir:
            store = EvidenceStore(Path(temp_dir) / "test_evidence.sqlite")
            one = store.create_case("Case One")
            two = store.create_case("Case Two")
            self.assertEqual(store.upsert_many(one.id, [item]), 1)
            self.assertEqual(store.upsert_many(two.id, [item]), 1)
            self.assertEqual(len(store.all(one.id)), 1)
            self.assertEqual(len(store.all(two.id)), 1)

    def test_seed_sample_case_is_idempotent(self):
        with TemporaryDirectory() as temp_dir:
            store = EvidenceStore(Path(temp_dir) / "test_evidence.sqlite")
            first = store.seed_sample_case(SAMPLE)
            first_count = len(store.all(first.id))
            second = store.seed_sample_case(SAMPLE)
            second_count = len(store.all(second.id))
        self.assertEqual(first.id, second.id)
        self.assertEqual(first_count, second_count)
        self.assertGreater(first_count, 0)

    def test_manual_evidence_round_trips_with_entities(self):
        manual = create_manual_evidence(
            "Witness statement",
            "witness_note",
            "Maya Rao met Leo Grant near Northstar Energy. Keep confidential.",
            source_person="Jordan Lee",
            tags=["witness", "murder"],
        )
        enrich_entities([manual])
        with TemporaryDirectory() as temp_dir:
            store = EvidenceStore(Path(temp_dir) / "test_evidence.sqlite")
            record = store.create_case("Manual Case")
            store.upsert_many(record.id, [manual])
            loaded = store.all(record.id)
        self.assertEqual(loaded[0].source_type, "witness_note")
        self.assertIn("Maya Rao", loaded[0].entities["people"])
        self.assertIn("confidential", loaded[0].entities["risk_terms"])

    def test_custom_case_answer_uses_only_selected_case_evidence(self):
        northstar = create_manual_evidence(
            "Northstar note",
            "report",
            "Maya Rao discussed Northstar Energy with Leo Grant.",
        )
        unrelated = create_manual_evidence(
            "Warehouse note",
            "scene_note",
            "A witness saw Jordan Lee at the warehouse.",
        )
        enrich_entities([northstar, unrelated])
        case_one = InvestigationCase([northstar])
        case_two = InvestigationCase([unrelated])
        supported = InvestigationAgent(case_one).answer("What connects Maya Rao to Northstar Energy?")
        unsupported = InvestigationAgent(case_two).answer("What connects Maya Rao to Northstar Energy?")
        self.assertTrue(supported.citations)
        self.assertEqual(unsupported.confidence, "none")
        self.assertEqual(unsupported.fallback_reason, "unsupported_question")

    def test_groq_not_called_when_case_llm_disabled(self):
        case = build_case_from_folder(SAMPLE)
        llm = SpyLLM()
        agent = InvestigationAgent(case, None)
        answer = agent.answer("What connects Maya Rao to Northstar Energy?")
        self.assertFalse(llm.answer_calls)
        self.assertEqual(answer.mode, "local")

    def test_groq_citation_rejection_reports_fallback_reason(self):
        case = build_case_from_folder(SAMPLE)
        llm = SpyLLM("Enhanced answer without citations")
        answer = InvestigationAgent(case, llm).answer("What connects Maya Rao to Northstar Energy?")
        self.assertEqual(answer.mode, "local")
        self.assertEqual(answer.fallback_reason, "groq_invalid_citations")

    def test_manual_murder_case_note_with_unknown_date_is_unknown(self):
        manual = create_manual_evidence(
            "Witness heard argument",
            "witness_note",
            "Jordan Lee heard an argument behind the scene entrance.",
            date_value=None,
            source_person="Pat Morgan",
        )
        case = InvestigationCase(enrich_entities([manual]))
        known, unknown = case.timeline()
        self.assertFalse(known)
        self.assertEqual(unknown[0].title, "Witness heard argument")

    def test_deduplicate_removes_manual_and_uploaded_duplicate_within_case(self):
        uploaded = parse_text_file(SAMPLE / "03_note.txt")
        manual = create_manual_evidence(uploaded.title, uploaded.source_type, uploaded.body, source_person=uploaded.source)
        manual.content_hash = uploaded.content_hash
        self.assertEqual(len(deduplicate([uploaded, manual])), 1)

    def test_memo_includes_custom_case_citations(self):
        manual = create_manual_evidence(
            "Scene report",
            "scene_note",
            "Maya Rao noted a confidential side letter at Northstar Energy.",
        )
        case = InvestigationCase(enrich_entities([manual]))
        memo = InvestigationAgent(case).memo("Custom Case")
        self.assertIn(manual.id, memo)

    def test_leadboard_ranks_people_by_cited_investigative_signals(self):
        scene = create_manual_evidence(
            "Scene statement",
            "scene_note",
            "Jordan Lee met Maya Rao near the loading bay. Keep confidential.",
            source_person="Pat Morgan",
        )
        interview = create_manual_evidence(
            "Interview follow-up",
            "interview",
            "Jordan Lee discussed the side letter with Northstar Energy.",
            date_value="2026-02-04",
            source_person="Case Analyst",
        )
        background = create_manual_evidence(
            "Background note",
            "other",
            "Maya Rao attended a planning meeting.",
        )
        case = InvestigationCase(enrich_entities([scene, interview, background]))
        leads = case.leadboard()
        self.assertTrue(leads)
        self.assertEqual(leads[0].name, "Jordan Lee")
        self.assertGreater(leads[0].score, leads[-1].score)
        self.assertTrue(leads[0].citations)

    def test_leadboard_reasons_refuse_culprit_finding(self):
        note = create_manual_evidence(
            "Witness note",
            "witness_note",
            "Jordan Lee was named by a witness.",
        )
        case = InvestigationCase(enrich_entities([note]))
        lead = case.leadboard()[0]
        self.assertTrue(any("not a guilt" in reason for reason in lead.reasons))

    def test_leadboard_filters_non_person_phrases_and_victim(self):
        report = create_manual_evidence(
            "Internal Budget review",
            "report",
            (
                "Meridian Biologics reported that victim Elena Voss was found in Lab 3B. "
                "Finance Officer reviewed an Internal Budget memo. Marcus Hale argued with Priya Nair "
                "near the rear entrance and told Owen Vale to delete footage."
            ),
            date_value="2026-02-04",
            source_person="Case Analyst",
        )
        case = InvestigationCase(enrich_entities([report]))
        names = [lead.name for lead in case.leadboard()]
        self.assertIn("Marcus Hale", names)
        self.assertIn("Priya Nair", names)
        self.assertIn("Owen Vale", names)
        self.assertNotIn("Meridian Biologics", names)
        self.assertNotIn("Finance Officer", names)
        self.assertNotIn("Internal Budget", names)
        self.assertNotIn("Elena Voss", names)

    def test_leadboard_scores_do_not_saturate_for_single_item(self):
        note = create_manual_evidence(
            "Scene note",
            "scene_note",
            "Marcus Hale met Priya Nair at Lab 3B. Keep confidential.",
            date_value="2026-02-04",
        )
        case = InvestigationCase(enrich_entities([note]))
        self.assertTrue(all(lead.score < 100 for lead in case.leadboard()))


if __name__ == "__main__":
    unittest.main()
