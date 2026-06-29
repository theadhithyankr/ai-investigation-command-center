import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from evidenceiq.agents import InvestigationAgent
from evidenceiq.case import InvestigationCase
from evidenceiq.entities import (
    enrich_entities,
    entity_classifications,
    extract_entities,
    has_strict_person_type,
    is_plausible_person_name,
    leadable_people,
)
from evidenceiq.llm import build_answer_prompt, build_entity_prompt, parse_entity_response
from evidenceiq.parsing import (
    create_manual_evidence,
    deduplicate,
    parse_date,
    parse_manual_note_payload,
    parse_manual_note_payloads,
    parse_text_file,
)
from evidenceiq.pipeline import build_case_from_folder
from evidenceiq.search import EvidenceSearch
from evidenceiq.spatial import extract_map_pins
from evidenceiq.storage import EvidenceStore
from evidenceiq.wall import build_investigation_wall


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


class SpyEntityLLM:
    def __init__(self, entities):
        self.entities = entities
        self.calls = []

    def extract_entities(self, text):
        self.calls.append(text)
        return self.entities


class EvidenceIQTests(unittest.TestCase):
    def test_parse_date_handles_known_and_unknown_dates(self):
        self.assertEqual(parse_date("2026-02-03").year, 2026)
        self.assertEqual(parse_date("17/10/2023").day, 17)
        self.assertEqual(parse_date("17 October 2023").month, 10)
        self.assertEqual(parse_date("October 17, 2023").month, 10)
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

    def test_parse_entity_response_keeps_people_separate_from_roles_and_orgs(self):
        parsed = parse_entity_response(
            """
            {
              "people": ["Marcus Hale", "Finance Officer", "Meridian Biologics", "Second Street", "Upstairs Guest"],
              "organizations": ["Meridian Biologics"],
              "roles": ["Finance Officer"],
              "locations": ["Lab 3B", "Second Street"],
              "dates": ["17/10/2023"],
              "money": [],
              "risk_terms": ["delete"]
            }
            """
        )
        self.assertEqual(parsed["people"], ["Marcus Hale"])
        self.assertIn("Finance Officer", parsed["roles"])
        self.assertIn("Meridian Biologics", parsed["organizations"])

    def test_person_name_validator_rejects_title_and_location_phrases(self):
        self.assertTrue(is_plausible_person_name("Lizzie Borden"))
        self.assertTrue(is_plausible_person_name("Eli Bence"))
        self.assertFalse(is_plausible_person_name("Upstairs Guest"))
        self.assertFalse(is_plausible_person_name("Guest Room"))
        self.assertFalse(is_plausible_person_name("Second Street"))
        self.assertFalse(is_plausible_person_name("Failed Toxin"))
        self.assertFalse(is_plausible_person_name("Family Friend"))
        self.assertFalse(is_plausible_person_name("The Burned"))
        self.assertFalse(is_plausible_person_name("Detective Seaver"))

    def test_enrich_entities_uses_llm_people_over_regex_false_positives(self):
        item = create_manual_evidence(
            "Internal Budget review",
            "report",
            "Meridian Biologics and Finance Officer appear in the note. Marcus Hale told Priya Nair to delete footage.",
        )
        llm = SpyEntityLLM(
            {
                "people": ["Marcus Hale", "Priya Nair"],
                "organizations": ["Meridian Biologics"],
                "roles": ["Finance Officer"],
                "locations": ["Lab 3B"],
                "dates": [],
                "money": [],
                "risk_terms": ["delete"],
            }
        )
        enrich_entities([item], llm)
        self.assertEqual(item.entities["people"], ["Marcus Hale", "Priya Nair"])
        self.assertIn("Meridian Biologics", item.entities["organizations"])
        self.assertIn("Finance Officer", item.entities["roles"])
        self.assertTrue(llm.calls)

    def test_enrich_entities_removes_locations_from_people(self):
        item = create_manual_evidence(
            "Location noise",
            "scene_note",
            "Lizzie Borden was seen near Second Street and the Guest Room.",
        )
        llm = SpyEntityLLM(
            {
                "people": ["Lizzie Borden", "Second Street", "Guest Room", "Upstairs Guest"],
                "organizations": [],
                "roles": [],
                "locations": ["Second Street", "Guest Room"],
                "dates": [],
                "money": [],
                "risk_terms": [],
            }
        )
        enrich_entities([item], llm)
        self.assertEqual(item.entities["people"], ["Lizzie Borden"])

    def test_person_board_requires_strict_person_classification(self):
        item = create_manual_evidence(
            "Facility proximity",
            "report",
            "Lizzie Borden was seen near Union Church and Lincoln School.",
        )
        item.entities = {
            "people": ["Lizzie Borden", "Union Church", "Lincoln School", "City Hall"],
            "organizations": ["Union Church", "Lincoln School"],
            "locations": ["City Hall"],
            "roles": [],
            "emails": [],
            "money": [],
            "dates": [],
            "risk_terms": [],
            "victims": [],
        }

        self.assertEqual(entity_classifications(item, "Lizzie Borden"), ("PERSON",))
        self.assertEqual(entity_classifications(item, "Union Church"), ("ORGANIZATION", "PERSON"))
        self.assertEqual(entity_classifications(item, "City Hall"), ("FACILITY", "PERSON"))
        self.assertTrue(has_strict_person_type(item, "Lizzie Borden"))
        self.assertFalse(has_strict_person_type(item, "Union Church"))
        self.assertFalse(has_strict_person_type(item, "City Hall"))
        self.assertEqual(leadable_people(item), ["Lizzie Borden"])
        self.assertEqual([lead.name for lead in InvestigationCase([item]).leadboard()], ["Lizzie Borden"])

    def test_entity_prompt_rejects_institutional_people(self):
        payload = build_entity_prompt("John Morse walked past Union Church.")
        self.assertIn("Never classify churches, businesses, schools, or government bodies as people", payload.system)
        self.assertIn("belong in organizations or locations, never people", payload.system)

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

    def test_manual_note_payload_parses_json(self):
        payload = parse_manual_note_payload(
            """
            {
              "title": "Statement vs Log Discrepancies",
              "evidence_type": "contradiction_note",
              "date": "October 17, 2023",
              "source_person": "Detective Miller Case File",
              "tags": ["Contradiction", "Priya Nair"],
              "body": "Priya Nair and Marcus Hale gave conflicting accounts."
            }
            """
        )
        self.assertEqual(payload["title"], "Statement vs Log Discrepancies")
        self.assertEqual(payload["evidence_type"], "contradiction_note")
        self.assertFalse(payload["unknown_date"])
        self.assertIn("Priya Nair", payload["tags"])

    def test_manual_note_payload_parses_labeled_text_and_title_typo(self):
        payload = parse_manual_note_payload(
            """
            itle: Statement vs Log Discrepancies
            Evidence type: contradiction_note
            Date: October 17, 2023
            Source/person: Detective Miller Case File
            Tags: Contradiction, Priya Nair, Marcus Hale, Owen Vale
            Body:
            Priya Nair said she met Marcus Hale at 9:15 PM.
            Owen Vale's override code was used at 9:15 PM.
            """
        )
        self.assertEqual(payload["title"], "Statement vs Log Discrepancies")
        self.assertEqual(payload["evidence_type"], "contradiction_note")
        self.assertEqual(payload["source_person"], "Detective Miller Case File")
        self.assertEqual(payload["tags"], ["Contradiction", "Priya Nair", "Marcus Hale", "Owen Vale"])
        self.assertIn("override code", payload["body"])

    def test_manual_note_payloads_parse_json_array(self):
        payloads = parse_manual_note_payloads(
            """
            [
              {
                "title": "First note",
                "evidence_type": "witness_note",
                "date": "17/10/2023",
                "source_person": "Detective Miller",
                "tags": "Priya Nair, Marcus Hale",
                "body": "Priya Nair saw Marcus Hale near Lab 3B."
              },
              {
                "title": "Second note",
                "evidence_type": "scene_note",
                "date": "Unknown date",
                "source_person": "Scene team",
                "tags": ["Owen Vale"],
                "body": "Owen Vale was mentioned near the loading entrance."
              }
            ]
            """
        )
        self.assertEqual(len(payloads), 2)
        self.assertFalse(payloads[0]["unknown_date"])
        self.assertTrue(payloads[1]["unknown_date"])
        self.assertEqual(payloads[0]["tags"], ["Priya Nair", "Marcus Hale"])

    def test_manual_note_payloads_parse_multiple_labeled_notes(self):
        payloads = parse_manual_note_payloads(
            """
            --- EVIDENCE ITEM ---
            Title: First note
            Evidence type: witness_note
            Date: 17/10/2023
            Source/person: Detective Miller
            Tags: Priya Nair, Marcus Hale
            Body:
            Priya Nair saw Marcus Hale near Lab 3B.

            --- EVIDENCE ITEM ---
            Title: Second note
            Evidence type: scene_note
            Date: Unknown date
            Source/person: Scene team
            Tags: Owen Vale
            Body:
            Owen Vale was mentioned near the loading entrance.
            """
        )
        self.assertEqual(len(payloads), 2)
        self.assertEqual(payloads[0]["title"], "First note")
        self.assertTrue(payloads[1]["unknown_date"])

    def test_map_pins_parse_manual_location_metadata(self):
        item = create_manual_evidence(
            "Warehouse sighting",
            "scene_note",
            "Marcus Hale met Priya Nair by the loading door.",
            location="Warehouse A",
            latitude="12.9716",
            longitude="77.5946",
        )
        enrich_entities([item])
        pins = extract_map_pins([item])
        self.assertEqual(len(pins), 1)
        self.assertEqual(pins[0].location_label, "Warehouse A")
        self.assertEqual(pins[0].latitude, 12.9716)
        self.assertEqual(pins[0].longitude, 77.5946)

    def test_map_pins_parse_uploaded_coordinate_lines(self):
        item = create_manual_evidence(
            "Uploaded field note",
            "report",
            "Location: Lab 3B\nCoordinates: 40.7128, -74.0060\nPriya Nair noted an override.",
        )
        enrich_entities([item])
        pins = extract_map_pins([item])
        self.assertEqual(pins[0].location_label, "Lab 3B")
        self.assertEqual(pins[0].longitude, -74.006)

    def test_unknown_date_mapped_evidence_still_gets_pin_with_people(self):
        item = create_manual_evidence(
            "Unknown date location note",
            "witness_note",
            "Location: Safehouse\nCoordinates: 51.5074, -0.1278\nJordan Lee saw Maya Rao.",
            date_value=None,
        )
        enrich_entities([item])
        pin = extract_map_pins([item])[0]
        self.assertIsNone(pin.timestamp)
        self.assertIn("Jordan Lee", pin.people)
        self.assertIn("Maya Rao", pin.people)

    def test_risk_first_wall_connects_risk_to_evidence_to_entities(self):
        risky = create_manual_evidence(
            "Security override",
            "security_log",
            "Marcus Hale used an override near Priya Nair at Lab 3B.",
            location="Lab 3B",
            latitude="12.9716",
            longitude="77.5946",
        )
        case = InvestigationCase(enrich_entities([risky]))
        nodes, edges = build_investigation_wall(case)
        node_types = {node.node_type for node in nodes}
        relationships = {(edge.source.split(":", 1)[0], edge.target.split(":", 1)[0], edge.relationship) for edge in edges}
        self.assertIn("risk", node_types)
        self.assertIn("evidence", node_types)
        self.assertIn("person", node_types)
        self.assertIn(("risk", "evidence", "cites"), relationships)
        self.assertIn(("evidence", "person", "mentions"), relationships)

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

    def test_leadboard_filters_location_like_people_from_legacy_entities(self):
        item = create_manual_evidence(
            "Legacy noisy entities",
            "scene_note",
            "Lizzie Borden was near Second Street and the Guest Room.",
        )
        item.entities = {
            "people": ["Lizzie Borden", "Second Street", "Guest Room", "Upstairs Guest"],
            "organizations": [],
            "locations": ["Second Street", "Guest Room"],
            "roles": [],
            "emails": [],
            "money": [],
            "dates": [],
            "risk_terms": [],
            "victims": [],
        }
        names = [lead.name for lead in InvestigationCase([item]).leadboard()]
        self.assertEqual(names, ["Lizzie Borden"])

    def test_leadboard_filters_evidence_labels_relationships_and_official_roles(self):
        item = create_manual_evidence(
            "Borden noisy entities",
            "timeline_note",
            "Lizzie Borden was questioned. A failed toxin purchase, family friend note, and detective memo were logged.",
        )
        item.entities = {
            "people": ["Lizzie Borden", "Failed Toxin", "Family Friend", "The Burned", "Detective Seaver"],
            "organizations": [],
            "locations": [],
            "roles": ["Family Friend", "Detective Seaver"],
            "emails": [],
            "money": [],
            "dates": [],
            "risk_terms": ["confidential"],
            "victims": [],
        }
        names = [lead.name for lead in InvestigationCase([item]).leadboard()]
        self.assertEqual(names, ["Lizzie Borden"])

    def test_case_level_victims_are_excluded_from_leadboard(self):
        scene = create_manual_evidence(
            "Borden scene note",
            "scene_note",
            (
                "Andrew Borden was discovered deceased in the sitting room. "
                "Abby Borden was found dead upstairs. Lizzie Borden was present in the house."
            ),
            date_value="04/08/1892",
            source_person="Fall River Police",
            tags=["victim", "Andrew Borden", "Abby Borden"],
        )
        timeline = create_manual_evidence(
            "Borden timeline",
            "timeline_note",
            (
                "Andrew Borden appears in multiple timeline entries because he is the victim. "
                "Abby Borden also appears repeatedly as the deceased person. "
                "Lizzie Borden is connected to the burned dress and prussic acid inquiry."
            ),
            date_value="04/08/1892",
        )
        case = InvestigationCase(enrich_entities([scene, timeline]))
        victim_names = [victim.name for victim in case.victims()]
        lead_names = [lead.name for lead in case.leadboard()]
        self.assertIn("Andrew Borden", victim_names)
        self.assertIn("Abby Borden", victim_names)
        self.assertIn("Lizzie Borden", lead_names)
        self.assertNotIn("Andrew Borden", lead_names)
        self.assertNotIn("Abby Borden", lead_names)

    def test_borden_legacy_people_are_cleaned_across_lead_surfaces(self):
        scene = create_manual_evidence(
            "Property Dispute",
            "scene_note",
            (
                "Location: Borden House\n"
                "Coordinates: 41.7015, -71.1550\n"
                "Andrew Borden was found dead in the sitting room. "
                "Abby Borden was discovered deceased upstairs. "
                "Lizzie Borden, John Morse, Alice Russell, and Bridget Sullivan were documented."
            ),
            date_value="04/08/1892",
            source_person="Eli Bence (Pharmacist)",
            tags=["Property Dispute", "Prussic Acid", "Andrew Borden", "Abby Borden", "Bridget Sullivan"],
        )
        scene.entities = {
            "people": [
                "Property Dispute",
                "Prussic Acid",
                "Found Andrew",
                "Failed Toxin",
                "Andrew Borden",
                "Abby Borden",
                "Lizzie Borden",
                "John Morse",
                "Eli Bence",
                "Alice Russell",
                "Bridget Sullivan",
            ],
            "organizations": [],
            "locations": ["Borden House"],
            "roles": [],
            "emails": [],
            "money": [],
            "dates": [],
            "risk_terms": ["confidential"],
            "victims": ["Andrew Borden", "Abby Borden"],
        }
        case = InvestigationCase([scene])

        self.assertEqual(
            [lead.name for lead in case.leadboard()],
            ["Alice Russell", "Bridget Sullivan", "Eli Bence", "John Morse", "Lizzie Borden"],
        )
        self.assertCountEqual([victim.name for victim in case.victims()], ["Andrew Borden", "Abby Borden"])
        self.assertEqual(leadable_people(scene), ["Abby Borden", "Alice Russell", "Andrew Borden", "Bridget Sullivan", "Eli Bence", "John Morse", "Lizzie Borden"])
        self.assertEqual(extract_map_pins([scene])[0].people, tuple(leadable_people(scene)))

        nodes, _ = build_investigation_wall(case)
        person_labels = {node.label for node in nodes if node.node_type == "person"}
        self.assertEqual(
            person_labels,
            {"Abby Borden", "Alice Russell", "Andrew Borden", "Bridget Sullivan", "Eli Bence", "John Morse", "Lizzie Borden"},
        )
        for rejected in {"Property Dispute", "Prussic Acid", "Found Andrew", "Failed Toxin"}:
            self.assertNotIn(rejected, person_labels)

    def test_source_person_metadata_strips_role_suffix(self):
        item = create_manual_evidence(
            "Pharmacy interview",
            "interview",
            "Eli Bence described the prussic acid inquiry.",
            source_person="Eli Bence (Pharmacist)",
        )
        enrich_entities([item])
        self.assertIn("Eli Bence", item.entities["people"])
        self.assertNotIn("Eli Bence (Pharmacist)", item.entities["people"])

    def test_llm_victim_entities_are_excluded_from_leadboard(self):
        note = create_manual_evidence(
            "Structured extraction note",
            "scene_note",
            "Andrew Borden was found dead. Lizzie Borden was questioned about a burned dress.",
        )
        llm = SpyEntityLLM(
            {
                "people": ["Andrew Borden", "Lizzie Borden"],
                "victims": ["Andrew Borden"],
                "organizations": [],
                "roles": [],
                "locations": ["Borden house"],
                "dates": [],
                "money": [],
                "risk_terms": [],
            }
        )
        case = InvestigationCase(enrich_entities([note], llm))
        self.assertEqual([victim.name for victim in case.victims()], ["Andrew Borden"])
        self.assertEqual([lead.name for lead in case.leadboard()], ["Lizzie Borden"])

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
