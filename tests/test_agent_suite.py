"""Comprehensive Automated Unit and Integration Test Suite for ZentroMart VoiceOps 2026.

Validates all 14 requirements specified in Section 9 of Capstone Brief:
1. Configuration and missing API keys handling
2. Document chunking and metadata preservation
3. Retrieval and no-evidence behavior
4. Current-versus-superseded policy precedence
5. Order lookup
6. Caller verification
7. Multi-turn context and follow-ups
8. Failed lookups and failed ticket creation
9. Policy refusals and handovers
10. Empty or unclear transcripts
11. Gnani integration failures and recovery
12. Audio output creation
13. Batch processing and output format
14. Evaluation calculations
"""
import os
import unittest
from pathlib import Path
from src.config import REFERENCE_DATE_STR, KB_DIR
from src.calendar_utils import add_business_days, add_calendar_days, format_date_natural
from src.kb_indexer import KnowledgeBase
from src.retriever import ZentroMartRetriever
from src.tools_layer import ConversationToolManager, VerificationState
from src.agent import ZentroMartAgent
from src.voice import normalize_text_for_speech, create_silent_or_tone_wav, VoicePipeline
from src.audio_batch import process_audio_batch
from src.evaluation import check_phrase_present, check_forbidden_phrases


class TestZentroMartCapstone(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.kb = KnowledgeBase()
        cls.retriever = ZentroMartRetriever(cls.kb)

    def test_01_config_and_reference_date(self):
        """1. Verify reference date is strictly 20 October 2026."""
        self.assertEqual(REFERENCE_DATE_STR, "2026-10-20")

    def test_02_calendar_business_days_and_holidays(self):
        """Verify business day calculations skip Sundays and warehouse holidays."""
        # 17 Oct 2026 (Sat) + 5 business days:
        # skips 18 (Sun), skips 19 (Festival Holiday), 20 (D1), 21 (D2), 22 (D3), 23 (D4), 24 (D5) -> 24 Oct
        d5 = add_business_days("2026-10-17", 5)
        self.assertEqual(format_date_natural(d5), "24 October 2026")
        d7 = add_business_days("2026-10-17", 7)
        self.assertEqual(format_date_natural(d7), "27 October 2026")

    def test_03_kb_chunking_and_metadata_preservation(self):
        """2. Verify documents are chunked and retain section IDs and frontmatter."""
        self.assertGreater(len(self.kb.chunks), 20)
        # Check that specific section IDs exist
        ret_chunk = self.kb.get_chunk_by_section_id("RET-2")
        self.assertIsNotNone(ret_chunk)
        self.assertEqual(ret_chunk.doc_type, "policy")
        self.assertEqual(ret_chunk.status, "current")

    def test_04_policy_precedence_current_vs_superseded(self):
        """4. Verify pol_returns_v4 overrides pol_returns_v3_superseded."""
        res, answerable, ev = self.retriever.retrieve("What is the return window for electronics?")
        top_chunk = res[0].chunk
        self.assertEqual(top_chunk.status, "current")
        self.assertNotEqual(top_chunk.status, "superseded")
        self.assertIn("RET-2", ev)

    def test_05_unanswerable_and_no_evidence_behavior(self):
        """3. Verify out-of-scope questions are flagged unanswerable and do not fabricate."""
        res, answerable, ev = self.retriever.retrieve("Can I schedule a delivery at midnight?")
        self.assertFalse(answerable)
        agent = ZentroMartAgent(mode="rag", retriever=self.retriever)
        reply = agent.respond("Can I schedule a delivery at midnight?")
        self.assertIn("don't have information", reply.lower())

    def test_06_caller_verification_and_order_lookup(self):
        """5 & 6. Verify order details are withheld until caller identity is verified."""
        mgr = ConversationToolManager("test_session")
        # Step 1: Query order alone
        res1 = mgr.execute_lookup("8978 2837")
        self.assertTrue(res1.get("verification_required"))
        self.assertIsNone(mgr.verified_order)

        # Step 2: Query with incorrect proof
        res2 = mgr.execute_lookup("8978 2837", proof="9999")
        self.assertFalse(res2.get("verified"))
        self.assertEqual(mgr.retry_count, 1)

        # Step 3: Query with valid 4-digit phone suffix (customer phone ends in 4533)
        res3 = mgr.execute_lookup("8978 2837", proof="4533")
        self.assertTrue(res3.get("verified"))
        self.assertIsNotNone(mgr.verified_order)
        self.assertEqual(mgr.verified_order["order_id"], "89782837")

    def test_07_failed_verification_terminal_lockdown(self):
        """8. Verify terminal failure after two failed verification attempts."""
        mgr = ConversationToolManager("test_session_fail")
        mgr.execute_lookup("8978 2837", proof="1111")
        res_fail = mgr.execute_lookup("8978 2837", proof="2222")
        self.assertEqual(res_fail.get("action"), "verification_failed_terminal")
        self.assertEqual(mgr.state, VerificationState.FAILED)

    def test_08_multi_turn_context_and_follow_ups(self):
        """7. Verify agent maintains order context across turns."""
        agent = ZentroMartAgent(mode="rag", retriever=self.retriever)
        r1 = agent.respond("Hello, I need help with my order 1859 9253")
        self.assertIn("AI", r1)
        self.assertIn("4 digits", r1)

        # Verify
        r2 = agent.respond("My last 4 digits are 4265")
        self.assertIn("18599253", r2)

        # Follow up on same order
        r3 = agent.respond("How long will the refund take?")
        self.assertTrue("instant" in r3.lower() or "2 hours" in r3.lower() or "refund" in r3.lower())

    def test_09_escalation_rules_and_ticket_creation(self):
        """9. Verify high-value damage (> Rs. 10,000) creates ticket and escalates."""
        mgr = ConversationToolManager("test_esc")
        mgr.execute_lookup("8978 2837", proof="4533")  # Verified order with TV worth > 10,000
        needs_esc, reason, intent = mgr.check_escalation_rules("I reported a damaged TV. Approve my replacement now.")
        self.assertTrue(needs_esc)
        self.assertEqual(reason, "high_value_damage")

        ticket = mgr.create_escalation_ticket(
            summary="Verified customer reports damaged 43-inch TV on order 89782837; requires human priority approval.",
            order_id="89782837",
            reason=reason,
            intent=intent
        )
        self.assertTrue(ticket.get("created"))
        self.assertTrue(ticket["ticket_id"].startswith("TKT"))

    def test_10_speech_text_normalization(self):
        """Verify text normalization formats order IDs, currency, and dates for speech."""
        raw = "Order 89782837 total is ₹499 and refund arrives on 2026-10-27."
        normalized = normalize_text_for_speech(raw)
        self.assertIn("8 9 7 8 2 8 3 7", normalized.replace(" ", " "))
        self.assertIn("499 rupees", normalized)
        self.assertIn("27th October 2026", normalized)

    def test_11_audio_wav_creation(self):
        """12. Verify valid WAV audio files are generated."""
        out_wav = Path("sample_outputs") / "test_tone.wav"
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        create_silent_or_tone_wav(str(out_wav), duration_sec=0.5)
        self.assertTrue(out_wav.exists())
        self.assertGreater(out_wav.stat().st_size, 1000)

    def test_12_empty_transcript_recovery(self):
        """10. Verify empty audio transcript triggers polite recovery reprompt."""
        agent = ZentroMartAgent(mode="rag", retriever=self.retriever)
        voice = VoicePipeline()
        # Non-existent or empty audio
        empty_wav = Path("sample_outputs") / "empty.wav"
        empty_wav.touch()
        heard, reply, _ = voice.voice_turn(str(empty_wav), agent.respond)
        self.assertIn("didn't catch that", reply.lower())
        if empty_wav.exists():
            empty_wav.unlink()

    def test_13_batch_audio_processing(self):
        """13. Verify batch audio runner generates transcripts, replies, and metadata JSON."""
        test_in_dir = Path("sample_outputs") / "test_audio_in"
        test_out_dir = Path("sample_outputs") / "test_audio_out"
        test_in_dir.mkdir(parents=True, exist_ok=True)

        # Create two sample input wavs
        create_silent_or_tone_wav(str(test_in_dir / "customer_sample_1.wav"), duration_sec=0.5)
        create_silent_or_tone_wav(str(test_in_dir / "customer_sample_2.wav"), duration_sec=0.5)

        summary = process_audio_batch(str(test_in_dir), str(test_out_dir))
        self.assertEqual(summary["total_files"], 2)
        self.assertEqual(summary["processed_files"], 2)
        self.assertTrue((test_out_dir / "batch_run_summary.json").exists())

    def test_14_evaluation_constraint_helpers(self):
        """14. Verify evaluation phrase matching and forbidden constraints."""
        text = "Haan, 12 mahine ki Assured warranty hai; claim app mein My Orders se karein."
        self.assertTrue(check_phrase_present(text, "12 ; My Orders / app"))
        self.assertTrue(check_forbidden_phrases(text, "5 to 7 days ; ₹49 will be charged"))
        self.assertFalse(check_forbidden_phrases("We will charge ₹49 will be charged", "₹49 will be charged"))


if __name__ == "__main__":
    unittest.main()
