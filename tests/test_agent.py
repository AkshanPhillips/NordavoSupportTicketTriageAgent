import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from schema import TriageDecision
from agent import apply_safety_net, check_force_handoff, check_is_spam


class TestAgentSafetyNet(unittest.TestCase):
    def test_refund_forces_handoff_and_p2(self):
        decision = TriageDecision(
            ticket_id="T_REFUND",
            language="nl",
            category="billing",
            priority="P4",
            action="reply",
            reason="Model thought it could answer.",
            draft_reply="We will refund you.",
            confidence=0.95,
        )
        safe = apply_safety_net(decision, "Terugbetaling gevraagd", "Kunnen jullie het bedrag terugstorten?")
        self.assertEqual(safe.action, "handoff")
        self.assertEqual(safe.priority, "P2")
        self.assertIsNone(safe.draft_reply)
        self.assertIsNotNone(safe.handoff_note)
        self.assertIn("Refund or payment dispute", safe.reason)

    def test_cancellation_forces_handoff_and_p2(self):
        decision = TriageDecision(
            ticket_id="T_CANCEL",
            language="nl",
            category="billing",
            priority="P4",
            action="reply",
            reason="FAQ F05 mention.",
            draft_reply="Go to subscription.",
            confidence=0.90,
        )
        safe = apply_safety_net(decision, "Abonnement opzeggen", "Wij willen ons contract per direct beëindigen.")
        self.assertEqual(safe.action, "handoff")
        self.assertEqual(safe.priority, "P2")
        self.assertIsNone(safe.draft_reply)

    def test_gdpr_avg_forces_handoff_and_other_category(self):
        decision = TriageDecision(
            ticket_id="T_AVG",
            language="nl",
            category="question",
            priority="P4",
            action="reply",
            reason="Question about data.",
            draft_reply="Here is your data.",
            confidence=0.95,
        )
        safe = apply_safety_net(decision, "Inzageverzoek", "Ik wil graag inzage in mijn persoonsgegevens conform de AVG.")
        self.assertEqual(safe.action, "handoff")
        self.assertEqual(safe.priority, "P2")
        self.assertEqual(safe.category, "other")
        self.assertIsNone(safe.draft_reply)

    def test_spam_detection_forces_close(self):
        decision = TriageDecision(
            ticket_id="T_SPAM",
            language="nl",
            category="question",
            priority="P4",
            action="reply",
            reason="Marketing.",
            draft_reply="Not interested.",
            confidence=0.80,
        )
        safe = apply_safety_net(decision, "Investeringskans", "Verdubbel je geld met ons cryptofonds bitcoin!")
        self.assertEqual(safe.action, "close")
        self.assertEqual(safe.priority, "P4")
        self.assertIsNone(safe.draft_reply)

    def test_unsupported_ios15_covered_by_f12(self):
        decision = TriageDecision(
            ticket_id="T_IOS",
            language="en",
            category="bug",
            priority="P2",
            action="handoff",
            reason="Crash reported.",
            draft_reply=None,
            confidence=0.90,
        )
        safe = apply_safety_net(decision, "App crashes", "The app crashes immediately on my iPhone 8 (iOS 15).")
        self.assertEqual(safe.action, "reply")
        self.assertEqual(safe.priority, "P3")
        self.assertEqual(safe.category, "bug")
        self.assertIsNotNone(safe.draft_reply)
        self.assertIn("iOS 16", safe.draft_reply)

    def test_feature_request_covered_by_f10(self):
        decision = TriageDecision(
            ticket_id="T_FEAT",
            language="nl",
            category="feature_request",
            priority="P4",
            action="handoff",
            reason="Feature request.",
            draft_reply=None,
            confidence=0.90,
        )
        safe = apply_safety_net(decision, "Google Agenda", "Graag een koppeling met Google Agenda voor de diensten.")
        self.assertEqual(safe.action, "reply")
        self.assertEqual(safe.priority, "P4")
        self.assertEqual(safe.category, "feature_request")
        self.assertIsNotNone(safe.draft_reply)
        self.assertIn("roadmap.roosterly.nl", safe.draft_reply)


if __name__ == "__main__":
    unittest.main()
