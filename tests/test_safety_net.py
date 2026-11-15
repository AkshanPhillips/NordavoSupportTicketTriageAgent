import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from schema import TriageResult
from agent import apply_safety_net, CONFIDENCE_ESCALATION_THRESHOLD


class TestSafetyNet(unittest.TestCase):
    def test_low_confidence_forces_escalation(self):
        result = TriageResult(
            ticket_id="T01",
            category="question",
            priority="low",
            priority_reason="Uncertain question.",
            confidence=0.60,  # Below threshold
            can_auto_reply=True,
            draft_reply="Try clearing cache.",
            needs_human=False,
            escalation_reason=None,
        )
        safe = apply_safety_net(result, "Help", "I might have an issue")
        self.assertTrue(safe.needs_human)
        self.assertIsNone(safe.draft_reply)
        self.assertIn("confidence 0.60 below threshold", safe.escalation_reason)

    def test_refund_keyword_forces_escalation(self):
        result = TriageResult(
            ticket_id="T02",
            category="billing",
            priority="high",
            priority_reason="Refund ask.",
            confidence=0.95,
            can_auto_reply=False,
            draft_reply=None,
            needs_human=False,
            escalation_reason=None,
        )
        safe = apply_safety_net(result, "Please refund my subscription", "Charged twice.")
        self.assertTrue(safe.needs_human)
        self.assertIn("refund", safe.escalation_reason.lower())

    def test_bug_and_billing_never_auto_reply(self):
        for cat in ("bug", "billing"):
            result = TriageResult(
                ticket_id="T03",
                category=cat,
                priority="medium",
                priority_reason="Issue found.",
                confidence=0.90,
                can_auto_reply=True,
                draft_reply="Here is a quick answer.",
                needs_human=False,
                escalation_reason=None,
            )
            safe = apply_safety_net(result, "Issue title", "Something happened")
            self.assertFalse(safe.can_auto_reply)
            self.assertIsNone(safe.draft_reply)

    def test_data_loss_and_security_escalate(self):
        cases = [
            ("Data loss after update", "My database records were deleted."),
            ("Suspicious login alert", "Login from unknown IP address, compromised?"),
            ("SAML error", "SSO integration failed with SAML assertion error."),
            ("Compliance audit", "What is your data retention policy?"),
        ]
        for subj, body in cases:
            result = TriageResult(
                ticket_id="T04",
                category="other",
                priority="high",
                priority_reason="Important.",
                confidence=0.95,
                can_auto_reply=False,
                draft_reply=None,
                needs_human=False,
                escalation_reason=None,
            )
            safe = apply_safety_net(result, subj, body)
            self.assertTrue(safe.needs_human, f"Failed to escalate for {subj}")


if __name__ == "__main__":
    unittest.main()
