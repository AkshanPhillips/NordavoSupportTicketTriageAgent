import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from schema import TriageDecision, VALID_LANGUAGES, VALID_CATEGORIES, VALID_PRIORITIES, VALID_ACTIONS, TRIAGE_TOOL_SCHEMA


class TestSchema(unittest.TestCase):
    def test_valid_constants(self):
        self.assertIn("nl", VALID_LANGUAGES)
        self.assertIn("en", VALID_LANGUAGES)
        self.assertIn("de", VALID_LANGUAGES)

        self.assertIn("bug", VALID_CATEGORIES)
        self.assertIn("billing", VALID_CATEGORIES)
        self.assertIn("feature_request", VALID_CATEGORIES)
        self.assertIn("question", VALID_CATEGORIES)
        self.assertIn("other", VALID_CATEGORIES)

        self.assertIn("P1", VALID_PRIORITIES)
        self.assertIn("P2", VALID_PRIORITIES)
        self.assertIn("P3", VALID_PRIORITIES)
        self.assertIn("P4", VALID_PRIORITIES)

        self.assertIn("reply", VALID_ACTIONS)
        self.assertIn("handoff", VALID_ACTIONS)
        self.assertIn("close", VALID_ACTIONS)

    def test_decision_serialization(self):
        decision = TriageDecision(
            ticket_id="T100",
            language="nl",
            category="question",
            priority="P4",
            action="reply",
            reason="How-to answered by FAQ F02.",
            draft_reply="Hoi, ga naar Team...",
            confidence=0.95,
        )
        d = decision.to_dict()
        self.assertEqual(d["ticket_id"], "T100")
        self.assertEqual(d["language"], "nl")
        self.assertEqual(d["category"], "question")
        self.assertEqual(d["priority"], "P4")
        self.assertEqual(d["action"], "reply")

    def test_tool_schema_structure(self):
        self.assertEqual(TRIAGE_TOOL_SCHEMA["type"], "function")
        fn = TRIAGE_TOOL_SCHEMA["function"]
        self.assertEqual(fn["name"], "submit_triage")
        props = fn["parameters"]["properties"]
        self.assertIn("language", props)
        self.assertIn("category", props)
        self.assertIn("priority", props)
        self.assertIn("action", props)
        self.assertIn("reason", props)
        self.assertIn("draft_reply", props)
        self.assertIn("handoff_note", props)
        self.assertIn("confidence", props)


if __name__ == "__main__":
    unittest.main()
