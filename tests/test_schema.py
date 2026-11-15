import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from schema import TriageResult, VALID_CATEGORIES, VALID_PRIORITIES, TRIAGE_TOOL_SCHEMA


class TestSchema(unittest.TestCase):
    def test_valid_constants(self):
        self.assertIn("bug", VALID_CATEGORIES)
        self.assertIn("billing", VALID_CATEGORIES)
        self.assertIn("feature_request", VALID_CATEGORIES)
        self.assertIn("question", VALID_CATEGORIES)
        self.assertIn("other", VALID_CATEGORIES)

        self.assertIn("low", VALID_PRIORITIES)
        self.assertIn("medium", VALID_PRIORITIES)
        self.assertIn("high", VALID_PRIORITIES)
        self.assertIn("urgent", VALID_PRIORITIES)

    def test_triage_result_serialization(self):
        result = TriageResult(
            ticket_id="T100",
            category="question",
            priority="low",
            priority_reason="Simple how-to question.",
            confidence=0.95,
            can_auto_reply=True,
            draft_reply="You can export via Settings -> Export.",
            needs_human=False,
            escalation_reason=None,
            source="model",
            input_tokens=150,
            output_tokens=45,
        )
        d = result.to_dict()
        self.assertEqual(d["ticket_id"], "T100")
        self.assertEqual(d["category"], "question")
        self.assertEqual(d["priority"], "low")
        self.assertTrue(d["can_auto_reply"])
        self.assertFalse(d["needs_human"])
        self.assertEqual(d["input_tokens"], 150)
        self.assertEqual(d["output_tokens"], 45)

    def test_tool_schema_structure(self):
        self.assertEqual(TRIAGE_TOOL_SCHEMA["type"], "function")
        fn = TRIAGE_TOOL_SCHEMA["function"]
        self.assertEqual(fn["name"], "submit_triage")
        props = fn["parameters"]["properties"]
        self.assertIn("category", props)
        self.assertIn("priority", props)
        self.assertIn("can_auto_reply", props)
        self.assertIn("needs_human", props)
        self.assertIn("draft_reply", props)
        self.assertIn("confidence", props)


if __name__ == "__main__":
    unittest.main()
