import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent import TicketAgent
from graph import build_graph, run_ticket, start_review, resume_review
from langgraph.checkpoint.memory import InMemorySaver


class TestGraph(unittest.TestCase):
    def setUp(self):
        self.agent = TicketAgent()
        self.graph = build_graph(self.agent, human_in_the_loop=False)

    def test_run_ticket_auto_handle(self):
        # A simple free-tier question should route to auto_handle
        result = run_ticket(
            self.graph,
            ticket_id="T_TEST_1",
            subject="How do I export to CSV?",
            body="Where is the export button in settings?",
            tier="free",
        )
        self.assertEqual(result.category, "question")
        self.assertFalse(result.needs_human)
        self.assertIsNotNone(result.draft_reply)

    def test_run_ticket_human_queue(self):
        # A refund request must route to human_queue and strip any draft reply
        result = run_ticket(
            self.graph,
            ticket_id="T_TEST_2",
            subject="Refund request",
            body="I was charged twice, please refund.",
            tier="pro",
        )
        self.assertTrue(result.needs_human)
        self.assertIsNone(result.draft_reply)

    def test_interactive_human_review_pause_and_resume(self):
        review_graph = build_graph(self.agent, human_in_the_loop=True, checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "T_TEST_REVIEW"}}

        # Start review on an escalated ticket
        outcome = start_review(
            review_graph,
            config,
            ticket_id="T_TEST_REVIEW",
            subject="Urgent data loss",
            body="Data disappeared after update, please fix.",
            tier="enterprise",
        )

        # It must pause with an interrupt dictionary
        self.assertIsInstance(outcome, dict)
        self.assertEqual(outcome["ticket_id"], "T_TEST_REVIEW")

        # Resume with human decision to resolve
        decision = {"action": "resolve", "draft_reply": "Restored your backup successfully."}
        final_result = resume_review(review_graph, config, decision)
        self.assertFalse(final_result.needs_human)
        self.assertEqual(final_result.draft_reply, "Restored your backup successfully.")
        self.assertIn("Resolved by human reviewer", final_result.escalation_reason)


if __name__ == "__main__":
    unittest.main()
