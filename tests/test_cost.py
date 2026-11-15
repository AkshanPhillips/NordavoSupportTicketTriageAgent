import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cost import CostSummary, estimate_from_sample, PRICING


class TestCost(unittest.TestCase):
    def test_cost_calculation(self):
        # 100 tickets with 600 in and 150 out
        # Rate: $0.075 / MTok in, $0.30 / MTok out
        summary = CostSummary(
            model="openai/gpt-oss-20b",
            n_tickets=100,
            total_input_tokens=60_000,
            total_output_tokens=15_000,
        )
        # Expected:
        # 60000 / 1e6 * 0.075 = 0.0045
        # 15000 / 1e6 * 0.30  = 0.0045
        # Total = 0.009 USD
        self.assertAlmostEqual(summary.total_cost_usd, 0.009, places=5)
        self.assertAlmostEqual(summary.cost_per_100_tickets_usd, 0.009, places=5)

    def test_estimate_from_sample(self):
        cost_per_100 = estimate_from_sample("openai/gpt-oss-20b", 600.0, 150.0)
        self.assertAlmostEqual(cost_per_100, 0.009, places=5)

    def test_report_generation(self):
        summary = CostSummary(
            model="openai/gpt-oss-20b",
            n_tickets=10,
            total_input_tokens=6000,
            total_output_tokens=1500,
        )
        report = summary.report()
        self.assertIn("openai/gpt-oss-20b", report)
        self.assertIn("Estimated cost per 100 tickets", report)


if __name__ == "__main__":
    unittest.main()
