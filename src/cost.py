"""
Cost estimation for the triage agent.

Pricing below is the published per-million-token rate on GroqCloud as of
Nov 2026 - verify current numbers at https://console.groq.com/docs/models
before relying on this for budgeting, since prices do change.

openai/gpt-oss-20b (this agent's default model):
  input:  $0.075 / MTok
  output: $0.30  / MTok

openai/gpt-oss-120b (optional upgrade for stronger judgment on ambiguous/
high-stakes tickets):
  input:  $0.15 / MTok
  output: $0.60 / MTok
"""

from dataclasses import dataclass

PRICING = {
    "openai/gpt-oss-20b": {"input": 0.075, "output": 0.30},
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
}


@dataclass
class CostSummary:
    model: str
    n_tickets: int
    total_input_tokens: int
    total_output_tokens: int

    @property
    def avg_input_tokens(self) -> float:
        return self.total_input_tokens / self.n_tickets if self.n_tickets else 0

    @property
    def avg_output_tokens(self) -> float:
        return self.total_output_tokens / self.n_tickets if self.n_tickets else 0

    @property
    def total_cost_usd(self) -> float:
        rates = PRICING.get(self.model)
        if not rates:
            return 0.0
        return (
            self.total_input_tokens / 1_000_000 * rates["input"]
            + self.total_output_tokens / 1_000_000 * rates["output"]
        )

    @property
    def cost_per_100_tickets_usd(self) -> float:
        if not self.n_tickets:
            return 0.0
        return self.total_cost_usd / self.n_tickets * 100

    def report(self) -> str:
        rates = PRICING.get(self.model, {"input": 0, "output": 0})
        lines = [
            f"Model: {self.model}",
            f"Tickets processed: {self.n_tickets}",
            f"Avg input tokens/ticket:  {self.avg_input_tokens:.0f}",
            f"Avg output tokens/ticket: {self.avg_output_tokens:.0f}",
            f"Rate: ${rates['input']:.3f}/MTok in, ${rates['output']:.2f}/MTok out",
            f"Measured cost for this batch: ${self.total_cost_usd:.5f}",
            f"=> Estimated cost per 100 tickets: ${self.cost_per_100_tickets_usd:.4f}",
        ]
        return "\n".join(lines)


def estimate_from_sample(model: str, avg_input_tokens: float, avg_output_tokens: float) -> float:
    """Project cost per 100 tickets from an average token count (used when
    no live run has happened yet, e.g. in README docs)."""
    rates = PRICING.get(model, {"input": 0, "output": 0})
    per_ticket = (
        avg_input_tokens / 1_000_000 * rates["input"]
        + avg_output_tokens / 1_000_000 * rates["output"]
    )
    return per_ticket * 100
