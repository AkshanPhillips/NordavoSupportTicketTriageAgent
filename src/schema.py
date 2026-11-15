"""
Shared data model for a single ticket triage decision.

Every field here is deliberately simple (strings, enums, numbers) because the
agent's output is produced via *forced tool/function calling* (OpenAI-style,
which is what Groq's chat-completions API uses), not free-text parsing.
Keeping the schema small keeps the model's job easy and keeps validation
trivial on our side.

Note: `draft_reply` and `escalation_reason` use plain `"string"` (not a
nullable type) because some tool-calling backends handle `["string","null"]`
unions inconsistently. The model is instructed to send `""` when the field
doesn't apply, and we convert `"" -> None` right after parsing
(see agent.py `_normalize_empty`).
"""

from dataclasses import dataclass, asdict
from typing import Optional

VALID_CATEGORIES = ("bug", "billing", "feature_request", "question", "other")
VALID_PRIORITIES = ("low", "medium", "high", "urgent")

# OpenAI-style function/tool definition, used with Groq's chat.completions
# API (Groq is OpenAI-API-compatible for tool calling). This is the contract
# for structured output: the model must call this function with arguments
# matching this shape instead of replying in prose.
TRIAGE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_triage",
        "description": "Submit the triage decision for one support ticket.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": list(VALID_CATEGORIES),
                    "description": "The single best-fitting category for this ticket.",
                },
                "priority": {
                    "type": "string",
                    "enum": list(VALID_PRIORITIES),
                    "description": (
                        "Business urgency, considering customer tier, wording, "
                        "deadlines, data loss / security language, and whether "
                        "the customer is already upset."
                    ),
                },
                "priority_reason": {
                    "type": "string",
                    "description": "One short sentence explaining the priority.",
                },
                "confidence": {
                    "type": "number",
                    "description": (
                        "Your confidence (0.0-1.0) that the category AND "
                        "priority above are both correct."
                    ),
                },
                "can_auto_reply": {
                    "type": "boolean",
                    "description": (
                        "True only if this is a simple, unambiguous question "
                        "you can answer correctly without access to the "
                        "customer's account, billing records, or logs."
                    ),
                },
                "draft_reply": {
                    "type": "string",
                    "description": (
                        "A short, polite draft reply, ONLY if can_auto_reply "
                        "is true. Send an empty string \"\" otherwise."
                    ),
                },
                "needs_human": {
                    "type": "boolean",
                    "description": (
                        "True if a human should review this before anything is "
                        "sent or actioned (low confidence, billing/refund "
                        "money movement, security, data loss, legal/compliance, "
                        "angry customer, or anything can_auto_reply=false)."
                    ),
                },
                "escalation_reason": {
                    "type": "string",
                    "description": (
                        "If needs_human is true, one short sentence why. "
                        "Send an empty string \"\" otherwise."
                    ),
                },
            },
            "required": [
                "category",
                "priority",
                "priority_reason",
                "confidence",
                "can_auto_reply",
                "draft_reply",
                "needs_human",
                "escalation_reason",
            ],
        },
    },
}


@dataclass
class TriageResult:
    ticket_id: str
    category: str
    priority: str
    priority_reason: str
    confidence: float
    can_auto_reply: bool
    draft_reply: Optional[str]
    needs_human: bool
    escalation_reason: Optional[str]
    source: str = "model"  # "model" or "fallback" (see agent.py)
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict:
        return asdict(self)
