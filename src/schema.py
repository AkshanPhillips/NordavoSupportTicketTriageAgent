"""
Data models and function-calling schemas for Roosterly Support Ticket Triage.

Contract specifications (per Nordavo Helpdesk API reference):
- language: 'nl' | 'en' | 'de'
- category: 'bug' | 'billing' | 'feature_request' | 'question' | 'other'
- priority: 'P1' | 'P2' | 'P3' | 'P4'
- action: 'reply' | 'handoff' | 'close'
- reason: One or two sentences explaining why this category, priority, and action.
- draft_reply: Polite customer reply (in customer's language) if action is 'reply'.
- handoff_note: Operational note for support agent (in English) if action is 'handoff'.
- confidence: Float from 0.0 to 1.0.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

VALID_LANGUAGES = ("nl", "en", "de")
VALID_CATEGORIES = ("bug", "billing", "feature_request", "question", "other")
VALID_PRIORITIES = ("P1", "P2", "P3", "P4")
VALID_ACTIONS = ("reply", "handoff", "close")

TRIAGE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_triage",
        "description": "Submit structured triage decision for a Roosterly customer ticket.",
        "parameters": {
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "enum": list(VALID_LANGUAGES),
                    "description": "The language the customer wrote in (nl, en, de).",
                },
                "category": {
                    "type": "string",
                    "enum": list(VALID_CATEGORIES),
                    "description": "Primary category of the inquiry.",
                },
                "priority": {
                    "type": "string",
                    "enum": list(VALID_PRIORITIES),
                    "description": (
                        "P1: Many users blocked right now, or a security or legal risk. "
                        "P2: One customer blocked, or money or the contract is involved. "
                        "P3: Something is wrong or unclear, but there is a workaround. "
                        "P4: No urgency: how-to questions, ideas, cosmetic issues."
                    ),
                },
                "action": {
                    "type": "string",
                    "enum": list(VALID_ACTIONS),
                    "description": (
                        "reply: Answer directly using ONLY help centre facts (F01-F12). "
                        "handoff: Escalate to human support (refunds, bugs, GDPR/AVG, contract changes, angry customer, or not in FAQ). "
                        "close: Spam, marketing, crypto offers, or empty tickets."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": "One or two concise sentences in English explaining category, priority, and action.",
                },
                "draft_reply": {
                    "type": "string",
                    "description": (
                        "If action is 'reply', the polite answer in the customer's language based strictly on FAQ F01-F12. "
                        "Send empty string \"\" if action is not 'reply'."
                    ),
                },
                "handoff_note": {
                    "type": "string",
                    "description": (
                        "If action is 'handoff', an actionable summary in English for the human agent. "
                        "Send empty string \"\" if action is not 'handoff'."
                    ),
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score from 0.0 to 1.0.",
                },
            },
            "required": [
                "language",
                "category",
                "priority",
                "action",
                "reason",
                "draft_reply",
                "handoff_note",
                "confidence",
            ],
        },
    },
}


@dataclass
class TriageDecision:
    ticket_id: str
    language: str
    category: str
    priority: str
    action: str
    reason: str
    draft_reply: Optional[str] = None
    handoff_note: Optional[str] = None
    confidence: float = 1.0
    source: str = "model"  # "model" or "fallback"
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Backward-compatible alias
TriageResult = TriageDecision

