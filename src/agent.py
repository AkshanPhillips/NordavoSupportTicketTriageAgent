"""
Core triage agent - Groq-backed, with a LangGraph orchestration layer on top
(see graph.py).

Design in one paragraph: for each ticket we ask the model (via Groq) to fill
in a small structured "triage" object (category, priority, confidence,
optional draft reply, escalation flag) using forced function calling. We
never trust the model's escalation decision blindly - a short list of
hard-coded safety-net rules in `apply_safety_net()` can *override*
can_auto_reply/needs_human regardless of what the model said, because things
like refunds, security and data loss are exactly the cases where a silent
wrong guess is expensive. `classify_raw()` and `apply_safety_net()` are kept
as two separate functions (rather than one `triage()` call) specifically so
graph.py can run them as two distinct LangGraph nodes with a visible edge
between them.

If no GROQ_API_KEY is set, the agent falls back to a small keyword heuristic
(see `heuristic_classify`) so the whole pipeline still runs end-to-end for
demos/testing without API access. This is clearly marked in the output
(`source="fallback"`) and is NOT what you'd ship to production - see README
for why it exists.
"""

import os
import re
import time
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from schema import TriageResult, TRIAGE_TOOL_SCHEMA

MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
CONFIDENCE_ESCALATION_THRESHOLD = 0.75

SYSTEM_PROMPT = """You are a support-ticket triage assistant for a B2B SaaS product.

For every ticket you are given, call the submit_triage tool exactly once with
your decision. Rules of thumb:

- category: pick the single best fit:
  * bug: product defects, crashes, broken features, data loss, sync/login failures.
  * billing: charges, refunds, invoices, payment methods, VAT, seat pricing.
  * feature_request: requests for new features, integrations, or UX enhancements.
  * question: how-to questions, product usage, general inquiries.
  * other: security alerts, suspicious logins, or compromised account concerns (classify as "other", NOT "question" even if phrased as a question); praise, feedback, or non-product matters.
- priority:
  * urgent: reserved for active irreversible data loss, critical enterprise lockout with an imminent hard deadline (e.g. tomorrow), or active security compromise.
  * high: blocking bugs for paying customers (e.g. crash loop without data loss, SSO setup blocker), double charges / refund demands, or enterprise-tier issues without an immediate deadline.
  * medium: performance slowdowns, routine paid-tier inquiries, or non-blocking issues.
  * low: general free-tier inquiries, feature requests, cosmetic issues, or praise.
- can_auto_reply: only true for simple questions you could answer correctly
  from general product knowledge, with NO access to the customer's account,
  billing history, or logs. Never true for bugs, billing, or anything where
  being wrong would cost the customer money or time.
- needs_human: true whenever you are not highly confident, OR the ticket
  touches billing/refunds, security, data loss, legal/compliance, or a
  visibly upset customer. When in doubt, send it to a human - guessing wrong
  is worse than asking for review.
- confidence: be honest. 0.9+ only for genuinely unambiguous tickets.

Keep priority_reason and escalation_reason each to one short sentence.
Use an empty string "" for draft_reply / escalation_reason when they don't apply.
"""

# Keyword safety net: forces needs_human=True regardless of the model's
# confidence, because these categories carry outsized cost if handled wrong
# by an automated draft reply.
FORCE_ESCALATE_PATTERNS = [
    r"\brefund\b",
    r"charged (twice|wrong|in the wrong)",
    r"\bdata (loss|lost|disappeared|deleted)\b",
    r"\bcompromis(ed|e)\b",
    r"\bsuspicious login\b",
    r"\b(account locked|locked out)\b",
    r"\bsaml\b",
    r"\bsso\b.*\berror\b",
    r"\bvat\b",
    r"\bcompliance\b",
    r"\bretention policy\b",
]


def _matches_force_escalate(subject: str, body: str) -> Optional[str]:
    text = f"{subject} {body}".lower()
    for pattern in FORCE_ESCALATE_PATTERNS:
        if re.search(pattern, text):
            return f"Matched safety-net keyword pattern: /{pattern}/"
    return None


def _normalize_empty(value: Optional[str]) -> Optional[str]:
    """The model sends "" instead of null (see schema.py docstring)."""
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def apply_safety_net(result: TriageResult, subject: str, body: str) -> TriageResult:
    """Deterministic overrides applied on top of the model's output.

    This is intentionally simple and auditable (plain regex, not another
    model call) so a human reviewing the agent's behavior can see exactly
    why any given ticket was escalated, independent of model confidence.
    Run as its own LangGraph node (see graph.py) so it's a visible,
    separately-testable step rather than logic buried inside the model call.
    """
    reasons = []

    if result.confidence < CONFIDENCE_ESCALATION_THRESHOLD:
        result.needs_human = True
        reasons.append(
            f"confidence {result.confidence:.2f} below threshold "
            f"{CONFIDENCE_ESCALATION_THRESHOLD}"
        )

    keyword_hit = _matches_force_escalate(subject, body)
    if keyword_hit:
        result.needs_human = True
        reasons.append(keyword_hit)

    if result.can_auto_reply and result.category in ("bug", "billing"):
        # Never auto-reply to bugs/billing even if the model thought it could.
        result.can_auto_reply = False
        result.draft_reply = None
        reasons.append(f"category '{result.category}' is never auto-replied")

    if result.needs_human:
        # Belt-and-suspenders: nothing escalated to a human carries a draft
        # that might get sent automatically by some other part of the system.
        result.draft_reply = None

    if result.needs_human and reasons:
        extra = "; ".join(reasons)
        result.escalation_reason = (
            f"{result.escalation_reason}. Safety net: {extra}"
            if result.escalation_reason
            else f"Safety net: {extra}"
        )

    return result


def heuristic_classify(ticket_id: str, subject: str, body: str, tier: str) -> TriageResult:
    """Keyword-based fallback used only when no API key is configured.

    This exists purely so the pipeline is runnable end-to-end without API
    access (e.g. to sanity-check the data/plumbing). It is deliberately
    simple and is NOT a substitute for the model-based agent - see README.
    """
    text = f"{subject} {body}".lower()

    # 1. Security or account-alert issues are "other" (not bugs, not simple questions)
    if any(k in text for k in ["suspicious login", "compromised", "security concern", "security alert", "unauthorized"]):
        category = "other"
    # 2. Positive feedback / praise / non-actionable is "other"
    elif any(k in text for k in ["thank you", "great support", "appreciation", "no action needed"]):
        category = "other"
    # 3. Billing issues
    elif any(k in text for k in ["charged", "refund", "invoice", "billing", "vat", "currency", "payment", "subscription"]):
        category = "billing"
    # 4. Bugs and technical failures (crashes, errors, data loss, sync failures, lockouts)
    elif any(k in text for k in [
        "crash", "error", "bug", "freeze", "won't", "wont", "not working", "doesn't work",
        "broke", "cutting off", "500", "slow", "disappeared", "data loss", "locked out",
        "can't reset", "cannot reset", "not sending", "isn't sending", "failing", "spins forever"
    ]):
        category = "bug"
    # 5. Feature requests
    elif any(k in text for k in ["would be great", "feature", "suggestion", "idea", "shortcut", "could you add", "would love", "dark mode", "bulk edit"]):
        category = "feature_request"
    # 6. Questions
    elif any(k in text for k in ["how do i", "how do we", "what's the", "where is", "can i", "could i", "is there a way", "how long do you"]) or "?" in text:
        category = "question"
    else:
        category = "other"

    # Priority determination
    urgent_kw = ["extremely urgent", "urgently", "data loss", "compromised", "disappeared", "deadline tomorrow", "locked out"]
    high_kw = ["blocking", "every time", "unacceptable", "double charged", "charged twice", "refund", "crashes", "crash", "saml", "sso error", "deadline"]

    if any(k in text for k in urgent_kw):
        priority = "urgent"
    elif any(k in text for k in high_kw) or (tier == "enterprise" and category in ("bug", "billing")):
        priority = "high"
    elif tier == "enterprise" or tier == "pro":
        priority = "medium"
    else:
        priority = "low"

    can_auto_reply = category == "question" and priority == "low"
    needs_human = not can_auto_reply
    # Simple questions get high confidence so demo/offline runs can exercise
    # the auto_handle branch; everything else stays under the 0.75 escalation threshold.
    confidence = 0.85 if can_auto_reply else 0.55

    return TriageResult(
        ticket_id=ticket_id,
        category=category,
        priority=priority,
        priority_reason=f"Heuristic fallback (no API key): keyword match -> {priority}.",
        confidence=confidence,
        can_auto_reply=can_auto_reply,
        draft_reply=(
            "Thanks for reaching out! Could you share a bit more detail so "
            "our team can point you to the right answer?"
            if can_auto_reply
            else None
        ),
        needs_human=needs_human,
        escalation_reason=(
            None if can_auto_reply else "Heuristic fallback always escalates non-trivial tickets."
        ),
        source="fallback",
    )


class TicketAgent:
    def __init__(self, model: str = MODEL):
        self.model = model
        self.client = None
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key:
            try:
                import groq  # imported lazily so the module loads without the package too

                self.client = groq.Groq(api_key=api_key)
            except ImportError:
                print(
                    "[agent] 'groq' package not installed - run "
                    "`pip install groq`. Falling back to heuristic mode."
                )

    @property
    def using_model(self) -> bool:
        return self.client is not None

    def classify_raw(self, ticket_id: str, subject: str, body: str, tier: str) -> TriageResult:
        """Model (or fallback) call only - no safety net applied yet.

        Kept separate from apply_safety_net() so the two can be wired up as
        distinct LangGraph nodes with a visible edge between them (graph.py).
        """
        if not self.client:
            return heuristic_classify(ticket_id, subject, body, tier)

        user_message = f"Customer tier: {tier}\nSubject: {subject}\nBody: {body}"

        last_err = None
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=500,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    tools=[TRIAGE_TOOL_SCHEMA],
                    tool_choice={"type": "function", "function": {"name": "submit_triage"}},
                )
                message = response.choices[0].message
                tool_call = message.tool_calls[0]
                import json

                args = json.loads(tool_call.function.arguments)

                usage = getattr(response, "usage", None)
                result = TriageResult(
                    ticket_id=ticket_id,
                    category=args["category"],
                    priority=args["priority"],
                    priority_reason=args["priority_reason"],
                    confidence=float(args["confidence"]),
                    can_auto_reply=bool(args["can_auto_reply"]),
                    draft_reply=_normalize_empty(args.get("draft_reply")),
                    needs_human=bool(args["needs_human"]),
                    escalation_reason=_normalize_empty(args.get("escalation_reason")),
                    source="model",
                    input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                )
                return result
            except Exception as exc:  # noqa: BLE001 - we want to retry+fallback on anything
                last_err = exc
                time.sleep(min(2 ** attempt, 4))

        print(f"[agent] Model call failed for {ticket_id} after retries ({last_err}). Using fallback.")
        result = heuristic_classify(ticket_id, subject, body, tier)
        result.escalation_reason = f"{result.escalation_reason}; also: model call failed ({last_err})"
        result.needs_human = True
        return result

    def triage(self, ticket_id: str, subject: str, body: str, tier: str) -> TriageResult:
        """Convenience wrapper: classify_raw() + apply_safety_net() in one
        call. graph.py runs these as two separate nodes instead; this method
        is kept for simple scripts/tests that don't need the graph."""
        result = self.classify_raw(ticket_id, subject, body, tier)
        return apply_safety_net(result, subject, body)
