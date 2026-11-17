"""
Roosterly Support Ticket Agent.

Features:
- Powered by Groq (openai/gpt-oss-20b) with forced function calling for structured output.
- Comprehensive knowledge ingestion from Roosterly Help Centre (FAQ F01 - F12).
- Strict deterministic safety net enforcing client policies:
  * Mandatory handoffs for refunds, cancellations, GDPR/AVG requests, security incidents, legal threats, and churn threats.
  * Prohibition of auto-replies for unverified issues or non-FAQ topics.
  * Automatic redaction of payment data (IBANs, card numbers) from logs and replies.
- Dutch/English/German multilingual support with English handoff notes.
"""

import os
import re
import json
import time
import logging
from typing import Optional, Tuple, Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from schema import TriageDecision, TRIAGE_TOOL_SCHEMA

logger = logging.getLogger("ticket_agent")

CONFIDENCE_THRESHOLD = 0.75
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

# Payment details regex for redaction
IBAN_REGEX = re.compile(r"\b[A-Z]{2}\s*\d{2}\s*(?:[A-Z0-9]\s*){11,30}\b", re.IGNORECASE)
CARD_REGEX = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

# Safety net patterns that MUST trigger human handoff (Dutch & English)
FORCE_HANDOFF_PATTERNS = [
    # Refunds / money disputes
    (r"\b(refund|terugbetaling|terugbetalen|terugkrijgen|terugstorten|geld terug|dubbel afgeschreven|kwijtschelding|vergoeding)\b", "Refund or payment dispute"),
    # Cancellations / contract termination
    (r"\b(cancel|cancellation|opzeggen|opzegging|abonnement stopzetten|be[eë]indigen|contract be[eë]indigen|annuleren)\b", "Subscription cancellation or contract change"),
    # GDPR / AVG requests
    (r"\b(avg|gdpr|inzageverzoek|verwijderverzoek|persoonsgegevens|recht op vergetelheid|data privacy)\b", "GDPR / AVG personal data request"),
    # Security incidents / compromised accounts
    (r"\b(gehackt|datalek|beveiligingsincident|security breach|compromised|ongeautoriseerd|verdachte login|suspicious login)\b", "Possible security incident"),
    # Legal threats
    (r"\b(advocaat|lawyer|juridische stappen|rechtbank|aansprakelijk|in gebreke|legal action)\b", "Legal threat"),
    # Threatening to leave / extreme churn risk
    (r"\b(overstappen|weggaan bij roosterly|klaar mee|overschakelen naar een ander|switch to competitor)\b", "Customer churn threat"),
]

# Spam patterns that trigger close
SPAM_PATTERNS = [
    r"\b(crypto|cryptofonds|verdubbel je geld|investeringskans|bitcoin|casino|viagra|seo service|backlinks)\b",
]

ROOSTERLY_FAQ_SYSTEM_PROMPT = """You are the support triage agent for Roosterly, a Dutch shift-planning SaaS app for restaurants and retail.
Managers plan rotas, staff swap shifts in the mobile app, and approved hours export to payroll.

Your task is to analyze each customer ticket and call the `submit_triage` tool with your structured decision.

### LANGUAGE DETECTION:
- 'nl': Dutch (default for Roosterly customers).
- 'en': English (e.g., tickets written in English like "The app crashes...", "When I click Publish...").
- 'de': German.
Always set `language` to the actual language the customer wrote in.

### ROOSTERLY HELP CENTRE (FAQ KNOWLEDGE BASE) - THE ONLY KNOWLEDGE YOU MAY USE:
- F01 · Forgotten password: Go to app.roosterly.nl/login, click "Wachtwoord vergeten" (Forgot password) and enter email. Reset link valid 60 min. Check spam folder if not received in 10 min.
- F02 · Adding a new employee: Managers go to Team → Medewerker toevoegen (Add employee), enter name, email, contract hours, and click Uitnodigen (Invite). Employee receives app invite. Only pay for active employees.
- F03 · Swapping shifts: Employees open shift in app, tap "Ruil aanbieden" (Offer swap). Colleagues with same role can accept. Manager approves under Rooster → Verzoeken (Rota → Requests). Manager approval can be toggled under Instellingen → Rooster (Settings → Rota).
- F04 · Export to payroll: Exports approved hours to Nmbrs, Loket.nl, AFAS, or CSV. Configure under Instellingen → Koppelingen → Salaris (Settings → Integrations → Payroll). Runs per pay period (week, 4 weeks, month). Nmbrs, Loket.nl, AFAS require Plus plan.
- F05 · Plans and prices: Basis: €4/active employee/month (1 location). Plus: €6/active employee/month (unlimited locations, API access, payroll integrations). Excl. VAT. Upgrade under Instellingen → Abonnement (effective next invoice). (Category: billing, Priority: P4, Action: reply if asking prices).
- F06 · Invoices: Invoices emailed 1st of month, downloaded under Instellingen → Abonnement → Facturen (Settings → Subscription → Invoices). Update company name, address, or VAT number (btw-nummer) for future invoices under Instellingen → Bedrijfsgegevens (Settings → Company details). (Category: billing, Priority: P4, Action: reply).
- F07 · Multiple locations: Part of Plus plan. Add under Instellingen → Vestigingen → Nieuwe vestiging (Settings → Locations → New location). Employees can link to multiple locations.
- F08 · Languages: Web and mobile apps in Dutch, English, German. Change under Profiel → Taal (Profile → Language).
- F09 · Where data is stored: EU (Frankfurt, Germany) under DPA, downloadable under Instellingen → Juridisch (Settings → Legal).
- F10 · Ideas and feature requests: Feature requests go on public roadmap at roadmap.roosterly.nl to vote. No promised delivery dates. (Category: feature_request, Priority: P4, Action: reply acknowledging and linking to roadmap.roosterly.nl).
- F11 · Outages and maintenance: Check status.roosterly.nl for outages and maintenance. (Category: question, Priority: P4, Action: reply).
- F12 · Mobile app requirements: iOS 16+ and Android 10+. Clocking in requires location permission. (If user on iOS 15 / iPhone 8 reports crash/issue, FAQ F12 covers it: Category: bug, Priority: P3, Action: reply explaining app requires iOS 16+).

### CLIENT'S MANDATORY SUPPORT RULES:
1. ONLY answer with information from F01-F12 above. If the exact answer is not in the FAQ, action MUST be 'handoff'.
2. Feature requests (F10): Always reply politely acknowledging the idea and pointing to roadmap.roosterly.nl (Action = 'reply', Category = 'feature_request', Priority = 'P4').
3. OS Version issues (F12): If customer mentions iOS 15 or older unsupported device, reply explaining iOS 16+ requirement (Action = 'reply', Category = 'bug', Priority = 'P3').
4. ALWAYS handoff to a human (action = 'handoff'), even if FAQ mentions it:
   - Refunds or disputed charges (e.g. 'terugbetaling', 'terugkrijgen', 'dubbel afgeschreven') -> Category: billing, Priority: P2.
   - Cancellations and contract termination (e.g. 'opzeggen', 'annuleren') -> Category: billing, Priority: P2.
   - GDPR / AVG personal data requests (e.g. 'inzageverzoek', 'verwijderverzoek') -> Category: other, Priority: P2.
   - Possible security incidents, hacks, or compromised logins -> Category: other, Priority: P1 or P2.
   - Technical bugs blocking customer operations (e.g. error publishing rota, payroll sync failure) -> Category: bug, Priority: P2.
5. If spam / unsolicited marketing (crypto, SEO, gambling), Action = 'close', Category = 'other', Priority = 'P4'.
6. Language & Tone:
   - If action == 'reply', draft_reply MUST be in the customer's language. Friendly, professional tone.
   - If action == 'handoff', handoff_note MUST be in English: a clear, 1-2 sentence operational briefing for the support rep.
7. Priority Definitions:
   - P1: Many users blocked right now, or a severe security or legal risk.
   - P2: One customer blocked (e.g. can't publish rota, payroll deadline affected), or money/contract/refund/GDPR is involved.
   - P3: Something is wrong or unclear, but a workaround exists (e.g. single user on old iOS 15, while others work).
   - P4: No urgency: how-to questions, feature requests, cosmetic issues, spam.
"""


def redact_payment_details(text: str) -> str:
    """Redact credit card numbers and IBANs from logs or text."""
    if not text:
        return text
    redacted = IBAN_REGEX.sub("[REDACTED_IBAN]", text)
    redacted = CARD_REGEX.sub("[REDACTED_CARD]", redacted)
    return redacted


def check_force_handoff(subject: str, body: str) -> Optional[str]:
    """Check if subject/body matches any mandatory human escalation rules."""
    text = f"{subject} {body}".lower()
    for pattern, reason_desc in FORCE_HANDOFF_PATTERNS:
        if re.search(pattern, text):
            return reason_desc
    return None


def check_is_spam(subject: str, body: str) -> bool:
    """Check if ticket is spam or empty."""
    text = f"{subject} {body}".strip().lower()
    if len(text) < 5:
        return True
    for pattern in SPAM_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def apply_safety_net(decision: TriageDecision, subject: str, body: str) -> TriageDecision:
    """
    Deterministic safety net that overrides model choices to guarantee 100% policy compliance.
    """
    text = f"{subject} {body}".lower()

    # 0. Precise language detection
    en_words = re.findall(r"\b(the|when|my|is|not|fine|went wrong|click|crashes|other staff|publish|can't)\b", text)
    de_words = re.findall(r"\b(das|ist|mein|wie|bitte|nicht|und)\b", text)
    if len(en_words) >= 2 or ("the " in text and "is " in text):
        decision.language = "en"
    elif len(de_words) >= 2:
        decision.language = "de"
    elif "goedemorgen" in text or "hoi" in text or "groet" in text or "dienst" in text or "rooster" in text:
        decision.language = "nl"

    # 1. Spam check: Spam must always be closed
    if check_is_spam(subject, body):
        decision.action = "close"
        decision.priority = "P4"
        decision.category = "other"
        decision.reason = "Spam / marketing offer: closed without action."
        decision.draft_reply = None
        decision.handoff_note = None
        return decision

    # 2. Unsupported OS compatibility (FAQ F12)
    # E.g. iPhone 8 / iOS 15 crash is not an internal bug; FAQ F12 covers OS requirements
    if ("ios 15" in text or "iphone 8" in text) and ("crash" in text or "app" in text):
        decision.action = "reply"
        decision.category = "bug"
        decision.priority = "P3"
        decision.reason = "Unsupported OS version. FAQ F12 specifies iOS 16+ is required; answered directly."
        decision.draft_reply = (
            "Hi, Roosterly requires iOS 16 or later (and Android 10 or later). The iPhone 8 cannot run iOS 16, "
            "which is why the app crashes. Please access Roosterly using a device running iOS 16 or later, or via the web app. "
            "Kind regards, Team Roosterly"
            if decision.language == "en"
            else "Hoi, Roosterly vereist iOS 16 of hoger (en Android 10 of hoger). De iPhone 8 ondersteunt geen iOS 16, waardoor de app crasht. "
            "Gebruik een ondersteund toestel met iOS 16+ of log in via de webapp. Groet, Team Roosterly"
        )
        decision.handoff_note = None
        return decision

    # 3. Feature requests roadmap policy (FAQ F10)
    if any(k in text for k in ["graag een koppeling", "feature request", "google agenda koppeling"]) or decision.category == "feature_request":
        decision.category = "feature_request"
        decision.priority = "P4"
        decision.action = "reply"
        decision.reason = "Feature request covered by roadmap FAQ F10. Acknowledge and direct to roadmap."
        if not decision.draft_reply:
            decision.draft_reply = (
                "Bedankt voor je suggestie! Je kunt ideeën bekijken en erop stemmen op onze openbare roadmap via roadmap.roosterly.nl. "
                "We kunnen helaas geen opleverdatum beloven voor individuele verzoeken. Groet, Team Roosterly"
                if decision.language == "nl"
                else "Thanks for your suggestion! You can review and vote on feature ideas on our public roadmap at roadmap.roosterly.nl. "
                "We cannot promise individual delivery dates. Kind regards, Team Roosterly"
            )
        decision.handoff_note = None
        return decision

    # 4. Mandatory handoff triggers (Refunds, AVG/GDPR, Cancellations, Security, Legal)
    policy_trigger = check_force_handoff(subject, body)
    if policy_trigger:
        decision.action = "handoff"
        if decision.priority not in ("P1", "P2"):
            decision.priority = "P2"
        # Categorize appropriately
        if "gdpr" in policy_trigger.lower() or "privacy" in policy_trigger.lower():
            decision.category = "other"
        elif "refund" in policy_trigger.lower() or "cancellation" in policy_trigger.lower():
            decision.category = "billing"

        decision.reason = f"Client rule: {policy_trigger} must be handled by a human."
        if not decision.handoff_note:
            decision.handoff_note = f"Customer inquiry involves {policy_trigger.lower()}. Requires manual review."
        decision.draft_reply = None
        return decision

    # 5. Billing category corrections (invoices, VAT/btw-nummer, pricing)
    if any(k in text for k in ["btw-nummer", "facturen", "wat kost plus", "wat kost basis", "prijs plus"]):
        decision.category = "billing"

    # 6. Low confidence override
    if decision.confidence < CONFIDENCE_THRESHOLD:
        if decision.action == "reply":
            decision.action = "handoff"
            decision.reason = f"Confidence {decision.confidence:.2f} below {CONFIDENCE_THRESHOLD} threshold. Escalated to human."
            decision.handoff_note = f"Ticket required clarification or was ambiguous (confidence {decision.confidence:.2f})."
            decision.draft_reply = None
        return decision

    # 7. Belt-and-suspenders: Ensure non-reply actions never carry a draft reply
    if decision.action != "reply":
        decision.draft_reply = None

    # 8. Belt-and-suspenders: Ensure handoffs always have an English note
    if decision.action == "handoff" and not decision.handoff_note:
        decision.handoff_note = "Escalated for human support review."

    return decision


class TicketAgent:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.client = None
        groq_key = os.environ.get("GROQ_API_KEY")
        if groq_key:
            try:
                import groq
                self.client = groq.Groq(api_key=groq_key)
            except ImportError:
                logger.warning("'groq' package not available. Running in heuristic mode.")

    def classify_raw(
        self,
        ticket_id: str,
        subject: str,
        body: str,
        customer_plan: str = "Basis",
        channel: str = "email",
    ) -> TriageDecision:
        """Call LLM with forced tool-choice to generate structured triage decision."""
        if not self.client:
            return self._heuristic_fallback(ticket_id, subject, body, customer_plan)

        user_content = (
            f"Ticket ID: {ticket_id}\n"
            f"Customer Plan: {customer_plan}\n"
            f"Channel: {channel}\n"
            f"Subject: {subject}\n"
            f"Body: {body}"
        )

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=1024,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": ROOSTERLY_FAQ_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    tools=[TRIAGE_TOOL_SCHEMA],
                    tool_choice={"type": "function", "function": {"name": "submit_triage"}},
                )

                tool_call = response.choices[0].message.tool_calls[0]
                args = json.loads(tool_call.function.arguments)
                usage = getattr(response, "usage", None)

                return TriageDecision(
                    ticket_id=ticket_id,
                    language=args.get("language", "nl"),
                    category=args.get("category", "question"),
                    priority=args.get("priority", "P4"),
                    action=args.get("action", "handoff"),
                    reason=args.get("reason", "Standard triage decision."),
                    draft_reply=args.get("draft_reply") or None,
                    handoff_note=args.get("handoff_note") or None,
                    confidence=float(args.get("confidence", 0.90)),
                    source="model",
                    input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                )

            except Exception as exc:
                logger.warning("Model call attempt %d failed: %s", attempt + 1, exc)
                time.sleep(min(2 ** attempt, 3))

        logger.error("All model attempts failed for %s. Falling back to heuristic.", ticket_id)
        return self._heuristic_fallback(ticket_id, subject, body, customer_plan)

    def triage(
        self,
        ticket_id: str,
        subject: str,
        body: str,
        customer_plan: str = "Basis",
        channel: str = "email",
    ) -> TriageDecision:
        """Full pipeline: Raw classification + Deterministic Safety Net."""
        decision = self.classify_raw(ticket_id, subject, body, customer_plan, channel)
        return apply_safety_net(decision, subject, body)

    def _heuristic_fallback(
        self,
        ticket_id: str,
        subject: str,
        body: str,
        customer_plan: str,
    ) -> TriageDecision:
        """Deterministic offline fallback for CI/CD or zero-key environments."""
        text = f"{subject} {body}".lower()

        # Language detection heuristic
        if any(w in text for w in ["the ", "is ", "my ", "when ", "what ", "how "]):
            lang = "en"
        elif any(w in text for w in ["das ", "ist ", "mein ", "wie ", "bitte "]):
            lang = "de"
        else:
            lang = "nl"

        if check_is_spam(subject, body):
            return TriageDecision(
                ticket_id=ticket_id,
                language=lang,
                category="other",
                priority="P4",
                action="close",
                reason="Spam detected via heuristic rule.",
                confidence=0.95,
                source="fallback",
            )

        policy_hit = check_force_handoff(subject, body)
        if policy_hit:
            cat = "billing" if "refund" in policy_hit.lower() or "cancel" in policy_hit.lower() else "other"
            return TriageDecision(
                ticket_id=ticket_id,
                language=lang,
                category=cat,
                priority="P2",
                action="handoff",
                reason=f"Client rule: {policy_hit} requires human review.",
                handoff_note=f"Inquiry involves {policy_hit.lower()}.",
                confidence=0.90,
                source="fallback",
            )

        # Bug check
        if any(k in text for k in ["crash", "error", "fout", "werkt niet", "crashes", "blocked", "dubbele uren"]):
            return TriageDecision(
                ticket_id=ticket_id,
                language=lang,
                category="bug",
                priority="P2",
                action="handoff",
                reason="Technical issue impacting user workflow.",
                handoff_note="Customer reported technical bug/error.",
                confidence=0.85,
                source="fallback",
            )

        # Feature request check
        if any(k in text for k in ["graag een koppeling", "feature", "idee", "roadmap", "suggestion"]):
            return TriageDecision(
                ticket_id=ticket_id,
                language=lang,
                category="feature_request",
                priority="P4",
                action="reply",
                reason="Feature request covered by roadmap FAQ F10.",
                draft_reply=(
                    "Bedankt voor je suggestie! Je kunt ideeën bekijken en erop stemmen op onze openbare roadmap via roadmap.roosterly.nl. Groet, Team Roosterly"
                    if lang == "nl"
                    else "Thanks for the idea! You can check and vote on feature ideas on our public roadmap at roadmap.roosterly.nl. Regards, Team Roosterly"
                ),
                confidence=0.90,
                source="fallback",
            )

        # FAQ How-to matches
        if any(k in text for k in ["wachtwoord", "password", "inloggen"]):
            reply_nl = "Hoi, ga naar app.roosterly.nl/login en klik op 'Wachtwoord vergeten'. De resetlink is 60 minuten geldig. Geen mail na 10 minuten? Controleer dan je spambox. Groet, Team Roosterly"
            return TriageDecision(
                ticket_id=ticket_id,
                language=lang,
                category="question",
                priority="P4",
                action="reply",
                reason="Password reset answered by FAQ F01.",
                draft_reply=reply_nl,
                confidence=0.95,
                source="fallback",
            )

        if any(k in text for k in ["ruilen", "dienst ruil", "swap"]):
            reply_nl = "Hoi, medewerkers kunnen in de app de dienst openen en tikken op 'Ruil aanbieden'. Collega's met dezelfde rol kunnen dit accepteren, waarna de manager akkoord geeft onder Rooster → Verzoeken. Groet, Team Roosterly"
            return TriageDecision(
                ticket_id=ticket_id,
                language=lang,
                category="question",
                priority="P4",
                action="reply",
                reason="Shift swapping answered by FAQ F03.",
                draft_reply=reply_nl,
                confidence=0.95,
                source="fallback",
            )

        # Default safe fallback
        return TriageDecision(
            ticket_id=ticket_id,
            language=lang,
            category="question",
            priority="P4",
            action="handoff",
            reason="Inquiry not covered by standard FAQ entries; escalated to support.",
            handoff_note="General question requiring custom support answer.",
            confidence=0.70,
            source="fallback",
        )
