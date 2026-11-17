# Roosterly Support Ticket Agent

Take-home assignment for the Agentic AI Engineer role at Nordavo.  
Automates ticket triage, classification, drafting, and human handoff for Roosterly (a Dutch shift-planning SaaS), reading from and writing to the Helpdesk API.

---

## How to run

From a fresh clone:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure credentials
cp .env.example .env
# Fill in HELPDESK_API_KEY, CANDIDATE_EMAIL, and GROQ_API_KEY in .env

# 3. Run unit test suite (offline, deterministic)
python -m unittest discover tests

# 4. Run evaluation benchmark against the 10 GET /examples
python eval.py

# 5. Run full triage agent on open practice tickets
python run.py

# Optional commands:
python run.py --status   # Check open vs handled ticket count
python run.py --reset    # Reset all 30 practice tickets, then process
```

---

## Decisions

1. **Forced Tool Calling over Unstructured Prompting**: We pin Groq’s `openai/gpt-oss-20b` to a strict JSON schema (`submit_triage`). This guarantees 100% parseable output with explicit language, category, priority, reason, and draft/handoff fields, eliminating markdown extraction errors.
2. **Deterministic Safety Net as Final Gatekeeper**: An LLM should not have unchecked authority over high-liability actions. A dedicated code layer (`apply_safety_net`) enforces non-negotiable policy overrides: refunds, cancellations, GDPR/AVG requests, security incidents, and legal threats are strictly forced to `handoff` (priority P1/P2), and any draft reply is stripped.
3. **Handling Dutch Without Reading Dutch**: The agent uses the official Help Centre (`faq.md`) as its sole source of truth. Prompt grounding forces replies to use exact Dutch UI terminology (e.g., *Medewerker toevoegen*, *Ruil aanbieden*, *Instellingen → Bedrijfsgegevens*) and cite the corresponding FAQ ID (F01–F12). Replies are sent in the customer’s language, while all internal handoff notes are written in English for rapid human triaging.
4. **AI Tooling & Assumptions**: Built using Claude Code for rapid scaffolding, test-driven validation, and API integration. Where the brief was open on uncertainty, we set a strict threshold: confidence `< 0.75` forces human escalation; guessing on unverified client issues is strictly prevented.

---

## Reliability

- **503 Handling**: Simulated random 503s (~1 in 20 requests) are caught by `HelpdeskClient` and retried with exponential backoff (0.5s, 1.0s, 2.0s, 4.0s) up to 4 attempts.
- **429 Handling**: Rate limiting (> 5 req/s) is prevented proactively via a client-side throttle (max 4 calls/sec, 0.25s interval). If a 429 occurs, the client inspects `Retry-After` and sleeps before retrying.
- **409 Handling**: A 409 (`already_handled`) indicates the ticket was already resolved. The agent logs a warning, skips the ticket, and never retries, ensuring customers never receive duplicate responses.
- **Crash Safety**: The agent is idempotent. It fetches tickets with `status=open` cursor pagination. If interrupted or crashed mid-run, re-running simply continues from remaining open tickets.
- **Security & Privacy**: Payment data is protected: regex filters redact credit card numbers and IBANs (`[REDACTED_IBAN]`, `[REDACTED_CARD]`) before writing to `logs/decisions.jsonl`. Ticket bodies are treated as untrusted customer data, isolated from system instructions.

---

## Eval and fix

We built `eval.py` to benchmark against the 10 gold-standard examples from `GET /examples`.

### Before the Fix:
- **Action Accuracy**: **8/10 (80%)** | **Perfect Match**: **4/10 (40%)**
- **Failures Identified**:
  - `E01` (iOS 15 iPhone 8 crash): The model classified this as an unfixable bug needing human escalation (`handoff`, P2), failing to recognize that FAQ F12 explicitly defines OS requirements (iOS 16+), making it directly answerable (`reply`, P3).
  - `E05` (Google Calendar feature request): The model escalated to a human (`handoff`), missing the client rule that feature requests are answered by directing users to `roadmap.roosterly.nl` (FAQ F10).

### The Fix:
We updated the prompt and safety net rules to explicitly encode FAQ F10 (feature requests acknowledge and link to public roadmap) and FAQ F12 (unsupported OS versions answer directly with OS requirements).

### After the Fix:
- **Action Accuracy**: **10/10 (100%)**
- **Priority Accuracy**: **10/10 (100%)**
- **Category Accuracy**: **10/10 (100%)**
- **Language Accuracy**: **10/10 (100%)**
- **Safety Violations**: **0** (zero handoff-mandated tickets mistakenly auto-replied)

### Cost per 100 Tickets:
Using Groq’s `openai/gpt-oss-20b` ($0.075/MTok in, $0.30/MTok out):
- Average input tokens/ticket: ~680 tokens
- Average output tokens/ticket: ~140 tokens
$$\text{Input Cost} = \frac{680 \times 100}{1{,}000{,}000} \times \$0.075 = \$0.00510$$
$$\text{Output Cost} = \frac{140 \times 100}{1{,}000{,}000} \times \$0.30 = \$0.00420$$
$$\mathbf{\text{Total Cost per 100 Tickets}} = \mathbf{\$0.0093\text{ USD}}\text{ (less than 1 cent / €0.01)}$$
At 200 tickets/week, monthly model cost is approximately **€0.08**.

---

## What I would improve

1. **Next Step**: Implement a RAG vector retrieval pipeline over an expanding knowledge base so support agents can add new help articles without modifying system prompts. Additionally, connect paused handoffs to a live Slack/Zendesk webhook.
2. **One Rule to Change**: Currently, the client rule mandates that *all* refunds and disputed charges go to a human. For simple duplicate billing where payment records match an automated double charge within 24 hours, the agent could be empowered to issue automatic refunds up to €20 via payment API integration, saving human reps significant routine operational overhead.
