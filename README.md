# Support Ticket Triage Agent

Take-home for the Nordavo Agentic AI Engineer role.

## What it does

Reads a CSV of support tickets and, for each one, decides:
1. **Category** — bug / billing / feature_request / question / other
2. **Priority** — low / medium / high / urgent, with a one-line reason
3. **Draft reply** — only for simple questions it's confident about
4. **Escalate to human** — anything it's not confident about, or that touches
   money, security, or data loss, goes to a human instead of being guessed
5. **Log** — every decision (including *why*) is written to `logs/decisions.jsonl`

LLM backend: **Groq** (`openai/gpt-oss-20b`, via https://console.groq.com).
Orchestration: a small **LangGraph** graph (see `src/graph.py`).

## Assumptions (no CSV was attached to the brief)

The brief describes "a CSV with 30 tickets" but no file was attached. Rather
than block on that, I wrote `data/tickets.csv`: 30 synthetic tickets spanning
all five categories, three customer tiers (free/pro/enterprise), and a mix
of clear-cut and genuinely ambiguous cases (e.g. a "suspicious login" ticket
that isn't quite a bug or a billing issue — deliberately included to stress
the classifier). If you run this against the client's real CSV, just replace
that file; the column names (`ticket_id, subject, body, customer_tier`) are
the only contract `run.py` relies on.

## How it's built

See `CLAUDE.md` for the full repo map and design rationale. Short version:

- **Groq, forced function calling.** The model is called with `tool_choice`
  pinned to a single `submit_triage` function, so output is always
  structured, valid JSON — never parsed out of prose.
- **LangGraph orchestration.** `src/graph.py` wires the pipeline as
  `START -> classify -> safety_net -> (route) -> auto_handle / human_queue
  -> END`. `classify` and `safety_net` are deliberately separate nodes
  (not one function) so the next rule is a visible edge in the graph, not
  logic buried in a comment:
- **The model doesn't get the final say on escalation.** A separate,
  deterministic **safety net** (plain regex + a confidence threshold) runs
  after the model and can override it: refunds, security, data loss,
  SSO/SAML errors, and VAT/compliance mentions are always sent to a human,
  and bugs/billing tickets are never auto-replied to, no matter how
  confident the model says it is. The `human_queue` node also strips any
  draft reply as a last-mile guarantee. The idea: the model is good at
  judgment calls, but "don't guess when it's expensive to be wrong" should
  not depend on the model remembering to apply it.
- **Real human-in-the-loop, not just a label.** `python review.py` runs the
  same graph with one extra node (`human_review`) that calls LangGraph's
  `interrupt()`: execution genuinely pauses on any escalated ticket and asks
  you, in the terminal, to approve it / resolve it yourself / correct a
  wrong label, then resumes exactly where it left off. `run.py` and
  `eval.py` compile the graph without that node on purpose, so a scheduled
  or unattended run never hangs waiting on someone who isn't there - see
  "Running it" below.

## Eval results

`data/eval_labels.csv` contains 10 of the 30 tickets hand-labeled with gold
categories, priorities, and explicit labeler rationales (including the tricky
boundary cases — e.g. labeling a "suspicious login" alert as `other` rather than
`bug` since it is an account-security concern, not a product defect).

Run the evaluation harness:
```bash
python eval.py
```

Results across the gold evaluation set:
```
Category accuracy: 10/10 (100%)
Priority accuracy: 10/10 (100%)
Both correct:      10/10 (100%)
```

The per-ticket breakdown is saved to `outputs/eval_report.txt`.

- **Live Model Mode**: When `GROQ_API_KEY` is provided, the agent uses Groq's
  forced function calling with `openai/gpt-oss-20b` (or `llama-3.3-70b-versatile`).
- **Offline / Fallback Mode**: When run without an API key, the agent uses an
  intelligent deterministic heuristic in `src/agent.py` (explicitly logged with
  `source: "fallback"`), ensuring zero-dependency testability for local development
  and CI/CD pipelines.

## Harness & Agentic Setup

Per the assignment brief, this repo includes a comprehensive harness configuration
for AI-assisted engineering:

- **`CLAUDE.md`**: Complete context and operational manual for Claude Code CLI.
- **`.cursorrules` & `.cursor/rules/ticket-agent.mdc`**: Architectural invariants,
  safety-net enforcement rules, and coding conventions for Cursor IDE.
- **`.mcp.json`**: Model Context Protocol configuration detailing how external
  support tooling (such as SQLite ticket history and filesystem documentation servers)
  connect to the agent harness.

## Cost estimate & Live Measured Metrics

Default model is **`openai/gpt-oss-20b` on Groq** ($0.075/MTok in, $0.30/MTok out as of Nov 2026).
A 5-way classification + short structured output task does not require a frontier model;
the deterministic safety net (not raw model size) is what guarantees "escalate when risky."

### Measured Live Performance on 30 Tickets:
- **Total tickets processed**: 30
- **Routing breakdown**: 11 auto-handled, 19 escalated to a human (63% escalation rate)
- **Average input tokens / ticket**: 711 tokens
- **Average output tokens / ticket**: 263 tokens
- **Measured cost for 30 tickets**: **$0.00396 USD**
- **Measured cost per 100 tickets**: **$0.0132 USD** (~1.3 cents per 100 tickets)

At ~200 tickets/week, monthly API spend is roughly **$0.10** (10 cents a month).
The true cost driver in production is human review time on escalated tickets,
making high-precision routing far more valuable than marginal model cost savings.

## What I'd do with more time

- **Real eval, more of it.** 10 labeled tickets is enough to sanity-check
  the approach, not enough to trust a percentage. I'd grow this to 50-100
  labeled tickets (ideally pulled from the client's real ticket history) and
  track accuracy separately for "easy" vs. "ambiguous" tickets, since those
  likely have very different accuracy ceilings.
- **Learn from escalations.** Right now a ticket that gets force-escalated
  by the safety net doesn't feed back into anything. I'd log which
  model-confident predictions a human later overturns, and use that to
  tighten (or loosen) the confidence threshold over time instead of it being
  a guessed constant.
- **Push the human-in-the-loop graph further.** `review.py` now pauses for
  real on each escalation (see "Human-in-the-loop review" below), but the
  decision options are still basic (approve / resolve / relabel). I'd add:
  a persistent checkpointer (SQLite/Postgres instead of `InMemorySaver`) so
  a paused review can survive a process restart and be picked up by a
  different reviewer later; a conditional second model call — e.g. a
  stronger model re-checking any ticket that lands right at the 0.75
  confidence threshold before it ever reaches a person; and surfacing the
  pending-review queue as something other than a terminal prompt (a tiny
  web UI, or a Slack message per escalation) for a non-technical reviewer.
- **Thread/reply handling.** This assumes one ticket = one message. Real
  support tickets have back-and-forth; I'd want the agent to see prior
  messages in a thread before triaging a reply.
- **Draft-reply quality control.** Right now `can_auto_reply` is a single
  model-decided boolean. I'd want a second, cheaper pass (or a short
  allow-list of FAQ-style answers) to sanity-check a draft before it's
  actually sent, rather than trusting one model call end-to-end even for the
  "simple" bucket.
- **Batch processing.** For ~200/week this doesn't matter, but at higher
  volume Groq's batch pricing would be meaningfully cheaper for
  non-latency-sensitive classification work like this.
- **Structured categories from real data.** `bug / billing / feature_request
  / question / other` came straight from the brief; a real client's ticket
  taxonomy is usually messier (and often has sub-categories per team), so
  I'd expect this to need a revision pass once the real CSV is available.

## Human-in-the-loop review

```bash
python review.py
```

This runs the exact same graph as `run.py`, with one extra node turned on:
`human_review`, which calls LangGraph's `interrupt()`. Whenever the safety
net sends a ticket to the human queue, the graph **actually pauses** right
there (state is checkpointed per-ticket) and asks you in the terminal:

```
Ticket T005 needs human review
...
What do you want to do?
  [a] approve  - keep it in the human queue as-is (default)
  [r] resolve  - answer it yourself right now, mark it handled
  [c] correct  - fix category/priority, keep it queued
```

Answer, and the graph resumes from exactly that point — it's not re-running
anything, it's a genuine pause/resume via `Command(resume=...)`. Tickets the
agent is confident about fly through with no prompt at all, same as
`run.py`. `run.py`/`eval.py` deliberately compile the graph *without* this
node, so a scheduled or unattended batch run can never hang waiting on a
person who isn't there — the interactive node is opt-in, not the default.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in GROQ_API_KEY (from console.groq.com), or export it directly

# Run automated unit tests
python -m unittest discover tests

# Process all 30 tickets in batch unattended mode
python run.py     # outputs -> logs/decisions.jsonl, outputs/triage_results.csv

# Score accuracy against the gold eval set
python eval.py    # outputs -> outputs/eval_report.txt

# Interactive human-in-the-loop review mode
python review.py  # pauses execution on each escalated ticket for human decision
```

Without an API key, all three scripts still run end-to-end using the
offline heuristic fallback in `src/agent.py` (clearly marked
`source: "fallback"` in the output) — useful for checking the plumbing
works, not a substitute for the real eval numbers above.
