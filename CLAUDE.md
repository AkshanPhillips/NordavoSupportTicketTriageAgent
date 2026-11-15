# CLAUDE.md — harness notes for this repo

This file is for an AI coding assistant (Claude Code / Cursor) picking this
repo up, and for a human reviewer who wants to know how it was built.

## What this repo is

A support-ticket triage agent for the Nordavo take-home assignment. It reads
`data/tickets.csv`, and for each ticket produces: category, priority +
reason, an optional draft reply, and an escalate-to-human decision — logged
to `logs/decisions.jsonl`. The LLM backend is Groq (fast, cheap open-weight
models); orchestration is a small LangGraph graph (see `src/graph.py`).

## How it was built (process, for transparency)

Built in a single focused session using an agentic coding assistant (Claude)
driving the file/shell tools directly: data generation, the agent module,
the graph, the safety-net rules, the eval harness and this README were all
written, then actually executed locally (not just described) to confirm the
pipeline runs end-to-end and to get real numbers for the eval report and
cost estimate below — see `outputs/eval_report.txt` for the exact run.

## Repo map

```
data/tickets.csv        30 synthetic support tickets (no CSV was provided
                         with the brief, so this was generated to match the
                         brief's description — see README "Assumptions")
data/eval_labels.csv    10 of those tickets, hand-labeled with gold
                         category/priority + a one-line rationale per label
src/schema.py           the structured-output contract: an OpenAI-style
                         function/tool schema (Groq's chat.completions API
                         is OpenAI-compatible for tool calling) + the
                         TriageResult dataclass
src/agent.py            TicketAgent: classify_raw() calls Groq with forced
                         function calling (tool_choice pinned to
                         submit_triage); apply_safety_net() is a SEPARATE
                         deterministic step that can override the model
                         when confidence is low or the ticket matches a
                         hard-coded risk pattern (refunds, security, data
                         loss, SSO/SAML, compliance); heuristic_classify()
                         is an offline fallback used only when no API key is
                         present
src/graph.py             LangGraph wiring: START -> classify -> safety_net
                         -> (conditional route on needs_human) ->
                         auto_handle / human_queue -> END. classify_raw()
                         and apply_safety_net() from agent.py are run as two
                         distinct nodes on purpose (see docstring in
                         graph.py) rather than one call, so the "model
                         doesn't get the final say" rule is a visible edge
                         in the graph, not logic buried in a function.
                         build_graph(..., human_in_the_loop=True) adds a
                         human_review node that calls LangGraph's
                         interrupt() and genuinely pauses the graph for a
                         real person - see review.py.
src/cost.py             per-million-token Groq pricing + cost-per-100-
                         tickets math
run.py                  unattended batch mode: builds the graph (no human
                         review node), runs all 30 tickets through it,
                         writes logs + results CSV, prints a cost report.
                         Never pauses - safe to run with nobody watching.
eval.py                 runs the 10-ticket eval set through the same
                         non-interactive graph and scores category /
                         priority accuracy against data/eval_labels.csv
review.py               interactive mode: builds the graph WITH the
                         human_review node + a checkpointer. Any ticket the
                         safety net escalates genuinely pauses the graph
                         (LangGraph interrupt/Command.resume) and prompts
                         you in the terminal to approve it, resolve it
                         yourself, or correct a wrong label - then resumes.
                         Tickets that don't need a human fly through with no
                         prompt, same as run.py.
outputs/                generated: triage_results.csv, eval_report.txt,
                         triage_results_reviewed.csv (from review.py)
logs/                   generated: decisions.jsonl, decisions_reviewed.jsonl
tests/                  automated unit test suite (schema, safety_net, graph, cost)
.cursorrules            Cursor IDE system prompt & architecture standards
.cursor/rules/          modular Cursor rules (.mdc)
.mcp.json               MCP (Model Context Protocol) config for ticket DB & docs
.env.example            environment variables template
```

## Running it

```bash
pip install -r requirements.txt

# Run automated tests
python -m unittest discover tests

# Recommended: run against the real model (https://console.groq.com/playground)
export GROQ_API_KEY=gsk_...
python run.py        # processes all 30 tickets
python eval.py        # scores the 10-ticket eval set

# Works without a key too (keyword heuristic instead of Groq - for
# sanity-checking the plumbing only, not a stand-in for eval accuracy)
python run.py
python eval.py

# Interactive human-in-the-loop mode: a real person reviews each escalation
# live, in the terminal, before the graph resumes
python review.py
```

## Design choices worth knowing about

- **Structured output via forced function calling**, not prompting for JSON
  and regex-parsing it. `tool_choice={"type":"function","function":
  {"name":"submit_triage"}}` guarantees a parseable, schema-valid response
  every time.
- **The model doesn't get the final say on escalation.** `apply_safety_net()`
  in `src/agent.py`, run as its own LangGraph node, force-escalates on a
  low-confidence score *or* a keyword hit (refund, data loss, security,
  SAML/SSO, VAT, compliance), and never lets `bug`/`billing` tickets get an
  automated reply even if the model thought it could draft one. It also
  guarantees no `draft_reply` survives into the `human_queue` branch,
  regardless of what came out of the model. The idea: a wrong guess on money
  or security is expensive; a ticket that waits an extra hour for a human is
  not.
- **Why a graph for a 30-ticket batch job?** Overkill on its own, honestly —
  a plain function call would do for this exact assignment. It's here
  because (a) it makes the "model proposes, safety-net disposes" rule a
  structural edge you can point at instead of a comment, and (b) it's the
  natural place to grow this later: a human-in-the-loop interrupt before
  `human_queue`, a second "double-check" model call on borderline-confidence
  tickets, retries with backoff as their own node, etc. — all additive to
  the graph, not a rewrite of run.py.
- **Confidence threshold is a named constant** (`CONFIDENCE_ESCALATION_THRESHOLD
  = 0.75` in `src/agent.py`), not buried in logic, so it's one line to tune.
- **Model choice:** defaults to `openai/gpt-oss-20b` on Groq. This is a
  5-way classification + short-reasoning task on short inputs — it doesn't
  need a frontier model's reasoning depth, and the safety net (not the
  model) is what's actually carrying the "don't guess on risky stuff"
  requirement. Swap the `GROQ_MODEL` env var to `openai/gpt-oss-120b` for a
  stronger model on ambiguous tickets if real-world accuracy calls for it.
- **Every decision is logged** to `logs/decisions.jsonl` with the full
  reasoning (`priority_reason`, `escalation_reason`), not just the labels,
  so a human reviewing the queue (or debugging a bad call) can see *why*.
- **Human-in-the-loop is a real pause, not a metaphor.** `review.py` uses
  LangGraph's `interrupt()` + a checkpointer (`InMemorySaver`, one thread
  per ticket) so the graph execution itself stops and waits for a decision
  object, then resumes from exactly where it left off via
  `Command(resume=...)`. `run.py`/`eval.py` compile the graph *without*
  that node (`human_in_the_loop=False`, the default) specifically so
  unattended/scheduled runs never block on a person who isn't there.
