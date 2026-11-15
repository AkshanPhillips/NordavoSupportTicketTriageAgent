#!/usr/bin/env python3
"""
Interactive human-in-the-loop review session.

Runs every ticket through the SAME LangGraph pipeline as run.py, but with
the human_review node turned on: whenever safety_net sends a ticket to the
human queue, the graph genuinely PAUSES (LangGraph's interrupt(), backed by
a checkpointer) right here, asks you what to do about it, and only resumes
once you answer - one thread per ticket, so you could in principle walk
away and resume a specific ticket later from another process using the same
thread_id.

This is what `python run.py` deliberately does NOT do, so unattended batch
runs never hang waiting on a person. Use this script instead when a human
is actually available and you want them making the call on each escalation
as part of the same run, rather than reading a CSV afterwards.

Usage:
    export GROQ_API_KEY=gsk_...   # optional - omit to use fallback mode
    python review.py
"""

import csv
import json
import os
import sys
from datetime import datetime, timezone

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from agent import TicketAgent  # noqa: E402
from graph import build_graph, start_review, resume_review  # noqa: E402

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "tickets.csv")
LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "decisions_reviewed.jsonl")
OUT_PATH = os.path.join(os.path.dirname(__file__), "outputs", "triage_results_reviewed.csv")


def ask_reviewer(payload: dict) -> dict:
    p = payload["proposed"]
    print("\n" + "=" * 72)
    print(f"Ticket {payload['ticket_id']}  (tier: {payload['tier']}) needs human review")
    print(f"Subject: {payload['subject']}")
    print(f"Body:    {payload['body']}")
    print(
        f"\nAgent proposed: category={p['category']}  priority={p['priority']}  "
        f"confidence={p['confidence']:.2f}"
    )
    print(f"Priority reason:   {p['priority_reason']}")
    print(f"Escalation reason: {p['escalation_reason']}")
    print("\nWhat do you want to do?")
    print("  [a] approve  - keep it in the human queue as-is (default)")
    print("  [r] resolve  - answer it yourself right now, mark it handled")
    print("  [c] correct  - fix category/priority, keep it queued")
    choice = input("> ").strip().lower() or "a"

    if choice == "r":
        reply = input("Type the reply to send: ").strip()
        return {"action": "resolve", "draft_reply": reply}
    if choice == "c":
        category = input(f"Correct category (enter to keep '{p['category']}'): ").strip() or None
        priority = input(f"Correct priority (enter to keep '{p['priority']}'): ").strip() or None
        return {"action": "relabel", "category": category, "priority": priority}
    return {"action": "approve"}


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    agent = TicketAgent()
    graph = build_graph(agent, human_in_the_loop=True, checkpointer=InMemorySaver())
    mode = "LIVE (Groq)" if agent.using_model else "FALLBACK (heuristic)"
    print(f"Interactive review session | {mode} | model={agent.model}")
    print("Tickets that don't need a human will fly through with no prompt.\n")

    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        tickets = list(csv.DictReader(f))

    results = []
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        for row in tickets:
            config = {"configurable": {"thread_id": row["ticket_id"]}}
            outcome = start_review(
                graph, config, row["ticket_id"], row["subject"], row["body"], row["customer_tier"]
            )

            if isinstance(outcome, dict):  # paused - the graph wants a decision
                decision = ask_reviewer(outcome)
                result = resume_review(graph, config, decision)
            else:  # already finished - didn't need a human
                result = outcome

            results.append(result)
            log_file.write(
                json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), **result.to_dict()})
                + "\n"
            )
            flag = "-> HUMAN" if result.needs_human else "-> auto"
            print(f"{result.ticket_id:6s} {result.category:16s} {result.priority:7s} {flag}")

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ticket_id", "category", "priority", "priority_reason",
                "confidence", "can_auto_reply", "draft_reply", "needs_human",
                "escalation_reason", "source",
            ],
        )
        writer.writeheader()
        for r in results:
            d = r.to_dict()
            d.pop("input_tokens", None)
            d.pop("output_tokens", None)
            writer.writerow(d)

    n_human = sum(1 for r in results if r.needs_human)
    print(
        f"\n{len(results)} tickets processed: {len(results) - n_human} auto-handled "
        f"(including any you resolved live), {n_human} still in the human queue after review."
    )
    print(f"Log:     {LOG_PATH}")
    print(f"Results: {OUT_PATH}")


if __name__ == "__main__":
    main()
