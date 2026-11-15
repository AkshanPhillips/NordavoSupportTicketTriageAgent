#!/usr/bin/env python3
"""
Entry point: process every ticket in data/tickets.csv through the LangGraph
agent (classify -> safety_net -> route -> auto_handle/human_queue), write a
full log of each decision, and print a cost summary.

Usage:
    export GROQ_API_KEY=gsk_...   # optional - omit to run in fallback/demo mode
    python run.py
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

from agent import TicketAgent  # noqa: E402
from graph import build_graph, run_ticket  # noqa: E402
from cost import CostSummary  # noqa: E402

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "tickets.csv")
LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "decisions.jsonl")
OUT_PATH = os.path.join(os.path.dirname(__file__), "outputs", "triage_results.csv")


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    agent = TicketAgent()
    graph = build_graph(agent)
    mode = "LIVE (calling Groq API)" if agent.using_model else "FALLBACK (no API key - heuristic demo mode)"
    print(f"Running in {mode} | model={agent.model}\n")

    results = []
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        tickets = list(reader)

    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        for row in tickets:
            result = run_ticket(
                graph,
                ticket_id=row["ticket_id"],
                subject=row["subject"],
                body=row["body"],
                tier=row["customer_tier"],
            )
            results.append(result)

            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **result.to_dict(),
            }
            log_file.write(json.dumps(log_entry) + "\n")

            flag = "-> HUMAN" if result.needs_human else "-> auto"
            print(
                f"{result.ticket_id:6s} {result.category:16s} {result.priority:7s} "
                f"conf={result.confidence:.2f} {flag:10s} {result.priority_reason}"
            )

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
    n_auto = len(results) - n_human
    print(f"\n{len(results)} tickets processed: {n_auto} auto-handled, {n_human} sent to a human.")
    print(f"Full log:    {LOG_PATH}")
    print(f"Results CSV: {OUT_PATH}")

    if agent.using_model:
        cost = CostSummary(
            model=agent.model,
            n_tickets=len(results),
            total_input_tokens=sum(r.input_tokens for r in results),
            total_output_tokens=sum(r.output_tokens for r in results),
        )
        print("\n--- Cost ---")
        print(cost.report())
    else:
        print(
            "\n(No cost report - running in fallback mode without API calls. "
            "See README / src/cost.py for the projected per-100-ticket cost.)"
        )


if __name__ == "__main__":
    main()
