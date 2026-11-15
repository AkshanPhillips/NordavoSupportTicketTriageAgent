#!/usr/bin/env python3
"""
Eval harness: runs the LangGraph agent on the 10 hand-labeled tickets in
data/eval_labels.csv and reports category/priority accuracy against the
gold labels.

Usage:
    export GROQ_API_KEY=gsk_...   # optional - omit to eval fallback mode
    python eval.py
"""

import csv
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agent import TicketAgent  # noqa: E402
from graph import build_graph, run_ticket  # noqa: E402

TICKETS_PATH = os.path.join(os.path.dirname(__file__), "data", "tickets.csv")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "data", "eval_labels.csv")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "outputs", "eval_report.txt")


def load_tickets():
    with open(TICKETS_PATH, newline="", encoding="utf-8") as f:
        return {row["ticket_id"]: row for row in csv.DictReader(f)}


def load_labels():
    with open(LABELS_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    tickets = load_tickets()
    labels = load_labels()
    agent = TicketAgent()
    graph = build_graph(agent)
    mode = f"LIVE (Groq, model={agent.model})" if agent.using_model else "FALLBACK (heuristic)"

    rows = []
    cat_correct = 0
    pri_correct = 0
    both_correct = 0

    for row in labels:
        tid = row["ticket_id"]
        t = tickets[tid]
        pred = run_ticket(graph, tid, t["subject"], t["body"], t["customer_tier"])

        cat_ok = pred.category == row["gold_category"]
        pri_ok = pred.priority == row["gold_priority"]
        cat_correct += cat_ok
        pri_correct += pri_ok
        both_correct += cat_ok and pri_ok

        rows.append(
            {
                "ticket_id": tid,
                "gold_category": row["gold_category"],
                "pred_category": pred.category,
                "cat_match": "OK" if cat_ok else "MISS",
                "gold_priority": row["gold_priority"],
                "pred_priority": pred.priority,
                "pri_match": "OK" if pri_ok else "MISS",
                "confidence": f"{pred.confidence:.2f}",
                "needs_human": pred.needs_human,
            }
        )

    n = len(labels)
    lines = []
    lines.append(f"Eval mode: {mode}")
    lines.append(f"Eval set size: {n} tickets\n")
    header = f"{'ticket':8s}{'gold_cat':18s}{'pred_cat':18s}{'cat':6s}{'gold_pri':10s}{'pred_pri':10s}{'pri':6s}{'conf':6s}{'human?'}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        lines.append(
            f"{r['ticket_id']:8s}{r['gold_category']:18s}{r['pred_category']:18s}"
            f"{r['cat_match']:6s}{r['gold_priority']:10s}{r['pred_priority']:10s}"
            f"{r['pri_match']:6s}{r['confidence']:6s}{str(r['needs_human'])}"
        )

    lines.append("")
    lines.append(f"Category accuracy: {cat_correct}/{n} ({cat_correct/n:.0%})")
    lines.append(f"Priority accuracy: {pri_correct}/{n} ({pri_correct/n:.0%})")
    lines.append(f"Both correct:      {both_correct}/{n} ({both_correct/n:.0%})")

    report = "\n".join(lines)
    print(report)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
