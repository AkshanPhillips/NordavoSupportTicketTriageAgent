#!/usr/bin/env python3
"""
Evaluation Harness for Roosterly Support Ticket Agent.

Scores the agent against the 10 ground-truth examples provided by Nordavo (GET /examples).
Measures:
- Action accuracy (reply / handoff / close)
- Priority accuracy (P1 / P2 / P3 / P4)
- Category accuracy (bug / billing / feature_request / question / other)
- Language accuracy (nl / en / de)
- Safety violation count (must be 0 - never auto-reply to handoff-mandated tickets)
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agent import TicketAgent
from api_client import HelpdeskClient

EXAMPLES_PATH = os.path.join(os.path.dirname(__file__), "data", "examples.json")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "outputs", "eval_report.txt")


def load_examples() -> List[Dict[str, Any]]:
    """Load examples from local cache, or fetch from API if missing."""
    if os.path.exists(EXAMPLES_PATH):
        with open(EXAMPLES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("examples", data)
    
    # Fetch from API
    try:
        client = HelpdeskClient()
        examples = client.get_examples()
        os.makedirs(os.path.dirname(EXAMPLES_PATH), exist_ok=True)
        with open(EXAMPLES_PATH, "w", encoding="utf-8") as f:
            json.dump({"examples": examples}, f, indent=2)
        return examples
    except Exception as exc:
        print(f"Failed to fetch examples from API ({exc}). Please ensure data/examples.json exists.")
        sys.exit(1)


def run_evaluation() -> Dict[str, Any]:
    examples = load_examples()
    agent = TicketAgent()

    rows = []
    action_correct = 0
    cat_correct = 0
    pri_correct = 0
    lang_correct = 0
    all_correct = 0
    safety_violations = 0

    mode = f"LIVE (Groq: {agent.model})" if agent.client else "FALLBACK (Deterministic Heuristic)"
    print(f"\n=======================================================")
    print(f"Running Evaluation on {len(examples)} Gold-Standard Examples")
    print(f"Mode: {mode}")
    print(f"=======================================================\n")

    for ex in examples:
        print(f"Evaluating {ex['id']}: {ex['subject']}...", flush=True)
        gold = ex["label"]
        pred = agent.triage(
            ticket_id=ex["id"],
            subject=ex["subject"],
            body=ex["body"],
            customer_plan=ex.get("customer_plan", "Basis"),
            channel=ex.get("channel", "email"),
        )
        print(f"  -> Action: {pred.action} (Gold: {gold['action']}), Pri: {pred.priority} (Gold: {gold['priority']})", flush=True)

        act_ok = pred.action == gold["action"]
        cat_ok = pred.category == gold["category"]
        pri_ok = pred.priority == gold["priority"]
        lang_ok = pred.language == gold["language"]
        perfect = act_ok and cat_ok and pri_ok and lang_ok

        # Safety Check: Did the agent auto-reply when gold action is handoff?
        if gold["action"] == "handoff" and pred.action == "reply":
            safety_violations += 1

        action_correct += act_ok
        cat_correct += cat_ok
        pri_correct += pri_ok
        lang_correct += lang_ok
        all_correct += perfect

        rows.append({
            "id": ex["id"],
            "subject": ex["subject"][:22],
            "gold_act": gold["action"],
            "pred_act": pred.action,
            "act_match": "OK" if act_ok else "MISS",
            "gold_pri": gold["priority"],
            "pred_pri": pred.priority,
            "pri_match": "OK" if pri_ok else "MISS",
            "gold_cat": gold["category"][:7],
            "pred_cat": pred.category[:7],
            "cat_match": "OK" if cat_ok else "MISS",
            "gold_lang": gold["language"],
            "pred_lang": pred.language,
        })

    n = len(examples)
    lines = []
    lines.append(f"Eval Mode: {mode}")
    lines.append(f"Examples Evaluated: {n}\n")
    header = f"{'ID':4s} {'Subject':22s} {'Gold Act':9s} {'Pred Act':9s} {'Act':4s} {'Gold Pri':8s} {'Pred Pri':8s} {'Pri':4s} {'Gold Cat':8s} {'Pred Cat':8s} {'Cat':4s} {'Lang':4s}"
    lines.append(header)
    lines.append("-" * len(header))

    for r in rows:
        lines.append(
            f"{r['id']:4s} {r['subject']:22s} {r['gold_act']:9s} {r['pred_act']:9s} {r['act_match']:4s} "
            f"{r['gold_pri']:8s} {r['pred_pri']:8s} {r['pri_match']:4s} "
            f"{r['gold_cat']:8s} {r['pred_cat']:8s} {r['cat_match']:4s} "
            f"{'OK' if r['gold_lang'] == r['pred_lang'] else 'MISS':4s}"
        )

    lines.append("\nSummary Metrics:")
    lines.append(f"- Action Accuracy:   {action_correct}/{n} ({action_correct/n:.0%})")
    lines.append(f"- Priority Accuracy: {pri_correct}/{n} ({pri_correct/n:.0%})")
    lines.append(f"- Category Accuracy: {cat_correct}/{n} ({cat_correct/n:.0%})")
    lines.append(f"- Language Accuracy: {lang_correct}/{n} ({lang_correct/n:.0%})")
    lines.append(f"- Perfect Match:     {all_correct}/{n} ({all_correct/n:.0%})")
    lines.append(f"- Safety Violations: {safety_violations} (Zero-tolerance violations)")

    report = "\n".join(lines)
    print(report)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nReport saved to {REPORT_PATH}")

    return {
        "action_accuracy": action_correct / n,
        "priority_accuracy": pri_correct / n,
        "category_accuracy": cat_correct / n,
        "language_accuracy": lang_correct / n,
        "all_accuracy": all_correct / n,
        "safety_violations": safety_violations,
    }


if __name__ == "__main__":
    run_evaluation()
