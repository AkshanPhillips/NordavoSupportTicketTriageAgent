#!/usr/bin/env python3
"""
Entry point: Fetch open support tickets from the Roosterly Helpdesk API,
triage each ticket with the agent, post the decision back to the helpdesk,
and record redacted audit logs.

Usage:
    python run.py             # Process all open tickets
    python run.py --reset     # Reset practice tickets first, then process all 30
    python run.py --status    # Check helpdesk open/handled ticket counts
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agent import TicketAgent, redact_payment_details
from api_client import HelpdeskClient, HelpdeskAPIError
from cost import CostSummary

LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "decisions.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Roosterly Support Ticket Triage Runner")
    parser.add_argument("--reset", action="store_true", help="Reset all practice tickets before running")
    parser.add_argument("--status", action="store_true", help="Print current helpdesk status and exit")
    args = parser.parse_args()

    client = HelpdeskClient()
    agent = TicketAgent()

    if args.status:
        status = client.get_status()
        print(f"\nHelpdesk Status for {status.get('candidate', client.candidate_email)}:")
        print(f"- Mode:            {status.get('mode')}")
        print(f"- Open Tickets:    {status.get('open_tickets')}")
        print(f"- Handled Tickets: {status.get('handled_tickets')}\n")
        return

    if args.reset:
        print("Calling POST /practice/reset to re-open all 30 practice tickets...")
        client.reset_practice()
        print("Reset complete.\n")

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    mode_desc = f"LIVE (Groq: {agent.model})" if agent.client else "FALLBACK (Deterministic Heuristic)"
    print(f"=================================================================")
    print(f"Roosterly Support Agent | {mode_desc}")
    print(f"Candidate: {client.candidate_email}")
    print(f"Helpdesk:  {client.base_url}")
    print(f"=================================================================\n")

    print("Fetching open tickets from Helpdesk API...")
    open_tickets = client.get_all_open_tickets()
    print(f"Found {len(open_tickets)} open tickets to triage.\n")

    if not open_tickets:
        print("No open tickets found. If you want to run practice tickets, use `python run.py --reset`.")
        return

    handled_count = {"reply": 0, "handoff": 0, "close": 0}
    total_input_tokens = 0
    total_output_tokens = 0

    header = f"{'ID':6s} {'Lang':5s} {'Cat':16s} {'Pri':5s} {'Action':8s} {'Reason'}"
    print(header)
    print("-" * 80)

    # Open log file in append mode so re-runs are crash-safe and additive
    with open(LOG_PATH, "a", encoding="utf-8") as log_file:
        for ticket in open_tickets:
            tid = ticket["id"]
            subject = ticket.get("subject", "")
            body = ticket.get("body", "")
            plan = ticket.get("customer_plan", "Basis")
            channel = ticket.get("channel", "email")

            # 1. Triage with model & deterministic safety net
            decision = agent.triage(
                ticket_id=tid,
                subject=subject,
                body=body,
                customer_plan=plan,
                channel=channel,
            )

            total_input_tokens += decision.input_tokens
            total_output_tokens += decision.output_tokens

            # 2. Post decision to Helpdesk API
            try:
                if decision.action == "reply":
                    client.post_reply(
                        ticket_id=tid,
                        language=decision.language,
                        category=decision.category,
                        priority=decision.priority,
                        reason=decision.reason,
                        draft_reply=decision.draft_reply or "",
                        confidence=decision.confidence,
                    )
                elif decision.action == "handoff":
                    client.post_handoff(
                        ticket_id=tid,
                        language=decision.language,
                        category=decision.category,
                        priority=decision.priority,
                        reason=decision.reason,
                        handoff_note=decision.handoff_note or "",
                        confidence=decision.confidence,
                    )
                elif decision.action == "close":
                    client.post_close(
                        ticket_id=tid,
                        language=decision.language,
                        category=decision.category,
                        priority=decision.priority,
                        reason=decision.reason,
                        confidence=decision.confidence,
                    )

                handled_count[decision.action] = handled_count.get(decision.action, 0) + 1

            except HelpdeskAPIError as err:
                if err.status_code == 409:
                    print(f"[{tid}] Already handled (409) - skipping duplicate post.")
                else:
                    print(f"[{tid}] API Error: {err}")
                    continue

            # 3. Log decision with payment details strictly redacted
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ticket_id": tid,
                "language": decision.language,
                "category": decision.category,
                "priority": decision.priority,
                "action": decision.action,
                "reason": decision.reason,
                "draft_reply": redact_payment_details(decision.draft_reply) if decision.draft_reply else None,
                "handoff_note": redact_payment_details(decision.handoff_note) if decision.handoff_note else None,
                "confidence": decision.confidence,
                "source": decision.source,
            }
            log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            log_file.flush()

            # Terminal progress print
            reason_snip = decision.reason[:35] + "..." if len(decision.reason) > 35 else decision.reason
            print(f"{tid:6s} {decision.language:5s} {decision.category:16s} {decision.priority:5s} {decision.action:8s} {reason_snip}")

    # Summary
    print("-" * 80)
    print(f"\nRun Completed Successfully:")
    print(f"- Total Tickets Processed: {len(open_tickets)}")
    print(f"- Direct Replies Sent:     {handled_count.get('reply', 0)}")
    print(f"- Human Handoffs:          {handled_count.get('handoff', 0)}")
    print(f"- Closed (Spam):           {handled_count.get('close', 0)}")
    print(f"- Audit Log:               {LOG_PATH}")

    if agent.client and len(open_tickets) > 0:
        cost = CostSummary(
            model=agent.model,
            n_tickets=len(open_tickets),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
        )
        print("\n--- Model Token Cost Summary ---")
        print(cost.report())


if __name__ == "__main__":
    main()
