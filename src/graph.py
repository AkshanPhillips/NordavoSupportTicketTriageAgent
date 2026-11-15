"""
LangGraph orchestration for the ticket agent.

Why a graph instead of just calling agent.triage()? Two reasons:

1. It makes the "the model doesn't get the final say" design explicit and
   inspectable as a diagram, not just a comment in agent.py: `classify` and
   `safety_net` are separate nodes with a real edge between them, and the
   routing after `safety_net` is a conditional edge you can point at,
   log, or unit-test independently of the model call.
2. It's the natural place to add real human-in-the-loop review (see below)
   without restructuring run.py/eval.py.

Graph shape (human_in_the_loop=False, used by run.py/eval.py — unattended
batch mode, never blocks on a person):

    START -> classify -> safety_net -> (route) -> auto_handle   -> END
                                     (or)      -> human_queue    -> END

Graph shape (human_in_the_loop=True, used by review.py — a person is
actually at the keyboard):

    START -> classify -> safety_net -> (route) -> auto_handle   -> END
                                     (or)      -> human_queue
                                                -> human_review  -> END

`classify`      : calls TicketAgent.classify_raw() (Groq, or the offline
                  fallback heuristic if no API key is set)
`safety_net`    : calls agent.apply_safety_net() - deterministic overrides
                  for low confidence / risky keywords / bug+billing auto-reply
`route`         : conditional edge - reads result["needs_human"]
`auto_handle`   : no-op pass-through node (the draft_reply already decided
                  by safety_net stands) - kept as its own node so it's a
                  visible branch in the graph, and a natural place to add a
                  draft-quality check later (see README)
`human_queue`   : clears any draft_reply as a last-mile guarantee that
                  nothing escalated can carry an auto-sendable reply,
                  whatever upstream did
`human_review`  : calls LangGraph's interrupt() - this genuinely PAUSES the
                  graph (checkpointed, so it survives the process being
                  idle) until review.py resumes it with the reviewer's
                  decision: approve the escalation as-is, resolve it
                  themselves with their own reply, or correct a wrong
                  category/priority before it reaches the queue.
"""

from typing import Literal, Optional, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

from agent import TicketAgent, apply_safety_net
from schema import TriageResult


class TicketState(TypedDict):
    ticket_id: str
    subject: str
    body: str
    tier: str
    result: Optional[dict]


def _append_note(existing: Optional[str], note: str) -> str:
    return f"{existing} | {note}" if existing else note


def build_graph(agent: TicketAgent, human_in_the_loop: bool = False, checkpointer=None):
    """Compile the triage graph.

    human_in_the_loop=False (default): matches the original behavior used by
    run.py/eval.py - escalated tickets land in `human_queue` and the graph
    finishes. No checkpointer, nothing can pause.

    human_in_the_loop=True: adds a `human_review` node after `human_queue`
    that calls interrupt() and genuinely pauses the graph for a real person
    (see review.py). Requires a checkpointer so the paused run can be
    resumed later - one is created (InMemorySaver) if you don't pass one.
    """

    def classify(state: TicketState) -> dict:
        result = agent.classify_raw(
            state["ticket_id"], state["subject"], state["body"], state["tier"]
        )
        return {"result": result.to_dict()}

    def safety_net(state: TicketState) -> dict:
        result = TriageResult(**state["result"])
        result = apply_safety_net(result, state["subject"], state["body"])
        return {"result": result.to_dict()}

    def route(state: TicketState) -> Literal["human_queue", "auto_handle"]:
        return "human_queue" if state["result"]["needs_human"] else "auto_handle"

    def auto_handle(state: TicketState) -> dict:
        return {}

    def human_queue(state: TicketState) -> dict:
        result = dict(state["result"])
        result["draft_reply"] = None
        return {"result": result}

    def human_review(state: TicketState) -> dict:
        # This call PAUSES the graph (see review.py for who resumes it, and
        # with what). The dict passed in is exactly what the reviewer sees;
        # whatever dict they respond with comes back as `decision` below.
        decision = interrupt(
            {
                "ticket_id": state["ticket_id"],
                "subject": state["subject"],
                "body": state["body"],
                "tier": state["tier"],
                "proposed": state["result"],
            }
        )
        decision = decision or {}
        result = dict(state["result"])
        action = decision.get("action", "approve")

        if action == "resolve":
            # Reviewer answered it themselves right now - no longer needs
            # to sit in the queue.
            result["needs_human"] = False
            result["can_auto_reply"] = True
            result["draft_reply"] = decision.get("draft_reply") or result.get("draft_reply")
            result["escalation_reason"] = _append_note(
                result.get("escalation_reason"), "Resolved by human reviewer."
            )
        elif action == "relabel":
            if decision.get("category"):
                result["category"] = decision["category"]
            if decision.get("priority"):
                result["priority"] = decision["priority"]
            result["escalation_reason"] = _append_note(
                result.get("escalation_reason"), "Re-labeled by human reviewer."
            )
        else:  # "approve" - keep it in the human queue as-is
            result["escalation_reason"] = _append_note(
                result.get("escalation_reason"), "Confirmed by human reviewer."
            )

        return {"result": result}

    builder = StateGraph(TicketState)
    builder.add_node("classify", classify)
    builder.add_node("safety_net", safety_net)
    builder.add_node("auto_handle", auto_handle)
    builder.add_node("human_queue", human_queue)
    builder.add_edge(START, "classify")
    builder.add_edge("classify", "safety_net")
    builder.add_conditional_edges(
        "safety_net", route, {"human_queue": "human_queue", "auto_handle": "auto_handle"}
    )
    builder.add_edge("auto_handle", END)

    if human_in_the_loop:
        builder.add_node("human_review", human_review)
        builder.add_edge("human_queue", "human_review")
        builder.add_edge("human_review", END)
        return builder.compile(checkpointer=checkpointer or InMemorySaver())

    builder.add_edge("human_queue", END)
    return builder.compile()


def run_ticket(graph, ticket_id: str, subject: str, body: str, tier: str) -> TriageResult:
    """Run one ticket through a non-interactive (human_in_the_loop=False)
    compiled graph and return a TriageResult. Used by run.py / eval.py."""
    final_state = graph.invoke(
        {"ticket_id": ticket_id, "subject": subject, "body": body, "tier": tier, "result": None}
    )
    return TriageResult(**final_state["result"])


def start_review(graph, config: dict, ticket_id: str, subject: str, body: str, tier: str):
    """Run one ticket through an interactive (human_in_the_loop=True)
    compiled graph. Returns a TriageResult if the ticket didn't need review
    (finished in one pass), or a dict (the interrupt payload - see
    human_review() above) if the graph paused and is waiting on
    resume_review(). `config` must carry a unique thread_id per ticket, e.g.
    {"configurable": {"thread_id": ticket_id}}."""
    state = graph.invoke(
        {"ticket_id": ticket_id, "subject": subject, "body": body, "tier": tier, "result": None},
        config,
    )
    if "__interrupt__" in state:
        return state["__interrupt__"][0].value
    return TriageResult(**state["result"])


def resume_review(graph, config: dict, decision: dict) -> TriageResult:
    """Resume a paused interactive graph with the reviewer's decision (see
    human_review() above for the accepted shapes) and return the final
    TriageResult. `config` must be the SAME dict (same thread_id) passed to
    the start_review() call that paused."""
    state = graph.invoke(Command(resume=decision), config)
    return TriageResult(**state["result"])
