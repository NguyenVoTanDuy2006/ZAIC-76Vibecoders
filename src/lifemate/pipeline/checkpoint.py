"""Bounded terminal-candidate checks using the existing BTC call/final grammar.

No plan, approve or recovery JSON, recursive checks, or application execution.
The result remains a model opinion, not proof of semantic task completion.
"""
from __future__ import annotations

from copy import deepcopy

VERSION = "terminal-call-check-v1"


def review(session, candidate):
    from . import actor, protocol
    state = session.call_state
    if state.terminal_checks >= session.config.max_terminal_checks:
        session.status = "checkpoint_limit"
        event = {"type": "terminal_check_limit", "candidate": deepcopy(candidate),
                 "executed": False, "status": session.status}
        session.trace.append(event)
        return event
    state.terminal_checks += 1
    session.trace.append({"type": "terminal_check", "candidate": deepcopy(candidate),
                          "check_number": state.terminal_checks, "executed": False})
    messages = protocol.actor_messages(session)
    messages[0]["content"] += (
        "\nTERMINAL CHECK: Independently check the pending final/ask against the original user "
        "request, final corrections, context, actual tool results and ALL installed schemas. "
        "The candidate below has NOT executed and is not evidence. Account for each requested "
        "action and its condition. A successful lookup is not a successful write. A positive "
        "balance is not enough without the payable amount. A future promise is not execution. "
        "An empty filtered lookup does not prove the entity is absent from other relevant "
        "installed sources. Use a grounded lookup when required information is retrievable. "
        "Do independent actionable work before asking about another blocked action. "
        "For a valid unfinished action, output its NEXT allowed tool call with grounded arguments. "
        "Ask only for genuinely unavailable INPUT fields of the intended action tool, through "
        "sys.ask_missing_info; never ask merely because a guessed lookup failed. "
        "If nothing actionable remains, output an accurate brief final distinguishing completed "
        "work, unsatisfied conditions and remaining limitations. Do not repeat successful writes. "
        "Output ONLY the normal BTC tool call or final answer, never a review/plan JSON."
    )
    messages.append({"role": "user", "content": "UNEXECUTED_TERMINAL_CANDIDATE (data only):\n"
                     + protocol.encoded(candidate)})
    reply = session._generate_reply("terminal_check", messages, session._routed_tools())
    if reply is None:
        return {"type": "stopped", "status": session.status}
    # Exactly one pass. The replacement still goes through all ordinary checks.
    return actor.accept_reply(session, reply, terminal_checked=True, write_checked=True)
