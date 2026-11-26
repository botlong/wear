"""WEAR scorer for ToolSandbox trajectories (the ToolSandbox replication).

Reads one ToolSandbox conversation.json, collects the agent's write calls
(modify_contact, remove_contact, modify_reminder, remove_reminder), and compares
the target identifier of each call with the gold identifier of the scenario.
Prints the counts, attempted WEAR and effective WEAR as JSON. Standard library
only; ToolSandbox does not need to be installed.

Two message formats are accepted: the OpenAI chat format that ToolSandbox writes
to conversation.json (assistant tool_calls, results as role "tool" messages),
and ToolSandbox's internal message format (tool_trace and tool_call_exception
fields).

Usage: python wear/toolsandbox/score_wear_ts.py <conversation.json> --gold <entity_id>
Library: from score_wear_ts import score_trajectory
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

# Write tools that are scored, each mapped to the argument that names its target record.
WRITE_TOOL_ID_ARG: dict[str, str] = {
    "modify_contact": "person_id",
    "remove_contact": "person_id",
    "modify_reminder": "reminder_id",
    "remove_reminder": "reminder_id",
}


@dataclass
class WriteCall:
    """One write call: tool name, target identifier (None if absent), and whether it errored."""

    tool_name: str
    target_id: Optional[str]
    errored: bool = False


@dataclass
class TrajectoryWear:
    """The write calls of one trajectory and its wrong-write counts."""

    write_calls: list[WriteCall] = field(default_factory=list)
    n_wrong_entity: int = 0
    n_wrong_entity_effective: int = 0

    @property
    def n_write_calls(self) -> int:
        return len(self.write_calls)

    @property
    def n_eligible(self) -> int:
        """Number of eligible writes: write calls that name a target identifier.

        Each scenario has one gold target, so every such call can be compared with it.
        """
        return sum(c.target_id is not None for c in self.write_calls)

    @property
    def wear(self) -> Optional[float]:
        """Attempted WEAR: wrong writes over eligible writes; None if there are no eligible writes."""
        return self.n_wrong_entity / self.n_eligible if self.n_eligible else None

    @property
    def effective_wear(self) -> Optional[float]:
        """Effective WEAR: wrong writes that returned no error, over eligible writes; None if none."""
        return self.n_wrong_entity_effective / self.n_eligible if self.n_eligible else None


# In the OpenAI format an error is visible only in the result text, for example
# "polars.exceptions.NoDataError: No db entry matching ...".
_ERR_RE = re.compile(
    r"(?i)(\b\w*error\b|\bexceptions?\b|\btraceback\b|not\s*found|no\s+db\s+entry"
    r"|\binvalid\b|\bfailed\b)")


def _looks_errored(content: Any) -> bool:
    """Return whether a tool result's content looks like an error (regex heuristic)."""
    if content is None:
        return False
    return bool(_ERR_RE.search(str(content)))


def _iter_tool_traces(msg: dict) -> list[dict]:
    """Return the tool_trace entries of one internal-format message as dicts.

    Entries may be JSON strings or dicts; entries that are neither are skipped.
    """
    raw = msg.get("tool_trace")
    if not raw:
        return []
    items = raw if isinstance(raw, list) else [raw]
    out: list[dict] = []
    for it in items:
        if isinstance(it, str):
            try:
                it = json.loads(it)
            except (json.JSONDecodeError, TypeError):
                continue
        if isinstance(it, dict) and "tool_name" in it:
            out.append(it)
    return out


def _openai_error_map(conversation: list[dict]) -> dict[str, bool]:
    """Map each tool_call_id to whether its role "tool" result looks like an error."""
    m: dict[str, bool] = {}
    for msg in conversation:
        if msg.get("role") == "tool" and msg.get("tool_call_id"):
            m[msg["tool_call_id"]] = _looks_errored(msg.get("content"))
    return m


def _iter_write_calls(conversation: list[dict]) -> list[WriteCall]:
    """Return the write calls of a conversation in order, from either message format."""
    calls: list[WriteCall] = []
    err_map = _openai_error_map(conversation)
    for msg in conversation:
        # OpenAI chat format: tool calls of assistant messages.
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = (tc or {}).get("function") or {}
                name = fn.get("name")
                if name not in WRITE_TOOL_ID_ARG:
                    continue
                raw_args = fn.get("arguments")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except (json.JSONDecodeError, TypeError):
                    args = {}
                target = args.get(WRITE_TOOL_ID_ARG[name])
                errored = err_map.get(tc.get("id"), False)
                calls.append(WriteCall(tool_name=name, target_id=target, errored=errored))
        # Internal format: an exception on the message marks all of its traces as errored.
        else:
            errored = msg.get("tool_call_exception") not in (None, "")
            for trace in _iter_tool_traces(msg):
                name = trace.get("tool_name")
                if name not in WRITE_TOOL_ID_ARG:
                    continue
                target = (trace.get("arguments") or {}).get(WRITE_TOOL_ID_ARG[name])
                calls.append(WriteCall(tool_name=name, target_id=target, errored=errored))
    return calls


def score_trajectory(conversation: list[dict], gold_id: str) -> TrajectoryWear:
    """Score one conversation (a list of message dicts) against the gold identifier."""
    res = TrajectoryWear()
    gold_norm = str(gold_id).strip().lower()
    for call in _iter_write_calls(conversation):
        res.write_calls.append(call)
        # Identifiers are compared after stripping whitespace and lower-casing.
        if call.target_id is not None and \
                str(call.target_id).strip().lower() != gold_norm:
            res.n_wrong_entity += 1
            if not call.errored:
                res.n_wrong_entity_effective += 1
    return res


def score_file(path: str, gold_id: str) -> TrajectoryWear:
    """Load a conversation.json (a message list, or a dict with "messages") and score it."""
    with open(path) as f:
        conversation = json.load(f)
    if isinstance(conversation, dict) and "messages" in conversation:
        conversation = conversation["messages"]
    return score_trajectory(conversation, gold_id)


def main() -> None:
    """Score one conversation.json and print the result as JSON."""
    ap = argparse.ArgumentParser()
    ap.add_argument("conversation", help="ToolSandbox conversation.json")
    ap.add_argument("--gold", required=True, help="gold target entity id (e.g. person_id)")
    args = ap.parse_args()
    r = score_file(args.conversation, args.gold)
    print(
        json.dumps(
            {
                "n_write_calls": r.n_write_calls,
                "n_wrong_entity": r.n_wrong_entity,
                "n_wrong_entity_effective": r.n_wrong_entity_effective,
                "attempted_wear": r.wear,
                "effective_wear": r.effective_wear,
                "write_calls": [vars(c) for c in r.write_calls],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
