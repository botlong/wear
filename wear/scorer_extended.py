"""Extended WEAR scoring: annotated alignment of writes to gold writes, and state evidence.

score_binding() gives each agent write one status: correct, wrong_target,
unmatched_tool, extra_write (the allowed attempts of an annotated step are used
up), unassessable, or ambiguous. Correct and wrong-target writes form the eligible
set, and binding_error_rate (wrong / eligible) is attempted WEAR.

Without an alignment plan the rule is the same as in scorer.py: a write is aligned
only when its tool has a single gold write. With a plan, writes are aligned by
annotation instead, and their targets come from the plan's intents rather than from
the gold actions. This scores tasks with several same-tool gold writes, and it
lets a repair write that a user authorized later in the conversation count as
correct (extended scoring). The scorer never infers authorization from the text:
it only checks that the user message an annotation cites exists, comes earlier,
and matches the quoted text.

The tool response of each write is recorded as error, nonerror or unknown; it is
never taken as proof that stored state changed. Whether a write changed its target
record is known only from evidence the caller passes (`mutations`), for example
from a replay of the trajectory.

Alignment plan for one trajectory:
    {"sim_id": ...,                          # optional; must equal the simulation id
     "phases": [{                            # ordered, disjoint message windows
         "start_message": int, "end_message": int,   # half-open [start, end)
         "mode": "ordered" | "by_parameters",
         "authorization_message": int,       # optional: an earlier user message
         "authorization_quote": str,         # its exact content
         "intents": [{
             "id": str,                      # unique across the plan
             "tool": str,                    # a write tool
             "targets": [{id_key: value}],   # accepted targets, all with the same keys
             "selector": {arg: value},       # by_parameters only; non-identifier arguments
             "max_attempts": int | None,     # default 1 (must be 1 in ordered mode); None: no limit
             "retry_on_error": bool}]}]}     # an errored attempt does not use up the step
Each entry of wear/annotations/recovery_authorization_plans.json (the recovery-suite
annotations) holds such a plan under "plan". Library module; no command line.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from scorer import extract_gold_write_actions


def normalize_id(value: Any, mode: str = "legacy") -> str:
    """Normalize an identifier for comparison.

    The default mode strips whitespace and lowercases, as scorer.py does; mode
    "exact" compares str(value) unchanged. Raises ValueError for any other mode.
    """
    if mode == "legacy":
        return str(value).strip().lower()
    if mode == "exact":
        return str(value)
    raise ValueError("ID normalization must be 'legacy' or 'exact'")


def _matches_target(arguments: dict, targets: list[dict], mode: str) -> bool:
    """True if, for some target, every target key is present in arguments, not None,
    and equal to the target value after normalization."""
    return any(all(key in arguments and arguments[key] is not None
                   and normalize_id(arguments[key], mode) == normalize_id(value, mode)
                   for key, value in target.items()) for target in targets)


@dataclass
class ObservedWrite:
    """One agent write call.

    response is "error" or "nonerror" from the call's tool response, or "unknown"
    when no response can be attributed to the call. It says nothing about state.
    """
    index: int
    message_index: int
    call_id: str | None
    tool: str
    arguments: dict
    response: str


def extract_writes(sim: dict, write_tools: set[str]) -> list[ObservedWrite]:
    """Return the agent's write calls in order.

    A tool response sets the response status only if the call id is used by exactly
    one agent call, exactly one tool message carries that id, that message comes
    after the call, and it has an "error" field. Otherwise the status is "unknown".
    """
    messages = sim.get("messages") or []
    responses: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    ids: Counter = Counter()
    for i, message in enumerate(messages):
        if message.get("role") == "tool" and message.get("id") is not None:
            responses[message["id"]].append((i, message))
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                if call.get("requestor", "assistant") == "assistant":
                    ids[call.get("id")] += 1
    result = []
    for i, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if call.get("requestor", "assistant") != "assistant" or call.get("name") not in write_tools:
                continue
            call_id = call.get("id")
            candidates = responses.get(call_id, [])
            status = "unknown"
            if (call_id is not None and ids[call_id] == 1 and len(candidates) == 1
                    and candidates[0][0] > i and "error" in candidates[0][1]):
                status = "error" if candidates[0][1]["error"] else "nonerror"
            result.append(ObservedWrite(len(result), i, call_id, call["name"],
                                        dict(call.get("arguments") or {}), status))
    return result


@dataclass
class WriteAssessment:
    """Scoring result for one write.

    status and reason give the alignment outcome; intent is the aligned plan intent,
    if any; expected_targets are the targets the write was compared with; and
    target_changed is the caller's state evidence (None when not supplied).
    """
    index: int
    message_index: int
    call_id: str | None
    tool: str
    status: str
    reason: str
    intent: str | None
    expected_targets: list[dict]
    response: str
    target_changed: bool | None


@dataclass
class BindingReport:
    """All write assessments of one trajectory.

    With a plan, also lists the intents that no write attempted and the intents
    that no write hit with a correct target.
    """
    sim_id: Any
    task_id: Any
    writes: list[WriteAssessment] = field(default_factory=list)
    unattempted_intents: list[str] = field(default_factory=list)
    intents_without_correct_target: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return the report as a dict of counts, rates and per-write rows.

        binding_error_rate is wrong / eligible (attempted WEAR), None without an
        eligible write. n_wrong_nonerror counts the wrong writes with no tool error
        (the numerator of effective WEAR). The n_wrong_verified_* counts split the
        wrong writes by state evidence: changed, unchanged, or no evidence.
        """
        counts = Counter(w.status for w in self.writes)
        n = len(self.writes)
        wrong = [w for w in self.writes if w.status == "wrong_target"]
        eligible = counts["correct"] + len(wrong)
        unmatched = counts["unmatched_tool"] + counts["extra_write"]
        unassessed = counts["unassessable"] + counts["ambiguous"]
        assert eligible + unmatched + unassessed == n
        return {
            "sim_id": self.sim_id, "task_id": self.task_id,
            "n_writes": n, "no_write": n == 0,
            "n_eligible": eligible, "n_correct": counts["correct"],
            "n_wrong_target": len(wrong),
            "binding_error_rate": len(wrong) / eligible if eligible else None,
            "n_unmatched": unmatched, "n_unmatched_tool": counts["unmatched_tool"],
            "n_extra": counts["extra_write"],
            "unmatched_write_rate": unmatched / n if n else None,
            "n_unassessable": counts["unassessable"], "n_ambiguous": counts["ambiguous"],
            "eligible_fraction": eligible / n if n else None,
            "n_wrong_nonerror": sum(w.response == "nonerror" for w in wrong),
            "n_wrong_error": sum(w.response == "error" for w in wrong),
            "n_wrong_response_unknown": sum(w.response == "unknown" for w in wrong),
            "n_wrong_verified_mutation": sum(w.target_changed is True for w in wrong),
            "n_wrong_verified_noop": sum(w.target_changed is False for w in wrong),
            "n_wrong_state_unverified": sum(w.target_changed is None for w in wrong),
            "unattempted_intents": self.unattempted_intents,
            "intents_without_correct_target": self.intents_without_correct_target,
            "writes": [asdict(w) for w in self.writes],
        }


def _gold_targets(gold, object_id_keys: set[str]) -> list[dict]:
    """Return the target of a gold action as [{key: value}], or [] when it names none.

    A gold action names no target when tau2 matches it by tool name only
    (compare_args == []), when it has no target-identifier key, or when a key's
    value is missing.
    """
    if gold.compare_args == []:
        return []
    keys = (set(gold.compare_args) & object_id_keys if gold.compare_args is not None
            else set(gold.arguments) & object_id_keys)
    if not keys or any(k not in gold.arguments or gold.arguments[k] is None for k in keys):
        return []
    return [{k: gold.arguments[k] for k in sorted(keys)}]


def _validate_plan(plan: dict, sim: dict, write_tools: set[str], object_id_keys: set[str]) -> list[dict]:
    """Check an alignment plan against the trajectory and return its phases.

    Raises ValueError on any violation of the plan format (see the module
    docstring). A cited authorization is checked against the trajectory but not
    interpreted.
    """
    messages = sim.get("messages") or []
    if "sim_id" in plan and plan["sim_id"] != sim.get("id"):
        raise ValueError("Alignment plan belongs to another simulation")
    phases = plan.get("phases")
    if not isinstance(phases, list):
        raise ValueError("An alignment plan needs a phases list")
    previous_end, labels = 0, set()
    for phase in phases:
        start, end = phase.get("start_message"), phase.get("end_message")
        if not (type(start) is int and type(end) is int
                and previous_end <= start < end <= len(messages)):
            raise ValueError("Phase windows must be ordered, disjoint, and within messages")
        previous_end = end
        if phase.get("mode") not in ("ordered", "by_parameters"):
            raise ValueError("Phase mode must be ordered or by_parameters")
        auth = phase.get("authorization_message")
        if auth is not None:
            if not (type(auth) is int and 0 <= auth < start and messages[auth].get("role") == "user"):
                raise ValueError("Authorization must cite an earlier user message")
            if phase.get("authorization_quote") != messages[auth].get("content"):
                raise ValueError("Authorization quote does not match the cited user message")
        intents = phase.get("intents")
        if not isinstance(intents, list) or not intents:
            raise ValueError("Each phase needs explicit intents")
        for intent in intents:
            label = intent.get("id")
            if not isinstance(label, str) or not label or label in labels:
                raise ValueError("Intent ids must be nonempty and globally unique")
            labels.add(label)
            if intent.get("tool") not in write_tools:
                raise ValueError("Intent tool must be an annotated WRITE tool")
            targets = intent.get("targets")
            if not isinstance(targets, list) or not targets:
                raise ValueError("An intent needs at least one explicit target")
            keys = set(targets[0]) if isinstance(targets[0], dict) else set()
            if not keys or not keys <= object_id_keys:
                raise ValueError("Target keys must be known entity fields")
            if any(not isinstance(t, dict) or set(t) != keys or any(v is None for v in t.values())
                   for t in targets):
                raise ValueError("Alternative targets must have the same non-null entity fields")
            selector = intent.get("selector", {})
            if not isinstance(selector, dict) or set(selector) & object_id_keys:
                raise ValueError("Intent selectors cannot depend on the observed target ID")
            if phase["mode"] == "ordered" and selector:
                raise ValueError("Ordered alignment uses step order; selectors require by_parameters mode")
            limit = intent.get("max_attempts", 1)
            if limit is not None and (type(limit) is not int or limit < 1):
                raise ValueError("max_attempts must be positive or null for explicit repetition")
            if phase["mode"] == "ordered" and limit != 1:
                raise ValueError("Ordered intents each consume one attempt; repeat steps explicitly")
            if type(intent.get("retry_on_error", False)) is not bool:
                raise ValueError("retry_on_error must be a boolean")
    return phases


def score_binding(
    sim: dict,
    task: dict,
    write_tools: set[str],
    object_id_keys: set[str],
    *,
    plan: dict | None = None,
    normalization: str = "legacy",
    mutations: Mapping[str, bool] | None = None,
) -> BindingReport:
    """Score the writes of one trajectory and return a BindingReport.

    Without a plan, a write is aligned only when its tool has a single gold write.
    With a plan, a write is aligned within the phase whose message window contains
    it: in ordered mode to the next step not yet used up, in by_parameters mode to
    the one same-tool intent whose selector matches the write's arguments. The
    target identifier the agent passed is never used for alignment.

    `mutations` maps write call IDs (each used by exactly one write) to whether the
    target record's stored data changed across that write, from an independent
    state comparison (the state just before the write, after any perturbation,
    against the state just after it). Each bool is the caller's assertion; this
    function replays nothing. Writes without evidence get target_changed=None.
    `normalization` is passed to normalize_id.
    """
    normalize_id("", normalization)  # reject an unknown mode before scoring
    observed = extract_writes(sim, write_tools)
    mutations = {} if mutations is None else mutations
    ids = Counter(w.call_id for w in observed)
    if any(key is None or ids[key] != 1 or type(value) is not bool for key, value in mutations.items()):
        raise ValueError("Mutation evidence needs unique write IDs and explicit boolean observations")
    report = BindingReport(sim.get("id"), sim.get("task_id"))
    gold_by_tool = defaultdict(list)
    for gold in extract_gold_write_actions(task, write_tools):
        gold_by_tool[gold.name].append(gold)
    phases = _validate_plan(plan, sim, write_tools, object_id_keys) if plan is not None else None
    consumed: Counter = Counter()
    attempted, correct = set(), set()

    for write in observed:
        intent, targets, reason = None, [], ""
        if phases is None:
            golds = gold_by_tool.get(write.tool, [])
            if not golds:
                status, reason = "unmatched_tool", "no gold action for this WRITE tool"
            elif len(golds) > 1:
                status, reason = "ambiguous", "multiple same-tool gold actions need explicit alignment"
            else:
                targets = _gold_targets(golds[0], object_id_keys)
                if not targets:
                    status, reason = "unassessable", "gold action has no assessable entity fields"
                else:
                    status = "correct" if _matches_target(write.arguments, targets, normalization) else "wrong_target"
                    reason = "single same-tool gold target"
        else:
            phase = next((p for p in phases if p["start_message"] <= write.message_index < p["end_message"]), None)
            if phase is None:
                status, reason = "unassessable", "write lies outside annotated intent windows"
            elif phase["mode"] == "ordered":
                remaining = [a for a in phase["intents"] if consumed[a["id"]] == 0]
                if not remaining:
                    status, reason = "extra_write", "all annotated ordered steps were already consumed"
                elif write.tool != remaining[0]["tool"]:
                    # a write with another tool does not advance the steps
                    status, reason = "unmatched_tool", "tool does not match the current ordered step"
                else:
                    intent = remaining[0]
            else:
                same_tool = [a for a in phase["intents"] if a["tool"] == write.tool]
                matches = [a for a in same_tool if all(k in write.arguments and write.arguments[k] == v
                           for k, v in a.get("selector", {}).items())]
                # Intents whose attempts are used up stay in `matches`, so neither the
                # target ID nor a used-up budget can single out one of several
                # intents with the same selector.
                if not same_tool:
                    status, reason = "unmatched_tool", "no annotated intent for this WRITE tool"
                elif not matches:
                    status, reason = "unassessable", "parameters do not identify an annotated intent"
                elif len(matches) > 1:
                    status, reason = "ambiguous", "non-target arguments match several intents"
                elif (matches[0].get("max_attempts", 1) is not None
                      and consumed[matches[0]["id"]] >= matches[0].get("max_attempts", 1)):
                    status, reason = "extra_write", "intent attempt budget is exhausted"
                else:
                    intent = matches[0]
            if intent is not None:
                label = intent["id"]
                attempted.add(label)
                targets = intent["targets"]
                status = "correct" if _matches_target(write.arguments, targets, normalization) else "wrong_target"
                reason = "explicit " + phase["mode"] + " alignment"
                if status == "correct":
                    correct.add(label)
                # with retry_on_error, an errored attempt does not use up the intent
                if not (intent.get("retry_on_error", False) and write.response == "error"):
                    consumed[label] += 1
        report.writes.append(WriteAssessment(
            write.index, write.message_index, write.call_id, write.tool, status, reason,
            intent["id"] if intent is not None else None, targets, write.response,
            mutations.get(write.call_id),
        ))
    if phases is not None:
        labels = [a["id"] for p in phases for a in p["intents"]]
        report.unattempted_intents = [label for label in labels if label not in attempted]
        report.intents_without_correct_target = [label for label in labels if label not in correct]
    return report
