"""Unit tests for scorer_extended.py on constructed trajectories (no model or network calls).

Covers the eligible-set denominator, annotated alignment (ordered and by parameters),
authorizations, tool-response status and state evidence. Run from the repository root:
tau2-bench/.venv/bin/python -m pytest wear/test_scorer_extended.py
"""
from copy import deepcopy

import pytest

from scorer import score_simulation
from scorer_extended import score_binding

UPDATE = "modify_pending_order_address"
CANCEL = "cancel_pending_order"
WRITES = {UPDATE, CANCEL}
KEYS = {"order_id", "user_id"}


def action(target="A", name=UPDATE, **kwargs):
    """A call or gold action on order `target`; extra kwargs become arguments."""
    return {"name": name, "arguments": {"order_id": target, **kwargs}}


def task(*actions):
    return {"id": "t", "evaluation_criteria": {"actions": list(actions)}}


def sim(*calls):
    """Build a trajectory from call specs and user messages.

    Each call gets an assistant message and, unless response=False, a tool message
    whose error flag is the spec's "error" value. A spec with role "user" is copied
    as a user message.
    """
    messages = []
    for i, call in enumerate(calls):
        if call.get("role") == "user":
            messages.append(call)
            continue
        call_id = call.get("id", f"call-{i}")
        messages.append({"role": "assistant", "tool_calls": [{
            "id": call_id, "name": call["name"], "arguments": call["arguments"],
            "requestor": call.get("requestor", "assistant"),
        }]})
        if call.get("response", True):
            messages.append({"role": "tool", "id": call_id, "error": call.get("error", False)})
    return {"id": "s", "task_id": "t", "messages": messages}


def intent(label, target="A", name=UPDATE, **kwargs):
    """A plan intent with one accepted target order."""
    return {"id": label, "tool": name, "targets": [{"order_id": target}], **kwargs}


def plan(s, intents, mode="ordered"):
    """A one-phase plan whose window covers the whole trajectory."""
    return {"sim_id": s["id"], "phases": [{
        "start_message": 0, "end_message": len(s["messages"]),
        "mode": mode, "intents": intents,
    }]}


def score(s, t=None, **kwargs):
    """score_binding(...).to_dict(), by default against a single gold update of order A."""
    return score_binding(s, t or task(action()), WRITES, KEYS, **kwargs).to_dict()


def test_unmatched_harmful_writes_cannot_dilute_binding_error_rate():
    original = sim(action("B"))
    padded = sim(action("B"), *[action("VICTIM", name=CANCEL) for _ in range(9)])
    legacy = score_simulation(padded, task(action()), WRITES, KEYS)
    assert legacy.wear == 1 and legacy.wear_all_writes == 0.1  # only the simpler rule is diluted
    assert score(original)["binding_error_rate"] == score(padded)["binding_error_rate"] == 1
    result = score(padded)
    assert (result["n_wrong_target"], result["n_eligible"], result["n_unmatched_tool"]) == (1, 1, 9)
    assert result["unmatched_write_rate"] == 0.9


def test_no_write_and_only_unmatched_are_distinct_undefined_rates():
    empty, unrelated = score(sim()), score(sim(action(name=CANCEL)))
    assert empty["binding_error_rate"] is unrelated["binding_error_rate"] is None
    assert empty["no_write"] and not unrelated["no_write"]
    assert unrelated["unmatched_write_rate"] == 1


@pytest.mark.parametrize("gold", [
    {**action(), "compare_args": []},
    {"name": UPDATE, "arguments": {"address1": "Home"}},
    {"name": UPDATE, "arguments": {}, "compare_args": ["order_id"]},
    action(None),
])
def test_unassessable_gold_never_inflates_the_eligible_denominator(gold):
    result = score(sim(action("B")), task(gold))
    assert result["n_unassessable"] == 1 and result["n_eligible"] == 0
    assert result["binding_error_rate"] is None


def test_missing_actual_id_is_wrong_when_gold_identifies_target():
    result = score(sim({"name": UPDATE, "arguments": {"address1": "Home"}}))
    assert result["n_wrong_target"] == result["n_eligible"] == 1


def test_multigold_without_alignment_is_not_certified_by_match_any():
    t, s = task(action("A"), action("B")), sim(action("A"), action("A"))
    legacy = score_simulation(s, t, WRITES, KEYS)
    assert legacy.wear is None and legacy.n_ambiguous == 2  # not certified by matching any gold
    result = score(s, t)
    assert result["n_ambiguous"] == 2 and result["binding_error_rate"] is None


def test_ordered_alignment_catches_reusing_a_gold_action_twice():
    s = sim(action("A"), action("A"))
    result = score(s, plan=plan(s, [intent("first", "A"), intent("second", "B")]))
    assert [w["status"] for w in result["writes"]] == ["correct", "wrong_target"]
    assert result["binding_error_rate"] == 0.5
    assert result["intents_without_correct_target"] == ["second"]


def test_extra_writes_and_omitted_steps_have_explicit_counts():
    extra = sim(action(), action())
    r = score(extra, plan=plan(extra, [intent("only")]))
    assert r["n_extra"] == 1 and r["n_eligible"] == 1
    omitted = sim(action())
    r = score(omitted, plan=plan(omitted, [intent("first"), intent("second", "B")]))
    assert r["unattempted_intents"] == ["second"]


def test_wrong_tool_does_not_advance_ordered_alignment():
    s = sim(action("B", name=CANCEL), action("A"), action("B"))
    r = score(s, plan=plan(s, [intent("first", "A"), intent("second", "B")]))
    assert r["n_unmatched_tool"] == 1 and r["n_correct"] == 2


def test_ordered_alignment_rejects_selectors_instead_of_ignoring_them():
    s = sim(action("A", address1="Other"))
    p = plan(s, [intent("home", selector={"address1": "Home"})])
    with pytest.raises(ValueError, match="selectors require by_parameters"):
        score(s, plan=p)


def test_error_retry_requires_an_explicit_annotation():
    failed = {**action(), "error": True}
    s = sim(failed, action(), action("B"))
    intents = [intent("first", retry_on_error=True), intent("second", "B")]
    r = score(s, plan=plan(s, intents))
    assert r["n_correct"] == 3 and r["n_extra"] == 0
    intents[0]["retry_on_error"] = False
    r = score(s, plan=plan(s, intents))
    assert (r["n_wrong_target"], r["n_extra"]) == (1, 1)


def test_equivalent_targets_are_explicit_within_one_intent():
    s = sim(action("B"))
    i = intent("choose_one")
    i["targets"].append({"order_id": "B"})
    assert score(s, plan=plan(s, [i]))["n_correct"] == 1


def test_parameter_alignment_allows_order_changes_without_using_target_ids():
    intents = [intent("new", "A", selector={"address1": "New"}),
               intent("restore", "B", selector={"address1": "Original"})]
    s = sim(action("B", address1="Original"), action("A", address1="New"))
    r = score(s, plan=plan(s, intents, "by_parameters"))
    assert [w["intent"] for w in r["writes"]] == ["restore", "new"]
    assert r["n_correct"] == 2
    wrong = sim(action("B", address1="New"), action("A", address1="Original"))
    r = score(wrong, plan=plan(wrong, intents, "by_parameters"))
    assert r["n_wrong_target"] == 2  # both IDs are plan targets; the address decides the intent


def test_identical_selectors_remain_ambiguous_even_when_one_target_matches():
    s = sim(action("A", address1="New"))
    intents = [intent("one", "A", selector={"address1": "New"}),
               intent("two", "B", selector={"address1": "New"})]
    r = score(s, plan=plan(s, intents, "by_parameters"))
    assert r["n_ambiguous"] == 1 and r["n_eligible"] == 0


def test_parameters_without_a_matching_intent_are_unassessable():
    s = sim(action("B", address1="Other"))
    r = score(s, plan=plan(s, [intent("home", selector={"address1": "Home"})], "by_parameters"))
    assert r["n_unassessable"] == 1


def recovery_example():
    """A wrong write to B, a user turn authorizing the fix, then a correct write to A and
    a restore of B; the plan's second phase cites that user turn."""
    s = sim(action("B", address1="New"),
            {"role": "user", "content": "Correct A and restore B."},
            action("A", address1="New"), action("B", address1="Original"))
    p = {"sim_id": "s", "phases": [
        {"start_message": 0, "end_message": 3, "mode": "by_parameters",
         "intents": [intent("initial", "A", max_attempts=None)]},
        {"start_message": 3, "end_message": len(s["messages"]), "mode": "by_parameters",
         "authorization_message": 2, "authorization_quote": "Correct A and restore B.",
         "intents": [intent("correct", "A", selector={"address1": "New"}),
                     intent("restore", "B", selector={"address1": "Original"})]},
    ]}
    return s, p


def test_later_authorization_does_not_excuse_initial_wrong_write():
    s, p = recovery_example()
    assert score_simulation(s, task(action()), WRITES, KEYS).wear == 2 / 3
    result = score(s, plan=p)
    assert [w["status"] for w in result["writes"]] == ["wrong_target", "correct", "correct"]
    assert result["binding_error_rate"] == 1 / 3
    assert score(s)["n_wrong_target"] == 2  # without the plan, the restore of B is wrong too


def test_future_authorization_cannot_be_used_for_an_earlier_phase():
    s, p = recovery_example()
    p["phases"][0]["end_message"] = 1
    p["phases"][1]["start_message"] = 1
    with pytest.raises(ValueError, match="earlier user"):
        score(s, plan=p)


def test_wrong_target_after_authorization_is_still_detected():
    s, p = recovery_example()
    s["messages"][3]["tool_calls"][0]["arguments"]["order_id"] = "B"
    r = score(s, plan=p)
    assert r["n_wrong_target"] == 2 and r["writes"][-1]["status"] == "correct"


def test_no_error_does_not_imply_mutation():
    s = sim(action("B"), action("B"), {**action("B"), "error": True},
            {**action("B"), "response": False})
    r = score(s, mutations={"call-0": False, "call-1": True})
    assert (r["n_wrong_nonerror"], r["n_wrong_error"], r["n_wrong_response_unknown"]) == (2, 1, 1)
    assert (r["n_wrong_verified_mutation"], r["n_wrong_verified_noop"], r["n_wrong_state_unverified"]) == (1, 1, 2)


def test_duplicate_response_or_call_id_is_unknown():
    s = sim({**action("B"), "id": "same"}, {**action("B"), "id": "same"})
    r = score(s)
    assert r["n_wrong_response_unknown"] == 2 and r["n_wrong_nonerror"] == 0
    with pytest.raises(ValueError, match="unique write IDs"):
        score(s, mutations={"same": True})


@pytest.mark.parametrize("mutation", [{"not-a-call": True}, {"call-0": "true"}])
def test_state_evidence_cannot_silently_attach_to_another_call(mutation):
    with pytest.raises(ValueError, match="unique write IDs"):
        score(sim(action("B")), mutations=mutation)


def test_case_sensitive_ids_can_opt_out_of_legacy_normalization():
    s = sim(action(" a "))
    assert score(s)["n_correct"] == 1
    assert score(s, normalization="exact")["n_wrong_target"] == 1
    assert score(sim(action(123)), task(action("123")))["n_correct"] == 1


@pytest.mark.parametrize("change", ["target_selector", "overlap", "quote", "sim_id", "missing_target"])
def test_invalid_annotations_fail_explicitly(change):
    s, p = recovery_example()
    p = deepcopy(p)
    if change == "target_selector":
        p["phases"][1]["intents"][0]["selector"]["order_id"] = "A"
    elif change == "overlap":
        p["phases"][0]["end_message"] = 4
    elif change == "quote":
        p["phases"][1]["authorization_quote"] = "Invented permission"
    elif change == "sim_id":
        p["sim_id"] = "another-trajectory"
    elif change == "missing_target":
        p["phases"][1]["intents"][0]["targets"] = [{}]
    with pytest.raises(ValueError):
        score(s, plan=p)


def test_reads_and_user_calls_are_not_agent_write_attempts():
    s = sim(action(name="get_order_details"), {**action(), "requestor": "user"})
    assert score(s)["no_write"]
