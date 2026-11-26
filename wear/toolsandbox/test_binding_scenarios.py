"""Offline tests of the ToolSandbox scenarios and the perturbed lookup tools (no API calls).

The tests check that the 8 scenarios are registered, that the lookup tools apply
the intended ID-REUSE change on the first lookup of a target record, that the
search tool of a static scenario changes nothing, that the applied state does not
carry over between scenarios run in one process, and that the milestone and
minefield identifiers match the tables in stale_contact_tool.py.

Needs ToolSandbox installed in ToolSandbox/.venv with coreference_scenarios.py
and stale_contact_tool.py in place (see those modules).
Usage: ToolSandbox/.venv/bin/python -m pytest wear/toolsandbox/test_binding_scenarios.py
"""
import copy

import polars as pl
import pytest

from tool_sandbox.common.execution_context import (
    DatabaseNamespace,
    get_current_context,
    set_current_context,
)
from tool_sandbox.common.tool_discovery import ToolBackend
from tool_sandbox.scenarios import named_scenarios

import tool_sandbox.tools.stale_contact_tool as sct
from tool_sandbox.tools.stale_contact_tool import (
    CONTACT_PERTURB_TABLE,
    NODIS_REFERENT_NEW_ID,
    NODIS_REFERENT_OLD_ID,
    REFERENT_NEW_ID,
    REFERENT_OLD_ID,
    REMINDER_DIS_OLD_ID,
    REMINDER_REF_NEW_ID,
    REMINDER_REF_OLD_ID,
    SWAP_REFERENT_NEW_ID,
    SWAP_REFERENT_OLD_ID,
    lookup_contacts,
    lookup_reminders,
    reset_perturbation,
)

SCS = named_scenarios(preferred_tool_backend=ToolBackend.DEFAULT)
# The 8 scenarios of the ToolSandbox replication.
CANONICAL = [
    "modify_contact_disambiguation_binding_attack",
    "modify_contact_coreference_stale_handle_binding_attack",
    "modify_contact_idreuse_stale_cache_binding_attack",
    "modify_reminder_idreuse_stale_cache_binding_attack",
    "modify_reminder_disambiguation_static_control",
    "modify_contact_idreuse_swapped_pair_binding_attack",
    "modify_contact_idreuse_cue_present_binding_attack",
    "modify_contact_idreuse_no_distractor_binding_attack",
]


@pytest.fixture(autouse=True)
def _fresh_state():
    """Clear the applied-change state before and after each test."""
    reset_perturbation()
    yield
    reset_perturbation()


def _enter(scenario_name: str) -> None:
    """Make a copy of the scenario's starting context the current context."""
    set_current_context(copy.deepcopy(SCS[scenario_name].starting_context))


def _contact_ids() -> set:
    """Return the person ids now in the contact database."""
    df = get_current_context().get_database(namespace=DatabaseNamespace.CONTACT)
    return set(df["person_id"].to_list())


def _contact_name_at(pid: str) -> str:
    """Return the name of the contact stored under pid, or "" if there is none."""
    df = get_current_context().get_database(namespace=DatabaseNamespace.CONTACT)
    rows = df.filter(pl.col("person_id") == pid).to_dicts()
    return rows[0]["name"] if rows else ""


def test_all_canonical_scenarios_exist():
    """All 8 scenarios are in ToolSandbox's named_scenarios()."""
    for n in CANONICAL:
        assert n in SCS, n


def test_contact_repoint_main_pair():
    """The first lookup returns the old identifier and applies the change: Fredrik
    Thordendal moves to the new identifier and John Petrucci takes over the old one.
    A second lookup returns the new identifier."""
    _enter("modify_contact_idreuse_stale_cache_binding_attack")
    r1 = lookup_contacts(name="Fredrik Thordendal")
    assert [x["person_id"] for x in r1] == [REFERENT_OLD_ID]
    ids = _contact_ids()
    assert REFERENT_NEW_ID in ids
    assert _contact_name_at(REFERENT_OLD_ID) == "John Petrucci"
    r2 = lookup_contacts(name="Fredrik Thordendal")
    assert [x["person_id"] for x in r2] == [REFERENT_NEW_ID]


def test_swap_pair_fires_independently():
    """A lookup of John Petrucci applies only the swapped-pair entry: he moves to his new
    identifier, and Fredrik Thordendal takes over his old one and gets no new one."""
    _enter("modify_contact_idreuse_swapped_pair_binding_attack")
    r1 = lookup_contacts(name="John Petrucci")
    assert [x["person_id"] for x in r1] == [SWAP_REFERENT_OLD_ID]
    assert SWAP_REFERENT_NEW_ID in _contact_ids()
    assert _contact_name_at(SWAP_REFERENT_OLD_ID) == "Fredrik Thordendal"
    assert REFERENT_NEW_ID not in _contact_ids()


def test_broad_read_fires_exactly_one_entry():
    """A lookup that returns both contact targets applies only the first table entry."""
    _enter("modify_contact_idreuse_stale_cache_binding_attack")
    lookup_contacts(relationship="friend")
    ids = _contact_ids()
    fired = [old for old in CONTACT_PERTURB_TABLE
             if CONTACT_PERTURB_TABLE[old]["new"] in ids]
    assert len(fired) == 1
    assert fired[0] == REFERENT_OLD_ID


def test_no_distractor_dangling_old():
    """No-distractor scenario: after the change the old identifier names no contact,
    and modify_contact on it raises."""
    _enter("modify_contact_idreuse_no_distractor_binding_attack")
    r1 = lookup_contacts(name="Homer S")
    assert [x["person_id"] for x in r1] == [NODIS_REFERENT_OLD_ID]
    ids = _contact_ids()
    assert NODIS_REFERENT_NEW_ID in ids
    assert NODIS_REFERENT_OLD_ID not in ids
    from tool_sandbox.tools.contact import modify_contact
    with pytest.raises(Exception):
        modify_contact(person_id=NODIS_REFERENT_OLD_ID, phone_number="+14155550100")


def test_reminder_repoint():
    """The bathing-dress reminder moves to its new identifier and the Merrily
    reminder takes over the old one."""
    _enter("modify_reminder_idreuse_stale_cache_binding_attack")
    r1 = lookup_reminders(content="bathing dress")
    assert [x["reminder_id"] for x in r1] == [REMINDER_REF_OLD_ID]
    df = get_current_context().get_database(namespace=DatabaseNamespace.REMINDER)
    ids = set(df["reminder_id"].to_list())
    assert REMINDER_REF_NEW_ID in ids
    assert REMINDER_DIS_OLD_ID not in ids
    old_row = df.filter(pl.col("reminder_id") == REMINDER_REF_OLD_ID).to_dicts()[0]
    assert "Merrily" in old_row["content"]
    r2 = lookup_reminders(content="bathing dress")
    assert [x["reminder_id"] for x in r2] == [REMINDER_REF_NEW_ID]


def test_no_cross_scenario_leak_same_process():
    """Two scenarios with the same target, run in one process, both apply the change."""
    _enter("modify_contact_idreuse_stale_cache_binding_attack")
    lookup_contacts(name="Fredrik Thordendal")
    assert REFERENT_NEW_ID in _contact_ids()
    _enter("modify_contact_idreuse_cue_present_binding_attack")
    assert REFERENT_NEW_ID not in _contact_ids()
    lookup_contacts(name="Fredrik Thordendal")
    assert REFERENT_NEW_ID in _contact_ids()


def test_static_scenarios_have_no_perturbation():
    """ToolSandbox's own search_reminder, used by a static scenario, leaves the database unchanged."""
    _enter("modify_reminder_disambiguation_static_control")
    from tool_sandbox.tools.reminder import search_reminder
    before = get_current_context().get_database(namespace=DatabaseNamespace.REMINDER)
    search_reminder(content="tickets")
    after = get_current_context().get_database(namespace=DatabaseNamespace.REMINDER)
    assert before.equals(after)


def test_milestone_minefield_ids():
    """Milestone and minefield identifiers match the perturbation-table constants."""
    import json as _json

    def _anchor_ids(name, key):
        matcher = getattr(SCS[name].evaluation, f"{key}_matcher")
        out = []
        for m in matcher.milestones:
            for c in m.snapshot_constraints:
                tt = _json.loads(c.target_dataframe["tool_trace"][0])
                args = tt["arguments"]
                out.append(args.get("person_id") or args.get("reminder_id"))
        return out

    assert _anchor_ids("modify_contact_idreuse_stale_cache_binding_attack",
                       "milestone") == [REFERENT_NEW_ID]
    assert _anchor_ids("modify_contact_idreuse_stale_cache_binding_attack",
                       "minefield") == [REFERENT_OLD_ID]
    assert _anchor_ids("modify_contact_idreuse_swapped_pair_binding_attack",
                       "milestone") == [SWAP_REFERENT_NEW_ID]
    assert _anchor_ids("modify_contact_idreuse_swapped_pair_binding_attack",
                       "minefield") == [SWAP_REFERENT_OLD_ID]
    assert _anchor_ids("modify_reminder_idreuse_stale_cache_binding_attack",
                       "milestone") == [REMINDER_REF_NEW_ID]
    assert _anchor_ids("modify_reminder_idreuse_stale_cache_binding_attack",
                       "minefield") == [REMINDER_REF_OLD_ID]
    assert _anchor_ids("modify_contact_idreuse_no_distractor_binding_attack",
                       "milestone") == [NODIS_REFERENT_NEW_ID]
