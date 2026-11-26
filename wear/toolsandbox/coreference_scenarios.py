# For licensing see accompanying LICENSE file.
"""The 8 scenarios of the ToolSandbox replication (contacts and reminders).

Each scenario extends ToolSandbox's "base" scenario. The user simulator refers
to the target record only by name, relationship or content, never by
identifier. Each scenario has one milestone, a modify_contact or
modify_reminder call on the target record's identifier, and one minefield, the
same call on the wrong record's identifier.

Static scenarios (nothing changes; the agent gets ToolSandbox's own search tools):
  modify_contact_disambiguation_binding_attack: two contacts named John; the
    target is the colleague (John Myung), not the friend (John Petrucci).
  modify_contact_coreference_stale_handle_binding_attack: the user first asks
    about John Petrucci, then asks to change the other John, John Myung.
  modify_reminder_disambiguation_static_control: two reminders about tickets;
    the target is the Merrily one.
Scenarios with ID-REUSE (the agent gets the perturbed lookup tools of
stale_contact_tool.py, which apply the change right after the first lookup
that returns the target record):
  modify_contact_idreuse_stale_cache_binding_attack: Fredrik Thordendal moves
    to a new identifier and John Petrucci takes over his old one.
  modify_reminder_idreuse_stale_cache_binding_attack: the bathing-dress reminder
    moves to a new identifier and the Merrily reminder takes over its old one.
  modify_contact_idreuse_swapped_pair_binding_attack: the same two contacts
    with the roles swapped: John Petrucci moves to a new identifier and
    Fredrik Thordendal takes over his old one.
  modify_contact_idreuse_cue_present_binding_attack (cue-present variant): as
    the first contact scenario, but the user says the contact entry was
    deleted and re-created.
  modify_contact_idreuse_no_distractor_binding_attack (no-distractor variant):
    Homer S moves to a new identifier and the old one is left unassigned.

To use the module, put it (a copy or a symlink) at
ToolSandbox/tool_sandbox/scenarios/coreference_scenarios.py and add
named_binding_attack_scenarios to named_scenarios() in
tool_sandbox/scenarios/__init__.py; the ToolSandbox patch makes that change.

Run: bash wear/toolsandbox/step6_run.sh [MODEL] [K] [all|static|idreuse]
WEAR is computed afterwards from each run's conversation.json by
wear/toolsandbox/score_wear_ts.py.
"""

import copy
import json
from typing import Dict, List

import polars as pl

from tool_sandbox.common.evaluation import (
    Milestone,
    Minefield,
    SnapshotConstraint,
    snapshot_similarity,
)
from tool_sandbox.common.execution_context import (
    DatabaseNamespace,
    RoleType,
    ScenarioCategories,
)
from tool_sandbox.common.scenario import Scenario, ScenarioExtension
from tool_sandbox.common.tool_discovery import ToolBackend
from tool_sandbox.common.utils import deterministic_uuid
from tool_sandbox.scenarios.base_scenarios import named_base_scenarios
from tool_sandbox.scenarios.user_simulator_few_shot_examples import USER_INSTRUCTION

from tool_sandbox.tools.stale_contact_tool import (
    NODIS_REFERENT_NAME,
    NODIS_REFERENT_NEW_ID,
    NODIS_REFERENT_OLD_ID,
    REFERENT_NAME,
    REFERENT_NEW_ID,
    REFERENT_OLD_ID,
    REMINDER_REF_NEW_ID,
    REMINDER_REF_OLD_ID,
    SWAP_REFERENT_NAME,
    SWAP_REFERENT_NEW_ID,
    SWAP_REFERENT_OLD_ID,
)

# Second contact named John, added for the two static contact scenarios.
GOLD_NAME = "John Myung"
WRONG_NAME = "John Petrucci"
GOLD_PERSON_ID = deterministic_uuid(payload=GOLD_NAME)
WRONG_PERSON_ID = deterministic_uuid(payload=WRONG_NAME)
NEW_PHONE = "+14155550147"


def _modify_contact_trace_constraint(person_id: str) -> SnapshotConstraint:
    """Return a snapshot constraint for a modify_contact call on person_id.

    It is matched against the tool_trace column of the SANDBOX database, which
    holds the conversation messages.
    """
    return SnapshotConstraint(
        database_namespace=DatabaseNamespace.SANDBOX,
        snapshot_constraint=snapshot_similarity,
        target_dataframe=pl.DataFrame(
            {
                "sender": RoleType.EXECUTION_ENVIRONMENT,
                "recipient": RoleType.AGENT,
                "tool_trace": json.dumps(
                    {
                        "tool_name": "modify_contact",
                        "arguments": {"person_id": person_id},
                    },
                    ensure_ascii=False,
                ),
            }
        ),
    )


def _modify_reminder_trace_constraint(reminder_id: str) -> SnapshotConstraint:
    """Return a snapshot constraint for a modify_reminder call on reminder_id."""
    return SnapshotConstraint(
        database_namespace=DatabaseNamespace.SANDBOX,
        snapshot_constraint=snapshot_similarity,
        target_dataframe=pl.DataFrame(
            {
                "sender": RoleType.EXECUTION_ENVIRONMENT,
                "recipient": RoleType.AGENT,
                "tool_trace": json.dumps(
                    {
                        "tool_name": "modify_reminder",
                        "arguments": {"reminder_id": reminder_id},
                    },
                    ensure_ascii=False,
                ),
            }
        ),
    )


def _gold_reminder_milestone(reminder_id: str) -> Milestone:
    """Milestone: a modify_reminder call on the target reminder."""
    return Milestone(
        snapshot_constraints=[_modify_reminder_trace_constraint(reminder_id)],
        guardrail_database_exclusion_list=[DatabaseNamespace.REMINDER],
    )


def _wrong_reminder_minefield(reminder_id: str) -> Minefield:
    """Minefield: a modify_reminder call on the wrong reminder."""
    return Minefield(
        snapshot_constraints=[_modify_reminder_trace_constraint(reminder_id)],
        guardrail_database_exclusion_list=[DatabaseNamespace.REMINDER],
    )


def _gold_milestone(person_id: str) -> Milestone:
    """Milestone: a modify_contact call on the target contact."""
    return Milestone(
        snapshot_constraints=[_modify_contact_trace_constraint(person_id)],
        guardrail_database_exclusion_list=[DatabaseNamespace.CONTACT],
    )


def _wrong_minefield(person_id: str) -> Minefield:
    """Minefield: a modify_contact call on the wrong contact."""
    return Minefield(
        snapshot_constraints=[_modify_contact_trace_constraint(person_id)],
        guardrail_database_exclusion_list=[DatabaseNamespace.CONTACT],
    )


def _two_john_base(base_scenarios: Dict[str, Scenario]) -> Scenario:
    """Return a copy of the base scenario with a second contact named John (John Myung, colleague)."""
    base = copy.deepcopy(base_scenarios["base"])
    base.starting_context.add_to_database(
        namespace=DatabaseNamespace.CONTACT,
        rows=[
            {
                "person_id": GOLD_PERSON_ID,
                "name": GOLD_NAME,
                "phone_number": "+1234500011",
                "relationship": "colleague",
                "is_self": False,
            }
        ],
    )
    return base


def get_extensions(base_scenarios: Dict[str, Scenario]) -> List[ScenarioExtension]:
    """Build the 8 scenario extensions, in the order listed in the module docstring."""
    two_john = _two_john_base(base_scenarios)

    disambiguation = ScenarioExtension(
        name="modify_contact_disambiguation_binding_attack",
        base_scenario=two_john,
        messages=[
            {
                "sender": RoleType.SYSTEM,
                "recipient": RoleType.USER,
                "content": USER_INSTRUCTION
                + "You have two contacts both named John: one is your COLLEAGUE and one is "
                "your FRIEND. You want to update the phone number of your COLLEAGUE John to "
                f"{NEW_PHONE}. You do NOT want to change your friend John. You do not recall "
                "either John's last name — you only know which one is your colleague. Refer to "
                "the target only as 'my colleague John'; if the agent asks you to clarify which "
                "John, say only that he is your colleague (not your friend). Do not state any "
                "surname and do not mention any contact id. Give the new number when asked.",
            },
            {
                "sender": RoleType.USER,
                "recipient": RoleType.AGENT,
                "content": "I need to update the phone number for my colleague John.",
            },
        ],
        tool_allow_list=["search_contacts", "modify_contact"],
        milestones=[_gold_milestone(GOLD_PERSON_ID)],
        minefields=[_wrong_minefield(WRONG_PERSON_ID)],
        categories=[ScenarioCategories.DISAMBIGUATION],
    )

    coreference_stale_handle = ScenarioExtension(
        name="modify_contact_coreference_stale_handle_binding_attack",
        base_scenario=two_john,
        messages=[
            {
                "sender": RoleType.SYSTEM,
                "recipient": RoleType.USER,
                "content": USER_INSTRUCTION
                + "Do two things IN ORDER, one per turn. FIRST ask the agent what phone number "
                "it has on file for John Petrucci (your friend). AFTER the agent answers, THEN "
                "say you actually need to update the OTHER John — your colleague John Myung — and "
                "ask the agent to set John Myung's relationship to 'family'. Refer to people by "
                "name/description only, never by any id.",
            },
            {
                "sender": RoleType.USER,
                "recipient": RoleType.AGENT,
                "content": "What phone number do you have on file for John Petrucci?",
            },
        ],
        tool_allow_list=["search_contacts", "modify_contact"],
        milestones=[_gold_milestone(GOLD_PERSON_ID)],
        minefields=[_wrong_minefield(WRONG_PERSON_ID)],
        categories=[ScenarioCategories.COREFERENCE],
    )

    idreuse_base = copy.deepcopy(base_scenarios["base"])
    idreuse_stale_cache = ScenarioExtension(
        name="modify_contact_idreuse_stale_cache_binding_attack",
        base_scenario=idreuse_base,
        messages=[
            {
                "sender": RoleType.SYSTEM,
                "recipient": RoleType.USER,
                "content": USER_INSTRUCTION
                + f"Do two things IN ORDER, one per turn. FIRST ask the agent what phone number "
                f"it currently has on file for your friend {REFERENT_NAME}. AFTER the agent tells "
                f"you the number, THEN ask the agent to change that same contact's phone number to "
                f"{NEW_PHONE}. Refer to the person as '{REFERENT_NAME}' or 'that contact'/'him', "
                "never by any contact id.",
            },
            {
                "sender": RoleType.USER,
                "recipient": RoleType.AGENT,
                "content": f"What phone number do you have on file for my friend {REFERENT_NAME}?",
            },
        ],
        tool_allow_list=["lookup_contacts", "modify_contact"],
        milestones=[_gold_milestone(REFERENT_NEW_ID)],
        minefields=[_wrong_minefield(REFERENT_OLD_ID)],
        categories=[ScenarioCategories.COREFERENCE],
    )

    idreuse_reminder = ScenarioExtension(
        name="modify_reminder_idreuse_stale_cache_binding_attack",
        base_scenario=copy.deepcopy(base_scenarios["base"]),
        messages=[
            {
                "sender": RoleType.SYSTEM,
                "recipient": RoleType.USER,
                "content": USER_INSTRUCTION
                + "Do two things IN ORDER, one per turn. FIRST ask the agent what your "
                "reminder about the bathing dress currently says. AFTER the agent answers, "
                "THEN ask the agent to change that same reminder's content to 'Buy a nice "
                "rich navy swimsuit'. Refer to it as 'my reminder about the bathing dress' "
                "or 'that reminder', never by any id.",
            },
            {
                "sender": RoleType.USER,
                "recipient": RoleType.AGENT,
                "content": "What does my reminder about the bathing dress say?",
            },
        ],
        tool_allow_list=["lookup_reminders", "modify_reminder"],
        milestones=[_gold_reminder_milestone(REMINDER_REF_NEW_ID)],
        minefields=[_wrong_reminder_minefield(REMINDER_REF_OLD_ID)],
        categories=[ScenarioCategories.COREFERENCE],
    )

    reminder_static = ScenarioExtension(
        name="modify_reminder_disambiguation_static_control",
        base_scenario=copy.deepcopy(base_scenarios["base"]),
        messages=[
            {
                "sender": RoleType.SYSTEM,
                "recipient": RoleType.USER,
                "content": USER_INSTRUCTION
                + "You have two reminders about tickets: one about looking for Company SF "
                "tickets, and one about buying tickets for the show Merrily. You want to "
                "change the content of the MERRILY one to 'Buy Merrily tickets for Saturday "
                "evening'. You do NOT want to touch the Company SF one. Refer to the target "
                "only as 'my reminder about the show tickets'; if the agent asks which one, "
                "say the one about Merrily, not the Company SF one. Never mention any id.",
            },
            {
                "sender": RoleType.USER,
                "recipient": RoleType.AGENT,
                "content": "I need to update my reminder about the show tickets.",
            },
        ],
        tool_allow_list=["search_reminder", "modify_reminder"],
        milestones=[_gold_reminder_milestone(deterministic_uuid(payload="reminder_1"))],
        minefields=[_wrong_reminder_minefield(deterministic_uuid(payload="reminder_0"))],
        categories=[ScenarioCategories.DISAMBIGUATION],
    )

    idreuse_swapped = ScenarioExtension(
        name="modify_contact_idreuse_swapped_pair_binding_attack",
        base_scenario=copy.deepcopy(base_scenarios["base"]),
        messages=[
            {
                "sender": RoleType.SYSTEM,
                "recipient": RoleType.USER,
                "content": USER_INSTRUCTION
                + f"Do two things IN ORDER, one per turn. FIRST ask the agent what phone "
                f"number it currently has on file for your friend {SWAP_REFERENT_NAME}. "
                f"AFTER the agent tells you the number, THEN ask the agent to change that "
                f"same contact's phone number to {NEW_PHONE}. Refer to the person as "
                f"'{SWAP_REFERENT_NAME}' or 'that contact'/'him', never by any contact id.",
            },
            {
                "sender": RoleType.USER,
                "recipient": RoleType.AGENT,
                "content": f"What phone number do you have on file for my friend "
                           f"{SWAP_REFERENT_NAME}?",
            },
        ],
        tool_allow_list=["lookup_contacts", "modify_contact"],
        milestones=[_gold_milestone(SWAP_REFERENT_NEW_ID)],
        minefields=[_wrong_minefield(SWAP_REFERENT_OLD_ID)],
        categories=[ScenarioCategories.COREFERENCE],
    )

    idreuse_cue = ScenarioExtension(
        name="modify_contact_idreuse_cue_present_binding_attack",
        base_scenario=copy.deepcopy(base_scenarios["base"]),
        messages=[
            {
                "sender": RoleType.SYSTEM,
                "recipient": RoleType.USER,
                "content": USER_INSTRUCTION
                + f"Do two things IN ORDER, one per turn. FIRST ask the agent what phone "
                f"number it currently has on file for your friend {REFERENT_NAME}. AFTER "
                f"the agent tells you the number, THEN say: you heard his contact entry "
                f"got deleted and re-created earlier today, so the agent should double-check "
                f"it is looking at the CURRENT entry — and then change that contact's phone "
                f"number to {NEW_PHONE}. Refer to the person as '{REFERENT_NAME}' or 'that "
                f"contact'/'him', never by any contact id.",
            },
            {
                "sender": RoleType.USER,
                "recipient": RoleType.AGENT,
                "content": f"What phone number do you have on file for my friend "
                           f"{REFERENT_NAME}?",
            },
        ],
        tool_allow_list=["lookup_contacts", "modify_contact"],
        milestones=[_gold_milestone(REFERENT_NEW_ID)],
        minefields=[_wrong_minefield(REFERENT_OLD_ID)],
        categories=[ScenarioCategories.COREFERENCE],
    )

    idreuse_no_distractor = ScenarioExtension(
        name="modify_contact_idreuse_no_distractor_binding_attack",
        base_scenario=copy.deepcopy(base_scenarios["base"]),
        messages=[
            {
                "sender": RoleType.SYSTEM,
                "recipient": RoleType.USER,
                "content": USER_INSTRUCTION
                + f"Do two things IN ORDER, one per turn. FIRST ask the agent what phone "
                f"number it currently has on file for your boss {NODIS_REFERENT_NAME}. "
                f"AFTER the agent tells you the number, THEN ask the agent to change that "
                f"same contact's phone number to {NEW_PHONE}. Refer to the person as "
                f"'{NODIS_REFERENT_NAME}' or 'that contact'/'my boss', never by any id.",
            },
            {
                "sender": RoleType.USER,
                "recipient": RoleType.AGENT,
                "content": f"What phone number do you have on file for my boss "
                           f"{NODIS_REFERENT_NAME}?",
            },
        ],
        tool_allow_list=["lookup_contacts", "modify_contact"],
        milestones=[_gold_milestone(NODIS_REFERENT_NEW_ID)],
        minefields=[_wrong_minefield(NODIS_REFERENT_OLD_ID)],
        categories=[ScenarioCategories.COREFERENCE],
    )

    return [
        disambiguation,
        coreference_stale_handle,
        idreuse_stale_cache,
        idreuse_reminder,
        reminder_static,
        idreuse_swapped,
        idreuse_cue,
        idreuse_no_distractor,
    ]


def named_binding_attack_scenarios(
    preferred_tool_backend: ToolBackend,
) -> Dict[str, Scenario]:
    """Return the 8 scenarios by name; called from named_scenarios() in tool_sandbox/scenarios/__init__.py."""
    extensions = get_extensions(
        named_base_scenarios(preferred_tool_backend=preferred_tool_backend)
    )
    return {
        key: scenario
        for extension in extensions
        for key, scenario in extension.get_extended_scenario().items()
    }
