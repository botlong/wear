#!/usr/bin/env python3
"""Generate the recovery suite: 5 retail ID-REUSE tasks in which a wrong write can be undone.

The tasks are built like the retail tasks of the main suite (gen_suite.py), with
two differences. The distractor's shipping address equals the user's profile
address, so after a wrong address change to the distractor its original address
can be restored from the profile. And the user makes a third request: after the address change,
the user asks the agent to look up the orders again, confirm that only the
target order has the new address, and fix anything that is off. A wrong write
that the agent then undoes leaves the final database equal to the gold state.

The script reads the tau2-bench retail database, checks each row with hard
asserts, and writes wear/deictic_tasks/retail_recover_<slug>.json. Output is
deterministic.

Usage: python3 wear/gen_recoverable.py [--check]   # --check builds and checks, writes nothing
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..", "tau2-bench", "data", "tau2", "domains")
OUT = os.path.join(HERE, "deictic_tasks")

# Columns: slug, user_id, target order, product the user names, distractor order,
# new identifier of the target, new shipping address (address1, address2, city,
# state, zip).
RECOVERABLE = [
    ("toothbrush", "harper_santos_8115", "#W6629830", "electric toothbrush",
     "#W4941028", "#W0000011",
     ("12 Alder Court", "Apt 6", "Columbus", "OH", "43215")),
    ("ereader", "liam_thomas_7882", "#W1654931", "e-reader",
     "#W3295833", "#W0000012",
     ("89 Pinehurst Boulevard", "Unit 11", "Tucson", "AZ", "85713")),
    ("backpack", "james_kim_7213", "#W9154975", "backpack",
     "#W3289292", "#W0000013",
     ("455 Meadowbrook Lane", "Apt 3A", "Richmond", "VA", "23220")),
    ("gamingmouse", "olivia_ito_3591", "#W5442520", "gaming mouse",
     "#W7941031", "#W0000014",
     ("1608 Willow Bend Drive", "Apt 12C", "Portland", "OR", "97214")),
    ("desklamp", "mia_jackson_2250", "#W7807323", "desk lamp",
     "#W1205816", "#W0000015",
     ("742 Juniper Street", "Suite 5", "Madison", "WI", "53703")),
]

# User simulator text: task instructions, and the reason for the call with {item}
# and {addr} filled in per task.
TASK_INSTRUCTIONS = (
    "You always refer to your orders by what is inside them (their products), "
    "never by order id (you don't remember any order codes). You reveal "
    "information only when the agent asks. Once you have named the order by "
    "its contents, in later turns refer to that order as 'that order' or 'it' "
    "instead of describing it again. Ask for only one thing at a time and "
    "wait for the agent's reply before the next request."
)
REASON = (
    "You want to do three things in order. FIRST, ask the agent to tell you "
    "which items are in your order that contains the {item}. AFTER the agent "
    "has listed those items, THEN ask the agent to change the shipping "
    "address of that same order to your new address: {addr}. If the agent "
    "asks how to verify your identity, give your email address (below); do "
    "NOT give the new shipping zip as your account zip. FINALLY, after the "
    "agent confirms the address change, say you want to be absolutely sure "
    "it went to the right order: ask the agent to look up your orders again "
    "and confirm that the order containing the {item} now has the new "
    "address AND that every other order of yours still has its original "
    "shipping address — and if anything is off, ask the agent to fix it so "
    "exactly that is true."
)


def build() -> list[dict]:
    """Check each row against the retail database and build its task dict."""
    db = json.load(open(os.path.join(BASE, "retail", "db.json")))
    users, orders = db["users"], db["orders"]
    tasks = []
    for slug, uid, oid, item, dis, new_id, addr in RECOVERABLE:
        u, o, d = users[uid], orders[oid], orders[dis]
        assert o["status"] == "pending", f"{slug}: referent not pending"
        assert d["status"] == "pending", f"{slug}: distractor not pending"
        assert oid in u["orders"] and dis in u["orders"] and dis != oid, f"{slug}: ids"
        # The distractor ships to the profile address, so a wrong write to it can be reverted.
        assert d["address"] == u["address"], \
            f"{slug}: distractor address != profile address (unrecoverable)"
        counts: dict[str, int] = {}
        for x in u["orders"]:
            for it in orders[x]["items"]:
                counts[it["name"].lower()] = counts.get(it["name"].lower(), 0) + 1
        assert counts.get(item, 0) == 1, f"{slug}: item '{item}' not unique"
        assert any(it["name"].lower() == item for it in o["items"]), \
            f"{slug}: item not in referent"
        for x in u["orders"]:
            for it in orders[x]["items"]:
                other = it["name"].lower()
                if other != item:
                    assert item not in other and other not in item, \
                        f"{slug}: phrase collision '{item}' vs '{other}'"
        assert new_id not in orders, f"{slug}: {new_id} exists"
        a1, a2, city, state, zc = addr
        assert zc != u["address"]["zip"] and zc != o["address"]["zip"], f"{slug}: zip"
        addr_str = f"{a1}, {a2}, {city}, {state} {zc}, USA"
        name = f'{u["name"]["first_name"]} {u["name"]["last_name"]}'
        tasks.append({
            "id": f"retail_recover_{slug}",
            "_doc": (
                f"RECOVERABLE id_reuse: referent = order with the "
                f"{item} ({oid}), distractor = {dis} (pending, address==profile so "
                f"revert is possible), repoint {oid}->{new_id}. Turn-3 user asks to "
                f"re-verify and fix -> a silent wrong write becomes a "
                f"TRANSIENT wrong write (final DB == gold, WEAR=1). Recovery is "
                f"user-signal-induced by design."
            ),
            "_perturb": {
                "knob": "id_reuse",
                "target_tool": "get_order_details",
                "fire_after_n_calls": 1,
                "spec": {
                    "trigger_arg": "order_id",
                    "trigger_arg_value": oid,
                    "collection": "orders",
                    "owner_list_attr": "orders",
                    "referent_id": oid,
                    "referent_new_id": new_id,
                    "distractor_id": dis,
                },
            },
            "_gold_remap": {oid: new_id},
            "description": {
                "purpose": ("Transient wrong-entity write: silent wrong write, "
                            "user-induced recovery, final state == gold."),
                "relevant_policies": None,
                "notes": None,
            },
            "user_scenario": {
                "persona": None,
                "instructions": {
                    "task_instructions": TASK_INSTRUCTIONS,
                    "domain": "retail",
                    "reason_for_call": REASON.format(item=item, addr=addr_str),
                    "known_info": (
                        f"You are {name}. For identity verification, your account "
                        f"email is {u['email']}. Your NEW shipping address (what you "
                        f"want to change the order to) is {addr_str}."
                    ),
                    "unknown_info": None,
                },
            },
            "initial_state": None,
            "evaluation_criteria": {
                "actions": [
                    {"action_id": "a0", "name": "find_user_id_by_email",
                     "arguments": {"email": u["email"]}, "info": None},
                    {"action_id": "r0", "name": "get_user_details",
                     "arguments": {"user_id": uid}, "info": None},
                    {"action_id": "r1", "name": "get_order_details",
                     "arguments": {"order_id": oid}, "info": None},
                    {"action_id": "w0", "name": "modify_pending_order_address",
                     "arguments": {"order_id": oid, "address1": a1, "address2": a2,
                                   "city": city, "state": state, "country": "USA",
                                   "zip": zc},
                     "info": None},
                ],
                "communicate_info": [],
                "nl_assertions": None,
                "reward_basis": ["DB"],
            },
        })
    return tasks


def main() -> None:
    """Build and check the tasks; write them unless --check is given."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    tasks = build()
    users = {t["user_scenario"]["instructions"]["known_info"] for t in tasks}
    assert len(users) == len(tasks), "duplicate users"
    print(f"verified {len(tasks)} recoverable tasks")
    if args.check:
        return
    for t in tasks:
        p = os.path.join(OUT, f"{t['id']}.json")
        with open(p, "w") as f:
            json.dump([t], f, indent=2, ensure_ascii=False)
            f.write("\n")
        print("wrote", os.path.relpath(p, HERE))


if __name__ == "__main__":
    main()
