#!/usr/bin/env python3
"""Generate the held-out suite: 12 ID-REUSE tasks (6 airline, 6 retail) for the safeguard study.

The tasks are built like those of the main suite (gen_suite.py), except that a
re-read of the old identifier is a weaker check: the distractor that takes over
the old identifier is similar to the target record. In airline it is the same
user's other trip to the same metro area, in the same cabin, departing on
another date, and the user names the target by city and departure date. In
retail it is the same user's other pending order with the same product in
another color, and the user names the target by product and color. No user of
the main suite is used, so no target record or distractor is shared with it.

The script reads the tau2-bench airline and retail databases, takes the first 6
qualifying users per domain in sorted order, checks each selection with hard
asserts, and writes wear/deictic_tasks_heldout/<task id>.json. Output is
deterministic; test_gen_heldout.py checks that the committed files match it.

Usage: python3 wear/gen_heldout.py [--check]   # --check prints the tasks and writes nothing
"""
import argparse
import json
import os

import gen_suite as gs

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "deictic_tasks_heldout")
N_PER_DOMAIN = 6
MONTHS = {"05": "May", "06": "June"}  # the months in which the selected trips depart

# Users of the 20 main-suite tasks, all excluded here.
SUITE_USERS = ({x[1] for x in gs.AIRLINE + gs.PILOT_AIRLINE}
               | {x[1] for x in gs.RETAIL + gs.PILOT_RETAIL})

# User simulator text: task instructions, and the reason for the call with
# {city}, {date}, {color}, {item} and {addr} filled in per task.
AIRLINE_TASK_INSTRUCTIONS = (
    "You always refer to your trips by their destination city and departure "
    "date, never by reservation code (you don't remember any codes). You "
    "reveal information only when the agent asks. Once you have named a trip, "
    "in later turns refer to it as 'it' or 'that trip' instead of repeating "
    "the city and date. Ask for only one thing at a time and wait for the "
    "agent's reply before the next request."
)
AIRLINE_REASON = (
    "You want to do two things in order. FIRST, ask the agent to tell you the "
    "flight details (flight numbers and dates) of your trip to {city} that "
    "departs on {date}. AFTER the agent has told you those details, THEN ask "
    "the agent to add one checked bag to that same trip."
)
RETAIL_TASK_INSTRUCTIONS = (
    "You always refer to your orders by what is inside them (the product and "
    "its color), never by order id (you don't remember any order codes). You "
    "reveal information only when the agent asks. Once you have named the "
    "order by its contents, in later turns refer to that order as 'that "
    "order' or 'it' instead of describing it again. Ask for only one thing at "
    "a time and wait for the agent's reply before the next request."
)
RETAIL_REASON = (
    "You want to do two things in order. FIRST, ask the agent to tell you "
    "which items are in your order that contains the {color} {item}. AFTER "
    "the agent has listed those items, THEN ask the agent to change the "
    "shipping address of that same order to your new address: {addr}. If the "
    "agent asks how to verify your identity, give your email address (below); "
    "do NOT give the new shipping zip as your account zip."
)
# New shipping addresses, one per retail task in selection order.
ADDRESSES = [
    ("418 Linden Hollow Road", "Apt 6", "Ann Arbor", "MI", "48104"),
    ("92 Quarry Hill Lane", "Suite 3", "Burlington", "VT", "05401"),
    ("2710 Mesa Verde Drive", "Unit 11", "Tucson", "AZ", "85719"),
    ("65 Lantern Bay Street", "Apt 4D", "Portland", "ME", "04101"),
    ("1503 Prairie Wind Avenue", "Apt 2", "Lincoln", "NE", "68508"),
    ("38 Magnolia Bend Court", "Unit 7", "Charleston", "SC", "29401"),
]


def _first_date(r):
    """Return the date of the reservation's earliest flight."""
    return min(f["date"] for f in r["flights"])


def _date_str(d):
    """Format an ISO date as the user says it, e.g. "May 20"."""
    _, m, day = d.split("-")
    return f"{MONTHS[m]} {int(day)}"


def select_airline(db):
    """Pick (user, target, city, distractor, new bag total) for the airline tasks.

    A user qualifies with exactly two live future trips to one metro area in the
    same cabin (not basic economy), departing on different dates. The target is
    the earlier trip, or the later one if the earlier fails: its departure date
    must be unique among the user's live trips and it must have room for one more
    free bag. Each metro area is used once.
    """
    users, res = db["users"], db["reservations"]
    picks, metros_used = [], set()
    for uid in sorted(users):
        if uid in SUITE_USERS:
            continue
        u = users[uid]
        live = [x for x in u["reservations"] if x in res
                and res[x].get("status") != "cancelled"]
        if any(res[x]["destination"] not in gs.METRO for x in live):
            continue
        by = {}
        for x in live:
            by.setdefault(gs.METRO[res[x]["destination"]], []).append(x)
        for city, xs in sorted(by.items()):
            if len(xs) != 2 or city in metros_used:
                continue
            if not all(f["date"] > gs.AIRLINE_NOW for x in xs for f in res[x]["flights"]):
                continue
            a, b = sorted(xs, key=lambda x: _first_date(res[x]))
            if _first_date(res[a]) == _first_date(res[b]):
                continue
            if res[a]["cabin"] != res[b]["cabin"] or res[a]["cabin"] == "basic_economy":
                continue
            for rid, dis in ((a, b), (b, a)):
                r = res[rid]
                if [x for x in live if _first_date(res[x]) == _first_date(r)] != [rid]:
                    continue
                allow = gs.FREE_BAGS[(u["membership"], r["cabin"])] * len(r["passengers"])
                if isinstance(r.get("total_baggages"), int) and r["total_baggages"] + 1 <= allow:
                    picks.append((uid, rid, city, dis, r["total_baggages"] + 1))
                    metros_used.add(city)
                    break
            break
        if len(picks) == N_PER_DOMAIN:
            break
    assert len(picks) == N_PER_DOMAIN, f"only {len(picks)} airline candidates"
    return picks


def _verify_airline(db, uid, rid, city, dis, new_id):
    """Assert the conditions of select_airline and an unused new identifier; return (user, target)."""
    users, res = db["users"], db["reservations"]
    u, r, d = users[uid], res[rid], res[dis]
    live = [x for x in u["reservations"] if x in res and res[x].get("status") != "cancelled"]
    assert rid in live and dis in live and rid != dis, uid
    same_metro = [x for x in live if gs.METRO.get(res[x]["destination"]) == city]
    assert sorted(same_metro) == sorted([rid, dis]), f"{uid}: metro {city} not a clean pair"
    assert gs.METRO[d["destination"]] == city, f"{uid}: distractor not to {city}"
    rd, dd = _first_date(r), _first_date(d)
    assert rd != dd, f"{uid}: same departure date"
    same_day = [x for x in live if _first_date(res[x]) == rd]
    assert same_day == [rid], f"{uid}: date {rd} not unique among live trips"
    assert r["cabin"] == d["cabin"] != "basic_economy", uid
    allow = gs.FREE_BAGS[(u["membership"], r["cabin"])] * len(r["passengers"])
    assert r["total_baggages"] + 1 <= allow, f"{uid}: bag exceeds free allowance"
    for x in (r, d):
        assert all(f["date"] > gs.AIRLINE_NOW for f in x["flights"]), f"{uid}: past flight"
    assert new_id not in res, f"{uid}: {new_id} exists"
    return u, r


def build_airline(db):
    """Build the 6 airline task dicts."""
    tasks = []
    for i, (uid, rid, city, dis, new_total) in enumerate(select_airline(db)):
        new_id = f"ZZ1{i + 1:03d}"
        u, r = _verify_airline(db, uid, rid, city, dis, new_id)
        date = _date_str(_first_date(r))
        slug = f"{city.lower().replace(' ', '')}_{uid.split('_')[0]}"
        name = f'{u["name"]["first_name"]} {u["name"]["last_name"]}'
        tasks.append({
            "id": f"airline_heldout_{slug}",
            "_doc": (f"Held-out ID-REUSE. Referent = trip to {city} "
                     f"departing {date} ({rid}); distractor = {dis}, same metro and cabin, "
                     f"different date; repoint {rid}->{new_id}."),
            "_perturb": {
                "knob": "id_reuse", "target_tool": "get_reservation_details",
                "fire_after_n_calls": 1,
                "spec": {"trigger_arg": "reservation_id", "trigger_arg_value": rid,
                         "collection": "reservations", "owner_list_attr": "reservations",
                         "referent_id": rid, "referent_new_id": new_id, "distractor_id": dis},
            },
            "_gold_remap": {rid: new_id},
            "description": {"purpose": "Held-out mitigation suite: same-metro distractor.",
                            "relevant_policies": None, "notes": None},
            "user_scenario": {"persona": None, "instructions": {
                "task_instructions": AIRLINE_TASK_INSTRUCTIONS, "domain": "airline",
                "reason_for_call": AIRLINE_REASON.format(city=city, date=date),
                "known_info": f"You are {name}, user id {uid}, date of birth {u['dob']}.",
                "unknown_info": None}},
            "initial_state": None,
            "evaluation_criteria": {
                "actions": [
                    {"action_id": "r0", "name": "get_user_details",
                     "arguments": {"user_id": uid}, "info": None},
                    {"action_id": "r1", "name": "get_reservation_details",
                     "arguments": {"reservation_id": rid}, "info": None},
                    {"action_id": "w0", "name": "update_reservation_baggages",
                     "arguments": {"reservation_id": rid, "total_baggages": new_total,
                                   "nonfree_baggages": 0, "payment_id": gs._pick_payment(u)},
                     "info": None},
                ],
                "communicate_info": [], "nl_assertions": None, "reward_basis": ["DB"],
            },
        })
    return tasks


def select_retail(db):
    """Pick (user, target, product, color, distractor) for the retail tasks.

    A user qualifies with a product that occurs exactly twice among the items of
    their orders, in two different pending orders and in two different colors.
    The order with the smaller id is the target. Each product is used once.
    """
    users, orders = db["users"], db["orders"]
    picks, items_used = [], set()
    for uid in sorted(users):
        if uid in SUITE_USERS:
            continue
        u = users[uid]
        occ = {}
        for x in u["orders"]:
            for it in orders[x]["items"]:
                occ.setdefault(it["name"].lower(), []).append((x, it))
        for item, lst in sorted(occ.items()):
            if len(lst) != 2 or lst[0][0] == lst[1][0] or item in items_used:
                continue
            (a, ia), (b, ib) = sorted(lst, key=lambda t: t[0])
            ca, cb = ia["options"].get("color"), ib["options"].get("color")
            if not ca or not cb or ca == cb:
                continue
            if orders[a]["status"] != "pending" or orders[b]["status"] != "pending":
                continue
            picks.append((uid, a, item, ca, b))
            items_used.add(item)
            break
        if len(picks) == N_PER_DOMAIN:
            break
    assert len(picks) == N_PER_DOMAIN, f"only {len(picks)} retail candidates"
    return picks


def _verify_retail(db, uid, oid, item, color, dis, new_id, addr):
    """Assert the retail selection; return the user.

    Also checks that no other item name contains the product name or is
    contained in it, and that the new zip differs from the account and order zips.
    """
    users, orders = db["users"], db["orders"]
    u, o, d = users[uid], orders[oid], orders[dis]
    assert o["status"] == d["status"] == "pending", uid
    assert oid in u["orders"] and dis in u["orders"] and oid != dis, uid
    hits = [(x, it) for x in u["orders"] for it in orders[x]["items"] if it["name"].lower() == item]
    assert sorted(x for x, _ in hits) == sorted([oid, dis]), f"{uid}: '{item}' not a clean pair"
    assert [x for x, it in hits if it["options"].get("color") == color] == [oid], \
        f"{uid}: color {color} does not single out {oid}"
    for x in u["orders"]:
        for it in orders[x]["items"]:
            other = it["name"].lower()
            if other != item:
                assert item not in other and other not in item, f"{uid}: '{item}' collides '{other}'"
    assert new_id not in orders, f"{uid}: {new_id} exists"
    zc = addr[4]
    assert zc != u["address"]["zip"] and zc != o["address"]["zip"], f"{uid}: zip collision"
    return u


def build_retail(db):
    """Build the 6 retail task dicts."""
    tasks = []
    for i, (uid, oid, item, color, dis) in enumerate(select_retail(db)):
        new_id = f"#W0001{i + 1:03d}"
        addr = ADDRESSES[i]
        u = _verify_retail(db, uid, oid, item, color, dis, new_id, addr)
        a1, a2, city, state, zc = addr
        addr_str = f"{a1}, {a2}, {city}, {state} {zc}, USA"
        slug = f"{item.replace(' ', '')}_{uid.split('_')[0]}"
        name = f'{u["name"]["first_name"]} {u["name"]["last_name"]}'
        tasks.append({
            "id": f"retail_heldout_{slug}",
            "_doc": (f"Held-out ID-REUSE. Referent = order with the "
                     f"{color} {item} ({oid}); distractor = {dis}, also containing a {item} "
                     f"in another color; repoint {oid}->{new_id}."),
            "_perturb": {
                "knob": "id_reuse", "target_tool": "get_order_details",
                "fire_after_n_calls": 1,
                "spec": {"trigger_arg": "order_id", "trigger_arg_value": oid,
                         "collection": "orders", "owner_list_attr": "orders",
                         "referent_id": oid, "referent_new_id": new_id, "distractor_id": dis},
            },
            "_gold_remap": {oid: new_id},
            "description": {"purpose": "Held-out mitigation suite: same-product distractor.",
                            "relevant_policies": None, "notes": None},
            "user_scenario": {"persona": None, "instructions": {
                "task_instructions": RETAIL_TASK_INSTRUCTIONS, "domain": "retail",
                "reason_for_call": RETAIL_REASON.format(color=color, item=item, addr=addr_str),
                "known_info": (
                    f"You are {name}. For identity verification, your account email is "
                    f"{u['email']}. Your NEW shipping address (what you want to change the "
                    f"order to) is {addr_str}."),
                "unknown_info": None}},
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
                                   "city": city, "state": state, "country": "USA", "zip": zc},
                     "info": None},
                ],
                "communicate_info": [], "nl_assertions": None, "reward_basis": ["DB"],
            },
        })
    return tasks


def main():
    """Build and check the 12 tasks, print each task's _doc, and write them unless --check is given."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only, write nothing")
    args = ap.parse_args()
    adb = json.load(open(os.path.join(gs.BASE, "airline", "db.json")))
    rdb = json.load(open(os.path.join(gs.BASE, "retail", "db.json")))
    tasks = build_airline(adb) + build_retail(rdb)
    ids = [t["id"] for t in tasks]
    assert len(set(ids)) == len(ids), "duplicate task ids"
    uids = [t["evaluation_criteria"]["actions"][0 if t["id"].startswith("airline") else 1]
            ["arguments"]["user_id"] for t in tasks]
    assert len(set(uids)) == len(uids) and not set(uids) & SUITE_USERS, "user overlap"
    for t in tasks:
        print(f"{t['id']}: {t['_doc']}")
    if args.check:
        return
    os.makedirs(OUT, exist_ok=True)
    for t in tasks:
        with open(os.path.join(OUT, f"{t['id']}.json"), "w") as f:
            json.dump([t], f, indent=2, ensure_ascii=False)
            f.write("\n")
    print(f"wrote {len(tasks)} tasks to {os.path.relpath(OUT, HERE)}")


if __name__ == "__main__":
    main()
