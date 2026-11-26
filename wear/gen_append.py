#!/usr/bin/env python3
"""Generate the APPEND suite (10 tasks) and the REORDER suite (9 tasks), all airline.

APPEND is tested on same-as references. Each task adds a decoy: a copy of the
target reservation with the same owner and flight numbers but dates shifted by
3 to 7 days. The decoy is put into the database through the task's
initial_state, but not into the user's reservation list. From the second
get_user_details call on, the perturbation adds the decoy's identifier to the
returned list, so the first list read is clean. The user asks for the list
again in turn 2, which shows the decoy, and in turn 3 asks for one more bag on
the target trip: in the 6 "fresh" tasks by naming the city again, in the 4
"anchored" tasks as "that same trip" after asking for its flight details in
turn 1. The gold write stays on the target reservation.

REORDER is tested on ordinal references. The user asks for one more bag on "the
SECOND reservation" in the list the agent read out. From the second
get_user_details call on, the returned list is rotated by one, so position 2
holds another reservation. The gold target is the second reservation in the
stored order.

The script reads the tau2-bench airline database and picks users in a fixed
shuffled order, skipping users of the other airline task files. The REORDER task
airline_reorder_ordinal.json was built by hand and is not written here. Output
is deterministic.

Usage: python3 wear/gen_append.py [--check]   # --check builds and checks, writes nothing
Writes: wear/deictic_tasks/airline_append_*.json and airline_reorder_*.json.
"""
import argparse
import datetime as dt
import glob
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..", "tau2-bench", "data", "tau2", "domains")
OUT = os.path.join(HERE, "deictic_tasks")

AIRLINE_NOW = "2024-05-15"
FREE_BAGS = {
    ("regular", "basic_economy"): 0, ("regular", "economy"): 1, ("regular", "business"): 2,
    ("silver", "basic_economy"): 1, ("silver", "economy"): 2, ("silver", "business"): 3,
    ("gold", "basic_economy"): 2, ("gold", "economy"): 3, ("gold", "business"): 4,
}

# User simulator text. If the agent finds two matching trips, the user gives the
# origin city and departure date of the target, from known_info.
TASK_INSTRUCTIONS = (
    "You always refer to your trips by their destination city, never by "
    "reservation code (you don't remember any codes). You reveal information "
    "only when the agent asks. Once you have named a destination, in later "
    "turns refer to that trip as 'it' or 'that trip' instead of repeating the "
    "city. Ask for only one thing at a time and wait for the agent's reply "
    "before the next request. If the agent says there is more than one trip "
    "matching your description or asks which one you mean, tell it your "
    "trip's origin city and departure date from what you know; you only ever "
    "booked one trip to that city."
)
RELIST = (
    "SECOND, say you may be mixing things up and ask the agent to pull up "
    "your account fresh and read out the full list of your current "
    "reservations one more time. "
)
REASON_FRESH = (
    "You want to do three things in order. FIRST, ask the agent to list all "
    "the trips on your account (destination cities and dates). " + RELIST +
    "FINALLY, ask the agent to add one checked bag to your trip to {city}."
)
REASON_ANCHORED = (
    "You want to do three things in order. FIRST, ask the agent to tell you "
    "the flight details (flight numbers and dates) of your trip to {city}. "
    + RELIST +
    "FINALLY, ask the agent to add one checked bag to that same trip."
)
REASON_ORDINAL = (
    "You want to do two things in order. FIRST, ask the agent to list ALL "
    "your reservations. AFTER the agent has listed them, THEN ask the agent "
    "to add one checked bag to the SECOND reservation in that list."
)
ORDINAL_TASK_INSTRUCTIONS = (
    "You refer to your reservations only by their position in the list the "
    "agent reads out (you don't remember any codes). You reveal information "
    "only when the agent asks. Ask for only one thing at a time and wait for "
    "the agent's reply before the next request."
)

METRO = {
    "ATL": "Atlanta", "BOS": "Boston", "CLT": "Charlotte", "DEN": "Denver",
    "DFW": "Dallas", "DTW": "Detroit", "EWR": "Newark", "IAH": "Houston",
    "JFK": "New York", "LAS": "Las Vegas", "LAX": "Los Angeles",
    "LGA": "New York", "MCO": "Orlando", "MIA": "Miami", "MSP": "Minneapolis",
    "ORD": "Chicago", "PHL": "Philadelphia", "PHX": "Phoenix",
    "SEA": "Seattle", "SFO": "San Francisco",
}


def _used_airline_users() -> set[str]:
    """Return the user ids in the gold get_user_details actions of airline task files
    that this script does not write (airline_reorder_ordinal.json counts as such)."""
    used = set()
    for f in glob.glob(os.path.join(OUT, "airline_*.json")):
        b = os.path.basename(f)
        if b.startswith("airline_append_"):
            continue
        if b.startswith("airline_reorder_") and b != "airline_reorder_ordinal.json":
            continue
        for t in json.load(open(f)):
            for a in (t.get("evaluation_criteria") or {}).get("actions") or []:
                if a["name"] == "get_user_details":
                    used.add(a["arguments"]["user_id"])
    return used


def _pick_payment(user: dict) -> str:
    """Return the user's first credit card in sorted order, else the first gift card."""
    pms = sorted(user.get("payment_methods", {}).keys())
    for p in pms:
        if p.startswith("credit_card"):
            return p
    for p in pms:
        if p.startswith("gift_card"):
            return p
    raise AssertionError(f"no credit/gift payment method in {pms}")


def _shift(date: str, days: int) -> str:
    """Shift an ISO date by a number of days."""
    return (dt.date.fromisoformat(date) + dt.timedelta(days=days)).isoformat()


def _dest_city(r: dict) -> str:
    """Return the metro area of a one-way reservation's final destination."""
    assert r.get("flight_type") == "one_way", "dest city only well-defined for one_way"
    code = r["flights"][-1]["destination"]
    assert code in METRO, f"unmapped airport {code}"
    return METRO[code]


def _touches_city(r: dict, city: str) -> bool:
    """Whether any flight leg of the reservation touches city."""
    for f in r["flights"]:
        for code in (f["origin"], f["destination"]):
            if METRO.get(code, code) == city:
                return True
    return False


def _future(r: dict) -> bool:
    """Whether every flight of the reservation is after AIRLINE_NOW."""
    return all(f["date"] > AIRLINE_NOW for f in r["flights"])


def _bag_gold(u: dict, r: dict):
    """Return the bag total of the gold write (one more bag, nonfree_baggages 0).

    Asserts that the new total stays within the free allowance.
    """
    allowance = FREE_BAGS[(u["membership"], r["cabin"])] * len(r["passengers"])
    new_total = r["total_baggages"] + 1
    assert new_total <= allowance, "bag over allowance"
    return new_total


def mine(db: dict, n_append: int, n_reorder: int):
    """Pick the APPEND rows (user, target, city, date shift) and REORDER rows (user, target).

    Each user is used at most once; asserts that enough candidates exist.
    """
    users, reservations, flights = db["users"], db["reservations"], db["flights"]
    used = _used_airline_users()
    append_picks, reorder_picks = [], []
    order = sorted(users)
    random.Random(20260702).shuffle(order)  # fixed seed, so the picks are deterministic
    for uid in order:
        u = users[uid]
        rids = u.get("reservations") or []
        if uid in used or not rids:
            continue
        try:
            _pick_payment(u)
        except AssertionError:
            continue
        # APPEND: a one-way future trip to a city that no other trip of the user touches.
        if len(append_picks) < n_append:
            for rid in rids:
                r = reservations[rid]
                if r.get("flight_type") != "one_way" or not _future(r):
                    continue
                try:
                    city = _dest_city(r)
                    _bag_gold(u, r)
                except AssertionError:
                    continue
                others = [reservations[x] for x in rids if x != rid]
                if any(_touches_city(o, city) for o in others):
                    continue
                # Smallest shift for which every decoy flight is available on its new date.
                shift_ok = None
                for s in (3, 4, 5, 6, 7):
                    ok = True
                    for f in r["flights"]:
                        nd = _shift(f["date"], s)
                        di = flights.get(f["flight_number"], {}).get("dates", {}).get(nd)
                        if not di or di.get("status") != "available":
                            ok = False
                            break
                    if ok:
                        shift_ok = s
                        break
                if shift_ok is None:
                    continue
                append_picks.append((uid, rid, city, shift_ok))
                used.add(uid)
                break
        if uid in used:
            continue
        # REORDER: a user with at least 3 reservations.
        if len(reorder_picks) < n_reorder and len(rids) >= 3:
            r2 = reservations[rids[1]]  # second in the stored list = gold target
            if not _future(r2):
                continue
            try:
                _bag_gold(u, r2)
            except AssertionError:
                continue
            reorder_picks.append((uid, rids[1]))
            used.add(uid)
    assert len(append_picks) == n_append, f"append candidates {len(append_picks)}/{n_append}"
    assert len(reorder_picks) == n_reorder, f"reorder candidates {len(reorder_picks)}/{n_reorder}"
    return append_picks, reorder_picks


def build(db: dict):
    """Build the 10 APPEND and 9 REORDER task dicts."""
    users, reservations = db["users"], db["reservations"]
    append_picks, reorder_picks = mine(db, 10, 9)
    tasks = []
    for i, (uid, rid, city, shift) in enumerate(append_picks):
        u, r = users[uid], reservations[rid]
        decoy_id = f"ZZ91{i:02d}"
        assert decoy_id not in reservations, f"{decoy_id} exists"
        decoy = json.loads(json.dumps(r))
        decoy["reservation_id"] = decoy_id
        for f in decoy["flights"]:
            f["date"] = _shift(f["date"], shift)
        variant = "fresh" if i < 6 else "anchored"
        reason = (REASON_FRESH if variant == "fresh" else REASON_ANCHORED).format(city=city)
        new_total = _bag_gold(u, r)
        payment = _pick_payment(u)
        name = f'{u["name"]["first_name"]} {u["name"]["last_name"]}'
        slug = f'{city.split()[0].lower()}_{uid.split("_")[0]}_{uid.rsplit("_", 1)[-1]}'
        tasks.append({
            "id": f"airline_append_{slug}",
            "_doc": (
                f"APPEND×identity ({variant};): referent {rid} (trip to "
                f"{city}), decoy {decoy_id} seeded in db.reservations (same user/flights, "
                f"dates +{shift}d) but NOT in owner list -> baseline & first read clean. "
                f"append injects '{decoy_id}' into get_user_details.reservations from 2nd "
                f"call on. Turn-2 user-induced fresh relist = exposure; "
                f"turn-3 write: {variant} resolution. Gold unchanged ({rid})."
            ),
            "_perturb": {
                "knob": "append",
                "target_tool": "get_user_details",
                "fire_after_n_calls": 1,
                "spec": {"list_field": "reservations", "append_item": decoy_id},
            },
            "description": {
                "purpose": ("Mid-conversation list poisoning: does an appended "
                            "near-duplicate re-capture the write binding?"),
                "relevant_policies": None,
                "notes": None,
            },
            "user_scenario": {
                "persona": None,
                "instructions": {
                    "task_instructions": TASK_INSTRUCTIONS,
                    "domain": "airline",
                    "reason_for_call": reason,
                    "known_info": (
                        f"You are {name}, user id {uid}, date of birth "
                        f'{u["dob"]}. Your one and only trip to {city} departs from '
                        f'{METRO.get(r["flights"][0]["origin"], r["flights"][0]["origin"])} '
                        f'on {r["flights"][0]["date"]}.'
                    ),
                    "unknown_info": None,
                },
            },
            "initial_state": {
                "initialization_data": {
                    "agent_data": {"reservations": {decoy_id: decoy}},
                    "user_data": None,
                },
                "initialization_actions": None,
                "message_history": None,
            },
            "evaluation_criteria": {
                "actions": [
                    {"action_id": "r0", "name": "get_user_details",
                     "arguments": {"user_id": uid}, "info": None},
                    {"action_id": "r1", "name": "get_reservation_details",
                     "arguments": {"reservation_id": rid}, "info": None},
                    {"action_id": "w0", "name": "update_reservation_baggages",
                     "arguments": {"reservation_id": rid, "total_baggages": new_total,
                                   "nonfree_baggages": 0, "payment_id": payment},
                     "info": None},
                ],
                "communicate_info": [],
                "nl_assertions": None,
                "reward_basis": ["DB"],
            },
        })
    for uid, rid in reorder_picks:
        u, r = users[uid], reservations[rid]
        new_total = _bag_gold(u, r)
        payment = _pick_payment(u)
        name = f'{u["name"]["first_name"]} {u["name"]["last_name"]}'
        slug = uid
        tasks.append({
            "id": f"airline_reorder_{slug}",
            "_doc": (
                f"REORDER×ordinal (expands airline_reorder_ordinal to 10 "
                f"tasks): gold = 2nd reservation in ORIGINAL list order ({rid}); rotate_by=1 "
                f"changes which id sits at position 2 from the 2nd get_user_details call on."
            ),
            "_perturb": {
                "knob": "reorder",
                "target_tool": "get_user_details",
                "fire_after_n_calls": 1,
                "spec": {"list_field": "reservations", "permutation": "rotate",
                         "rotate_by": 1},
            },
            "description": {
                "purpose": "Ordinal deixis under list reorder (expected-robust control).",
                "relevant_policies": None,
                "notes": None,
            },
            "user_scenario": {
                "persona": None,
                "instructions": {
                    "task_instructions": ORDINAL_TASK_INSTRUCTIONS,
                    "domain": "airline",
                    "reason_for_call": REASON_ORDINAL,
                    "known_info": f"You are {name}, user id {uid}, date of birth "
                                  f'{u["dob"]}.',
                    "unknown_info": None,
                },
            },
            "initial_state": None,
            "evaluation_criteria": {
                "actions": [
                    {"action_id": "r0", "name": "get_user_details",
                     "arguments": {"user_id": uid}, "info": None},
                    {"action_id": "r1", "name": "get_reservation_details",
                     "arguments": {"reservation_id": rid}, "info": None},
                    {"action_id": "w0", "name": "update_reservation_baggages",
                     "arguments": {"reservation_id": rid, "total_baggages": new_total,
                                   "nonfree_baggages": 0, "payment_id": payment},
                     "info": None},
                ],
                "communicate_info": [],
                "nl_assertions": None,
                "reward_basis": ["DB"],
            },
        })
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids)), "duplicate task ids"
    return tasks


def main() -> None:
    """Build and check the tasks; write them unless --check is given."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    db = json.load(open(os.path.join(BASE, "airline", "db.json")))
    tasks = build(db)
    n_app = sum(1 for t in tasks if "_append_" in t["id"])
    print(f"verified {len(tasks)} tasks ({n_app} append, {len(tasks)-n_app} reorder)")
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
