#!/usr/bin/env python3
"""Generate the main suite: 20 ID-REUSE tasks on the tau2-bench airline and retail domains.

In every task the user names the target record by its content (the trip's
destination city, or a product in the order) and later refers back to it as
"it", "that trip" or "that order". The agent reads the target record (the
binding read) and then makes one gold write: one more checked bag (airline) or a
new shipping address (retail). In the perturbed condition, ID-REUSE is applied
right after the binding read: the target record moves to a new identifier and a
distractor, another record of the same user, takes over the old identifier.
`_gold_remap` maps the old identifier of the gold write to the new one.

The script reads the tau2-bench airline and retail databases, checks every
selection with hard asserts, and writes 16 tasks (7 airline, 9 retail) to
wear/deictic_tasks/<task id>.json. The other 4 tasks of the suite (3 airline,
1 retail) were built by hand; the script checks them with the same asserts and
never rewrites them. Output is deterministic.

Usage: python3 wear/gen_suite.py [--check]   # --check checks all 20 tasks and writes nothing
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..", "tau2-bench", "data", "tau2", "domains")
OUT = os.path.join(HERE, "deictic_tasks")

# Airport code -> metro area, used to match the city the user names.
METRO = {
    "LGA": "New York", "JFK": "New York", "EWR": "New York",
    "DFW": "Dallas", "DAL": "Dallas",
    "IAH": "Houston", "HOU": "Houston",
    "ORD": "Chicago", "MDW": "Chicago",
    "SFO": "San Francisco", "OAK": "San Francisco", "SJC": "San Francisco",
    "LAX": "Los Angeles", "BUR": "Los Angeles", "SNA": "Los Angeles",
    "MIA": "Miami", "FLL": "Miami",
    "DCA": "Washington", "IAD": "Washington", "BWI": "Washington",
    "BOS": "Boston", "SEA": "Seattle", "MSP": "Minneapolis",
    "DEN": "Denver", "PHX": "Phoenix", "ATL": "Atlanta", "DTW": "Detroit",
    "CLT": "Charlotte", "LAS": "Las Vegas", "MCO": "Orlando",
    "PHL": "Philadelphia", "PHV": "PHV",
}
# Free checked bags per passenger by (membership, cabin), as in the tau2 airline
# policy. The gold write stays within this allowance, so nonfree_baggages is 0.
FREE_BAGS = {
    ("regular", "basic_economy"): 0, ("regular", "economy"): 1, ("regular", "business"): 2,
    ("silver", "basic_economy"): 1, ("silver", "economy"): 2, ("silver", "business"): 3,
    ("gold", "basic_economy"): 2, ("gold", "economy"): 3, ("gold", "business"): 4,
}

# Generated airline tasks. Columns: slug (task id suffix), user_id, target
# reservation, city the user names, distractor reservation, new identifier
# of the target.
AIRLINE = [
    ("boston", "mei_hernandez_8984", "R9QDGB", "Boston", "79G8A9", "ZZ0004"),
    ("chicago", "aarav_nguyen_1055", "H0S6OC", "Chicago", "MYKO5S", "ZZ0005"),
    ("seattle", "liam_santos_5621", "EFIAC5", "Seattle", "ZZSA4W", "ZZ0006"),
    ("minneapolis", "harper_kovacs_3082", "AYVIYO", "Minneapolis", "Q32WJB", "ZZ0007"),
    ("sanfrancisco", "yara_silva_1929", "PAW5FS", "San Francisco", "N9V9VX", "ZZ0008"),
    ("denver", "raj_garcia_4690", "4NQCM5", "Denver", "NPMZCA", "ZZ0009"),
    ("phoenix", "lucas_thomas_9373", "6S9Q4I", "Phoenix", "5KCJWY", "ZZ0010"),
]

# Current date of the tau2 airline policy ("2024-05-15 15:00:00 EST"); every
# flight of the target and the distractor must be later.
AIRLINE_NOW = "2024-05-15"
# Generated retail tasks. Columns: slug, user_id, target order, product the user
# names, distractor order, new identifier of the target, new shipping address
# (address1, address2, city, state, zip).
RETAIL = [
    ("officechair", "ivan_santos_6635", "#W8770097", "office chair", "#W5183325", "#W0000002",
     ("481 Maple Grove Lane", "Suite 12", "Austin", "TX", "78745")),
    ("espresso", "olivia_jackson_1219", "#W2090453", "espresso machine", "#W5663445", "#W0000003",
     ("1968 Willow Creek Drive", "Apt 7B", "Portland", "OR", "97218")),
    ("grill", "yusuf_ahmed_6232", "#W7756209", "grill", "#W7007896", "#W0000004",
     ("305 Cedar Hollow Road", "Unit 5", "Nashville", "TN", "37211")),
    ("airpurifier", "emma_smith_8564", "#W3614011", "air purifier", "#W2417020", "#W0000005",
     ("77 Birchwood Court", "Apt 21", "Madison", "WI", "53704")),
    ("electrickettle", "yusuf_khan_7091", "#W3579467", "electric kettle", "#W1787190", "#W0000006",
     ("1204 Sunset Bluff Avenue", "Apt 9", "San Diego", "CA", "92109")),
    ("yogamat", "liam_lee_5696", "#W2624389", "yoga mat", "#W7208030", "#W0000007",
     ("58 Juniper Way", "Unit 3", "Boulder", "CO", "80304")),
    ("teakettle", "daiki_johnson_9523", "#W1436802", "tea kettle", "#W5282037", "#W0000008",
     ("932 Harbor View Street", "Apt 14", "Savannah", "GA", "31401")),
    ("sunglasses", "omar_silva_7446", "#W9728773", "sunglasses", "#W9673784", "#W0000009",
     ("210 Foxglove Terrace", "Unit 8", "Raleigh", "NC", "27604")),
    ("actioncamera", "amelia_patel_7834", "#W2079779", "action camera", "#W9077472", "#W0000010",
     ("664 Copper Ridge Road", "Apt 2C", "Reno", "NV", "89509")),
]

# The 4 hand-built tasks of the main suite. Their files already exist in
# wear/deictic_tasks/; they are checked with the same asserts and never rewritten.
# Columns as in AIRLINE and RETAIL.
PILOT_AIRLINE = [
    ("dallas", "mia_li_3668", "NO6JO3", "Dallas", "HKEG34", "ZZ0001"),
    ("miami", "aarav_jackson_2879", "QOOWS8", "Miami", "QQ69B9", "ZZ0002"),
    ("houston", "daiki_jackson_9549", "H6CD72", "Houston", "SWY6D2", "ZZ0003"),
]
PILOT_RETAIL = [
    # No address column: the zip checks run on the address in the task file.
    ("laptop", "ava_lopez_2676", "#W8327915", "laptop", "#W5911003", "#W0000001"),
]

# User simulator text: task instructions, and the reason for the call with
# {city}, {item} and {addr} filled in per task.
AIRLINE_TASK_INSTRUCTIONS = (
    "You always refer to your trips by their destination city, never by "
    "reservation code (you don't remember any codes). You reveal information "
    "only when the agent asks. Once you have named a destination, in later "
    "turns refer to that trip as 'it' or 'that trip' instead of repeating the "
    "city. Ask for only one thing at a time and wait for the agent's reply "
    "before the next request."
)
AIRLINE_REASON = (
    "You want to do two things in order. FIRST, ask the agent to tell you the "
    "flight details (flight numbers and dates) of your trip to {city}. AFTER "
    "the agent has told you those details, THEN ask the agent to add one "
    "checked bag to that same trip."
)
RETAIL_TASK_INSTRUCTIONS = (
    "You always refer to your orders by what is inside them (their products), "
    "never by order id (you don't remember any order codes). You reveal "
    "information only when the agent asks. Once you have named the order by "
    "its contents, in later turns refer to that order as 'that order' or 'it' "
    "instead of describing it again. Ask for only one thing at a time and "
    "wait for the agent's reply before the next request."
)
RETAIL_REASON = (
    "You want to do two things in order. FIRST, ask the agent to tell you "
    "which items are in your order that contains the {item}. AFTER the agent "
    "has listed those items, THEN ask the agent to change the shipping "
    "address of that same order to your new address: {addr}. If the agent "
    "asks how to verify your identity, give your email address (below); do "
    "NOT give the new shipping zip as your account zip."
)


def _pick_payment(user):
    """Return the user's first credit card, else first gift card (the gold write's payment_id)."""
    pms = list(user.get("payment_methods", {}).keys())
    for pref in ("credit_card", "gift_card"):
        for p in pms:
            if p.startswith(pref):
                return p
    raise AssertionError(f"no credit/gift payment method in {pms}")


def _verify_airline_selection(users, res, slug, uid, rid, city, dis, new_id):
    """Assert that an airline row is a valid main-suite task; return (user, target, new bag total).

    The target and the distractor are live reservations of the user in the same
    cabin (not basic economy), all their flights are after AIRLINE_NOW, the target
    is the user's only live trip to that metro area, the new identifier is unused,
    and one more bag stays within the free allowance.
    """
    u, r = users[uid], res[rid]
    live = [x for x in u["reservations"]
            if x in res and res[x].get("status") != "cancelled"]
    assert rid in live, f"{slug}: referent {rid} missing/cancelled"
    assert r["cabin"] != "basic_economy", f"{slug}: basic_economy referent"
    for x in live:
        assert res[x]["destination"] in METRO, \
            f"{slug}: destination {res[x]['destination']} missing from METRO map"
    metros = [METRO[res[x]["destination"]] for x in live]
    assert METRO[r["destination"]] == city, \
        f"{slug}: referent dest {r['destination']} is not {city}"
    assert metros.count(city) == 1, f"{slug}: metro '{city}' ambiguous {metros}"
    d = res[dis]
    assert dis in live and dis != rid, f"{slug}: bad distractor {dis}"
    assert d["cabin"] == r["cabin"], f"{slug}: distractor cabin mismatch"
    assert new_id not in res, f"{slug}: {new_id} already exists"
    for x, who in ((rid, "referent"), (dis, "distractor")):
        for f in res[x]["flights"]:
            assert f["date"] > AIRLINE_NOW, \
                f"{slug}: {who} {x} flight {f['date']} not after {AIRLINE_NOW}"
    allowance = FREE_BAGS[(u["membership"], r["cabin"])] * len(r["passengers"])
    assert isinstance(r.get("total_baggages"), int), \
        f"{slug}: total_baggages missing/non-int on {rid}"
    new_total = r["total_baggages"] + 1
    assert new_total <= allowance, \
        f"{slug}: {new_total} bags exceeds free allowance {allowance}"
    return u, r, new_total


def build_airline():
    """Build the 7 generated airline tasks."""
    db = json.load(open(os.path.join(BASE, "airline", "db.json")))
    users, res = db["users"], db["reservations"]
    tasks = []
    for slug, uid, rid, city, dis, new_id in AIRLINE:
        u, r, new_total = _verify_airline_selection(
            users, res, slug, uid, rid, city, dis, new_id)
        payment = _pick_payment(u)
        name = f'{u["name"]["first_name"]} {u["name"]["last_name"]}'
        task = {
            "id": f"airline_idreuse_{slug}",
            "_doc": (
                f"ID-REUSE × content/temporal deixis (suite; mirrors airline_idreuse_dallas). "
                f"Referent = trip to {city} ({rid}), distractor = {dis} (same cabin "
                f"{r['cabin']}), repoint {rid}->{new_id}."
            ),
            "_perturb": {
                "knob": "id_reuse",
                "target_tool": "get_reservation_details",
                "fire_after_n_calls": 1,
                "spec": {
                    "trigger_arg": "reservation_id",
                    "trigger_arg_value": rid,
                    "collection": "reservations",
                    "owner_list_attr": "reservations",
                    "referent_id": rid,
                    "referent_new_id": new_id,
                    "distractor_id": dis,
                },
            },
            "_gold_remap": {rid: new_id},
            "description": {
                "purpose": "Measure wrong-entity binding under id-reuse for content/temporal deixis.",
                "relevant_policies": None,
                "notes": None,
            },
            "user_scenario": {
                "persona": None,
                "instructions": {
                    "task_instructions": AIRLINE_TASK_INSTRUCTIONS,
                    "domain": "airline",
                    "reason_for_call": AIRLINE_REASON.format(city=city),
                    "known_info": f"You are {name}, user id {uid}, date of birth {u['dob']}.",
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
                     "arguments": {"reservation_id": rid,
                                   "total_baggages": new_total,
                                   "nonfree_baggages": 0,
                                   "payment_id": payment},
                     "info": None},
                ],
                "communicate_info": [],
                "nl_assertions": None,
                "reward_basis": ["DB"],
            },
        }
        tasks.append(task)
    return tasks


def _verify_retail_selection(users, orders, slug, uid, oid, item, dis, new_id):
    """Assert that a retail row is a valid main-suite task; return (user, target order).

    The target and the distractor are pending orders of the user, the named
    product occurs in exactly one item of the user's orders (in the target), no
    other item name contains it or is contained in it, and the new identifier is
    unused. The callers check the zip codes.
    """
    u, o = users[uid], orders[oid]
    assert o["status"] == "pending", f"{slug}: referent {oid} not pending"
    assert orders[dis]["status"] == "pending", f"{slug}: distractor {dis} not pending"
    assert dis in u["orders"] and oid in u["orders"] and dis != oid, f"{slug}: ids"
    counts = {}
    for x in u["orders"]:
        for it in orders[x]["items"]:
            counts[it["name"].lower()] = counts.get(it["name"].lower(), 0) + 1
    assert counts.get(item, 0) == 1, f"{slug}: item '{item}' not unique ({counts.get(item)})"
    assert any(it["name"].lower() == item for it in o["items"]), \
        f"{slug}: item '{item}' not in referent {oid}"
    for x in u["orders"]:
        for it in orders[x]["items"]:
            other = it["name"].lower()
            if other == item:
                continue
            assert item not in other and other not in item, \
                f"{slug}: phrase '{item}' collides with item '{other}' in {x}"
    assert new_id not in orders, f"{slug}: {new_id} already exists"
    return u, o


def build_retail():
    """Build the 9 generated retail tasks."""
    db = json.load(open(os.path.join(BASE, "retail", "db.json")))
    users, orders = db["users"], db["orders"]
    tasks = []
    for slug, uid, oid, item, dis, new_id, addr in RETAIL:
        u, o = _verify_retail_selection(
            users, orders, slug, uid, oid, item, dis, new_id)
        a1, a2, city, state, zc = addr
        assert zc != u["address"]["zip"], f"{slug}: new zip equals account zip"
        assert zc != o["address"]["zip"], f"{slug}: new zip equals order zip"
        addr_str = f"{a1}, {a2}, {city}, {state} {zc}, USA"
        name = f'{u["name"]["first_name"]} {u["name"]["last_name"]}'
        task = {
            "id": f"retail_idreuse_{slug}",
            "_doc": (
                f"ID-REUSE × content deixis (suite; mirrors retail_idreuse_laptop). "
                f"Referent = order with the {item} ({oid}), distractor = {dis} "
                f"(pending), repoint {oid}->{new_id}."
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
                "purpose": "Measure wrong-entity binding under id-reuse for content deixis.",
                "relevant_policies": None,
                "notes": None,
            },
            "user_scenario": {
                "persona": None,
                "instructions": {
                    "task_instructions": RETAIL_TASK_INSTRUCTIONS,
                    "domain": "retail",
                    "reason_for_call": RETAIL_REASON.format(item=item, addr=addr_str),
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
        }
        tasks.append(task)
    return tasks


def _load_pilot(domain, slug):
    """Load a hand-built task file (one task per file)."""
    path = os.path.join(OUT, f"{domain}_idreuse_{slug}.json")
    tasks = json.load(open(path))
    assert len(tasks) == 1, path
    return tasks[0]


def _check_pilot_wiring(t, domain, slug, rid, dis, new_id):
    """Assert that a hand-built task's id, perturbation spec and _gold_remap match its row."""
    assert t["id"] == f"{domain}_idreuse_{slug}", t["id"]
    p = t["_perturb"]
    spec = p["spec"]
    assert p["knob"] == "id_reuse", slug
    assert spec["referent_id"] == rid == spec["trigger_arg_value"], slug
    assert spec["referent_new_id"] == new_id, slug
    assert spec["distractor_id"] == dis, slug
    assert t["_gold_remap"] == {rid: new_id}, slug


def verify_pilots():
    """Check the 4 hand-built tasks with the shared asserts and on their gold actions; return their count."""
    adb = json.load(open(os.path.join(BASE, "airline", "db.json")))
    rdb = json.load(open(os.path.join(BASE, "retail", "db.json")))
    for slug, uid, rid, city, dis, new_id in PILOT_AIRLINE:
        u, r, new_total = _verify_airline_selection(
            adb["users"], adb["reservations"], slug, uid, rid, city, dis, new_id)
        t = _load_pilot("airline", slug)
        _check_pilot_wiring(t, "airline", slug, rid, dis, new_id)
        acts = t["evaluation_criteria"]["actions"]
        assert any(a["name"] == "get_user_details"
                   and a["arguments"]["user_id"] == uid for a in acts), \
            f"{slug}: gold get_user_details({uid}) missing"
        w = acts[-1]["arguments"]
        assert w["reservation_id"] == rid, slug
        assert w["nonfree_baggages"] == 0, f"{slug}: gold nonfree != 0"
        assert w["total_baggages"] == new_total, \
            f"{slug}: gold baggages {w['total_baggages']} != current+1 ({new_total})"
        assert w["payment_id"] in u["payment_methods"], \
            f"{slug}: payment {w['payment_id']} not on user"
    for slug, uid, oid, item, dis, new_id in PILOT_RETAIL:
        u, o = _verify_retail_selection(
            rdb["users"], rdb["orders"], slug, uid, oid, item, dis, new_id)
        t = _load_pilot("retail", slug)
        _check_pilot_wiring(t, "retail", slug, oid, dis, new_id)
        acts = t["evaluation_criteria"]["actions"]
        assert any(a["name"] == "get_user_details"
                   and a["arguments"]["user_id"] == uid for a in acts), \
            f"{slug}: gold get_user_details({uid}) missing"
        w = acts[-1]["arguments"]
        assert w["order_id"] == oid, slug
        assert w["zip"] != u["address"]["zip"], f"{slug}: new zip equals account zip"
        assert w["zip"] != o["address"]["zip"], f"{slug}: new zip equals order zip"
    return len(PILOT_AIRLINE) + len(PILOT_RETAIL)


def main():
    """Build and check all 20 tasks; write the 16 generated ones unless --check is given."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only, write nothing")
    args = ap.parse_args()
    tasks = build_airline() + build_retail()
    n_pilot = verify_pilots()
    ids = [t["id"] for t in tasks]
    assert len(set(ids)) == len(ids), "duplicate task ids"
    users = [t["user_scenario"]["instructions"]["known_info"] for t in tasks]
    assert len(set(users)) == len(users), "duplicate users in suite"
    news = [list(t["_gold_remap"].values())[0] for t in tasks]
    assert len(set(news)) == len(news), "duplicate referent_new ids"
    # Users, targets and new identifiers are unique across all 20 rows.
    sel = AIRLINE + PILOT_AIRLINE + RETAIL + PILOT_RETAIL
    for idx, label in ((1, "user"), (2, "referent"), (5, "referent_new")):
        vals = [x[idx] for x in sel]
        assert len(set(vals)) == len(vals), f"duplicate {label} across all 20"
    n_air = sum(t["id"].startswith("airline") for t in tasks) + len(PILOT_AIRLINE)
    n_ret = sum(t["id"].startswith("retail") for t in tasks) + len(PILOT_RETAIL)
    print(f"verified {len(tasks) + n_pilot} tasks ({n_air} airline, {n_ret} "
          f"retail): {len(tasks)} emitted by this script + {n_pilot} pilot "
          f"originals re-verified in place")
    if args.check:
        return
    for t in tasks:
        path = os.path.join(OUT, f"{t['id']}.json")
        with open(path, "w") as f:
            json.dump([t], f, indent=2, ensure_ascii=False)
            f.write("\n")
        print("wrote", os.path.relpath(path, HERE))


if __name__ == "__main__":
    main()
