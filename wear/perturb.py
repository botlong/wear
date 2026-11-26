"""Perturbation layer: changes to the records of a tau2 environment during a rollout.

Entry point: wrap_environment(env, PerturbSchedule), which wraps the environment's
tool calls in place. The perturbation type (`knob`) is one of
  id_reuse  ID-REUSE: the target record moves to a new identifier and, optionally,
            a distractor record takes over the old one. The change is made in the
            database, just before the tool call that follows the binding read;
  append    APPEND: a decoy item is inserted into later responses of the read tool;
  reorder   REORDER: a list in later responses of the read tool is reversed or rotated;
  none      no change.
APPEND and REORDER change only responses, never the database. With fire_at_start,
the ID-REUSE change is applied before the rollout begins (the control condition).
Every step is logged in the controller's `events`.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, field
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Schedule                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class PerturbSchedule:
    """One perturbation: its type, the read tool that triggers it, and type-specific parameters."""

    knob: str  # "id_reuse" | "append" | "reorder" | "none"
    target_tool: str  # read tool whose agent calls count as trigger reads
    fire_after_n_calls: int = 1  # the Nth trigger read is the binding read
    spec: dict = field(default_factory=dict)  # type-specific parameters
    fire_at_start: bool = False  # control condition: apply id_reuse before the rollout

    def __post_init__(self):
        if self.knob not in ("id_reuse", "append", "reorder", "none"):
            raise ValueError(f"unknown knob: {self.knob}")
        if self.fire_after_n_calls < 1:
            raise ValueError("fire_after_n_calls must be >= 1")


# --------------------------------------------------------------------------- #
# Controller                                                                   #
# --------------------------------------------------------------------------- #
class PerturbController:
    """Perturbation state attached to a wrapped environment (env._perturb).

    It counts trigger reads, arms and applies the ID-REUSE change, transforms read
    responses for APPEND and REORDER, and logs each step in `events`.
    """

    def __init__(self, env: Any, schedule: PerturbSchedule):
        self.env = env
        self.sched = schedule
        self._hits = 0  # trigger reads so far
        self._armed = False  # id_reuse: the binding read has happened
        self._fired = False  # the change has been applied
        self._ncalls = 0  # tool calls seen so far (index of the next call)
        self._current_call_id = None  # id of the ToolCall being executed (get_response)
        self.events: list[dict] = []  # log of every perturbation step

    # --- db access ------------------------------------------------------- #
    @property
    def db(self):
        """The environment's agent-side database (env.tools.db), or None."""
        tk = getattr(self.env, "tools", None)
        return getattr(tk, "db", None) if tk is not None else None

    def _log(self, kind: str, **info):
        """Append an event, tagged with the current number of trigger reads."""
        self.events.append({"kind": kind, "hits": self._hits, **info})

    # --- hooks -------------------------------------------------------------- #
    def before(self, tool_name: str, requestor: str, kwargs: dict) -> None:
        """Pre-call hook: apply an armed id_reuse change before this call executes."""
        call_index = self._ncalls
        self._ncalls += 1
        if self._armed and not self._fired and self.sched.knob == "id_reuse":
            try:
                self._fire_id_reuse()
                self._fired = True
                # The change is applied before call `call_index` (0-based over all tool
                # calls that reach the environment). analyze_safeguards.py locates this call in
                # the trajectory by its call_id.
                self._log("fired", call_index=call_index, call_id=self._current_call_id,
                          tool=tool_name)
            except Exception as e:
                print(f"[perturb] id_reuse FAILED: {e}", file=sys.stderr)
                raise

    def after(self, tool_name: str, requestor: str, kwargs: dict, resp: Any) -> Any:
        """Post-call hook: count trigger reads; arm id_reuse or transform the response.

        A trigger read is an agent call to the target tool, restricted to one argument
        value when spec['trigger_arg'] is set; a call that raised never gets here. The
        Nth trigger read arms id_reuse; for append and reorder, the responses of later
        trigger reads are transformed. Returns the (possibly transformed) response.
        """
        knob = self.sched.knob
        if knob == "none":
            return resp

        is_trigger_tool = (
            requestor == "assistant" and tool_name == self.sched.target_tool
        )
        trig_arg = self.sched.spec.get("trigger_arg")
        if is_trigger_tool and trig_arg is not None:
            if kwargs.get(trig_arg) != self.sched.spec.get("trigger_arg_value"):
                is_trigger_tool = False
        if is_trigger_tool:
            self._hits += 1

        reached = self._hits >= self.sched.fire_after_n_calls

        if knob == "id_reuse":
            if is_trigger_tool and reached and not self._armed and not self._fired:
                self._armed = True
                self._log("armed", tool=tool_name)
            return resp

        if is_trigger_tool and self._hits > self.sched.fire_after_n_calls:
            if knob == "reorder":
                return self._xform_reorder(resp)
            if knob == "append":
                return self._xform_append(resp)
        return resp

    # --- id_reuse ---------------------------------------------------------- #
    def _fire_id_reuse(self) -> None:
        """Apply the ID-REUSE change to the database.

        The target record (spec['referent_id']) moves to spec['referent_new_id'], and
        the distractor (spec['distractor_id']), if given, moves to the freed old
        identifier. The owners' lists of identifiers are updated to match. Raises
        if an identifier is missing or already taken.
        """
        s = self.sched.spec
        old = s["referent_id"]
        new = s["referent_new_id"]
        dist = s.get("distractor_id")
        coll_name = s.get("collection", "reservations")
        list_attr = s.get("owner_list_attr", "reservations")

        db = self.db
        if db is None:
            raise RuntimeError("no db on env.tools")
        coll = getattr(db, coll_name)
        if old not in coll:
            raise KeyError(f"referent_id {old} not in db.{coll_name}")
        if new in coll:
            raise KeyError(f"referent_new_id {new} already exists in db.{coll_name}")

        ref = coll.pop(old)
        self._set_id_field(ref, coll_name, new)
        coll[new] = ref
        self._replace_in_owner_list(db, ref, list_attr, old, new)
        self._log("id_reuse.referent_moved", old=old, new=new)

        if dist:
            if dist not in coll:
                raise KeyError(f"distractor_id {dist} not in db.{coll_name}")
            d = coll.pop(dist)
            self._set_id_field(d, coll_name, old)
            coll[old] = d
            self._replace_in_owner_list(db, d, list_attr, dist, old)
            self._log("id_reuse.distractor_moved", from_id=dist, to_id=old)

    @staticmethod
    def _set_id_field(obj: Any, coll_name: str, new_id: str) -> None:
        """Set the record's own identifier field (reservation_id or order_id)."""
        field_name = "reservation_id" if coll_name == "reservations" else "order_id"
        setattr(obj, field_name, new_id)

    @staticmethod
    def _replace_in_owner_list(db, obj, list_attr, old_id, new_id) -> None:
        """Replace old_id with new_id in the owning user's list of record identifiers."""
        owner_id = getattr(obj, "user_id", None)
        if owner_id is None or owner_id not in db.users:
            return
        lst = getattr(db.users[owner_id], list_attr, None)
        if lst is None:
            return
        for i, v in enumerate(lst):
            if v == old_id:
                lst[i] = new_id

    # --- reorder / append -------------------------------------------------- #
    def _get_list_field(self, resp: Any):
        """Return (holder, attr, list) for spec['list_field'] on the response, else None."""
        attr = self.sched.spec.get("list_field")
        if attr is None:
            return None
        if isinstance(resp, list):
            return (None, None, resp)
        holder = resp
        lst = getattr(holder, attr, None)
        if isinstance(lst, list):
            return (holder, attr, lst)
        if isinstance(resp, dict) and isinstance(resp.get(attr), list):
            return (resp, attr, resp[attr])
        return None

    def _xform_reorder(self, resp: Any) -> Any:
        """Return a copy of the response whose list is reversed or rotated (spec['permutation'])."""
        got = self._get_list_field(resp)
        if not got:
            return resp
        holder, attr, lst = got
        resp = copy.deepcopy(resp)  # never mutate objects stored in the db
        got2 = self._get_list_field(resp)
        _, _, lst2 = got2
        perm = self.sched.spec.get("permutation", "reverse")
        if perm == "rotate":
            k = self.sched.spec.get("rotate_by", 1) % max(1, len(lst2))
            lst2[:] = lst2[k:] + lst2[:k]
        else:
            lst2.reverse()
        self._log("reorder", n=len(lst2), perm=perm)
        return resp

    def _xform_append(self, resp: Any) -> Any:
        """Return a copy of the response with spec['append_item'] inserted at spec['append_pos'] (default: end)."""
        got = self._get_list_field(resp)
        if not got:
            return resp
        resp = copy.deepcopy(resp)
        _, _, lst2 = self._get_list_field(resp)
        item = self.sched.spec.get("append_item")
        pos = self.sched.spec.get("append_pos", len(lst2))
        if item is not None:
            lst2.insert(pos, item)
            self._log("append", pos=pos, n=len(lst2))
        return resp


# --------------------------------------------------------------------------- #
# Wrap entry point                                                             #
# --------------------------------------------------------------------------- #
def wrap_environment(env: Any, schedule: PerturbSchedule) -> Any:
    """Wrap env.make_tool_call in place; returns the same env with the controller at env._perturb."""
    controller = PerturbController(env, schedule)
    env._perturb = controller

    # Control condition: apply the change before the rollout starts.
    if schedule.fire_at_start and schedule.knob == "id_reuse":
        controller._fire_id_reuse()
        controller._fired = True
        controller._log("fired_at_start")

    orig_make = env.make_tool_call

    def perturbed_make_tool_call(tool_name, requestor="assistant", **kwargs):
        controller.before(tool_name, requestor, kwargs)
        resp = orig_make(tool_name, requestor=requestor, **kwargs)
        return controller.after(tool_name, requestor, kwargs, resp)

    env.make_tool_call = perturbed_make_tool_call

    # Record which ToolCall is executing, so the `fired` event names the call by its
    # trajectory id (calls whose arguments fail to bind never reach make_tool_call).
    orig_get_response = env.get_response

    def perturbed_get_response(message):
        controller._current_call_id = getattr(message, "id", None)
        try:
            return orig_get_response(message)
        finally:
            controller._current_call_id = None

    env.get_response = perturbed_get_response
    return env
