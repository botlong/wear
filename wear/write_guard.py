"""Tool-level write guards: the CAS guard and the identity guard.

Entry point: wrap_write_guard(env, write_tools, mode). The guard wraps the
environment's tool calls; it sees the agent's calls and the stored records, but
neither the user's request nor the gold target.

Mode "cas" (compare-and-set). For every record the agent reads (a response
carrying reservation_id / order_id), the guard stores a fingerprint of the
contents returned. A later agent write that targets the same identifier is
rejected with a tool error ("Precondition failed: ...") if the record now stored
under that identifier differs from the fingerprint, i.e. the write is
conditional on the version the agent last saw. Writes to identifiers the agent
never read pass unchanged, and a write that passes refreshes the fingerprint, so
follow-up writes to the same record are not rejected.

Mode "identity" adds a check on record identity: the guard also binds each
identifier to the stored record object the agent first read under it. The
object stands in for an immutable internal key that stays with a record when
its public identifier is reused. A write is rejected whenever the identifier now
names a different record, even after the agent has re-read it; compare-and-set
alone passes that write, because the re-read refreshes the fingerprint.

Apply this wrapper *before* wrap_environment (perturb.py), so the perturbation
wrapper sits outside it: an armed ID-REUSE change is then applied before the
guard checks the write.
"""

from __future__ import annotations

import json
from typing import Any

ID_FIELDS = {"reservation_id": "reservations", "order_id": "orders"}
LABELS = {"reservations": "reservation", "orders": "order"}
MODES = ("cas", "identity")


def _fingerprint(obj: Any, id_field: str) -> str:
    """Canonical JSON of a record's contents, without its identifier field."""
    d = obj.model_dump() if hasattr(obj, "model_dump") else dict(obj)
    d.pop(id_field, None)
    return json.dumps(d, sort_keys=True, default=str)


class WriteGuard:
    """Per-rollout guard state.

    `seen` holds the fingerprint of each record as the agent last read it, `bound`
    (identity mode) the stored object first read under each identifier, and
    `events` one entry per checked write.
    """

    def __init__(self, env: Any, write_tools: set[str], mode: str = "cas"):
        assert mode in MODES, mode
        self.env = env
        self.write_tools = set(write_tools)
        self.mode = mode
        self.seen: dict[tuple[str, str], str] = {}
        # Identity mode: the stored object first read under each identifier. Strong
        # references keep the objects alive, so their identity cannot be recycled.
        self.bound: dict[tuple[str, str], Any] = {}
        self.events: list[dict] = []

    def _collection(self, coll: str):
        """The named collection of the agent-side database, or None."""
        db = getattr(getattr(self.env, "tools", None), "db", None)
        return getattr(db, coll, None) if db is not None else None

    def _target(self, kwargs: dict) -> tuple[str, str, str] | None:
        """(id field, collection, identifier) of the record a call targets, or None."""
        for id_field, coll in ID_FIELDS.items():
            if kwargs.get(id_field) is not None:
                return id_field, coll, str(kwargs[id_field])
        return None

    def record_read(self, resp: Any) -> None:
        """Store the fingerprint of the record a read response carries.

        In identity mode, an identifier read for the first time is also bound to the
        object now stored under it.
        """
        for id_field, coll in ID_FIELDS.items():
            eid = getattr(resp, id_field, None)
            if eid is None and isinstance(resp, dict):
                eid = resp.get(id_field)
            if eid is not None:
                key = (coll, str(eid))
                self.seen[key] = _fingerprint(resp, id_field)
                if self.mode == "identity" and key not in self.bound:
                    store = self._collection(coll)
                    cur = store.get(str(eid)) if store is not None else None
                    if cur is not None:
                        self.bound[key] = cur
                return

    def check_write(self, tool_name: str, kwargs: dict) -> None:
        """Raise ValueError ("Precondition failed: ...") if a write to a read record must be rejected.

        A write without a target identifier, or to an identifier never read or no
        longer stored, passes unchecked. Every other write is logged as
        write_allowed, write_rejected or write_rejected_identity.
        """
        tgt = self._target(kwargs)
        if tgt is None:
            return
        id_field, coll, eid = tgt
        if (coll, eid) not in self.seen:
            return
        store = self._collection(coll)
        cur = store.get(eid) if store is not None else None
        if cur is None:
            return  # dangling id: let the tool raise its own not-found error
        if self.mode == "identity" and (coll, eid) in self.bound \
                and cur is not self.bound[(coll, eid)]:
            self.events.append({"kind": "write_rejected_identity", "tool": tool_name,
                                "collection": coll, "id": eid})
            raise ValueError(
                f"Precondition failed: {LABELS[coll]} {eid} now refers to a different "
                f"{LABELS[coll]} than the one you retrieved earlier. Identify the "
                f"intended {LABELS[coll]} again from the user's current "
                f"{coll} before modifying it.")
        if _fingerprint(cur, id_field) != self.seen[(coll, eid)]:
            self.events.append({"kind": "write_rejected", "tool": tool_name,
                                "collection": coll, "id": eid})
            raise ValueError(
                f"Precondition failed: {LABELS[coll]} {eid} has changed since it "
                f"was last read. Retrieve it again before modifying it.")
        self.events.append({"kind": "write_allowed", "tool": tool_name,
                            "collection": coll, "id": eid})

    def refresh_after_write(self, kwargs: dict) -> None:
        """After a write that ran without error, store the record's new fingerprint.

        In identity mode, a bound identifier is re-bound to the object now stored
        under it (the write has just passed the identity check).
        """
        tgt = self._target(kwargs)
        if tgt is None:
            return
        id_field, coll, eid = tgt
        store = self._collection(coll)
        cur = store.get(eid) if store is not None else None
        if cur is not None:
            self.seen[(coll, eid)] = _fingerprint(cur, id_field)
            if self.mode == "identity" and (coll, eid) in self.bound:
                self.bound[(coll, eid)] = cur  # the write passed the identity check


def wrap_write_guard(env: Any, write_tools: set[str], mode: str = "cas") -> Any:
    """Wrap env.make_tool_call in place; the guard is attached at env._write_guard.

    Only agent calls are guarded: calls to a write tool are checked before they run,
    and every other agent call is treated as a read.
    """
    guard = WriteGuard(env, write_tools, mode)
    env._write_guard = guard
    orig_make = env.make_tool_call

    def guarded_make_tool_call(tool_name, requestor="assistant", **kwargs):
        is_write = requestor == "assistant" and tool_name in guard.write_tools
        if is_write:
            guard.check_write(tool_name, kwargs)
        resp = orig_make(tool_name, requestor=requestor, **kwargs)
        if requestor == "assistant":
            if is_write:
                guard.refresh_after_write(kwargs)
            else:
                guard.record_read(resp)
        return resp

    env.make_tool_call = guarded_make_tool_call
    return env
