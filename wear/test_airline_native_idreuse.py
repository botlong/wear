"""Offline tests that tau2's airline domain never reuses a reservation id on its own.

The allocator issues ids only from a fixed pool, cancelling a reservation keeps
its key, a cancelled id is not issued again, and an exhausted pool raises instead
of recycling. Without the perturbation layer, an identifier therefore never comes
to name a different reservation.
Run in the tau2 env: pytest wear/test_airline_native_idreuse.py
"""
import unittest

from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.tools import AirlineTools
from tau2.domains.airline.utils import AIRLINE_DB_PATH

POOL = ["HATHAT", "HATHAU", "HATHAV"]


def _tools() -> AirlineTools:
    """Airline tools on a freshly loaded airline database."""
    return AirlineTools(FlightDB.load(AIRLINE_DB_PATH))


class TestAirlineNativeIdReuse(unittest.TestCase):
    def test_allocator_only_mints_fixed_pool(self):
        t = _tools()
        nid = t._get_new_reservation_id()
        self.assertIn(nid, POOL, "allocator issued a code outside the fixed pool - upstream id generation may have changed")
        self.assertNotIn(nid, t.db.reservations, "issued code already absent from the DB")

    def test_cancel_keeps_key_status_only(self):
        t = _tools()
        rid = next(iter(t.db.reservations))
        self.assertNotEqual(t.db.reservations[rid].status, "cancelled")
        t.cancel_reservation(rid)
        self.assertIn(rid, t.db.reservations, "cancel must not delete the key (mark only)")
        self.assertEqual(t.db.reservations[rid].status, "cancelled")

    def test_freed_code_never_reissued(self):
        t = _tools()
        rid = next(iter(t.db.reservations))
        t.cancel_reservation(rid)
        nid = t._get_new_reservation_id()
        self.assertIn(nid, POOL)
        self.assertNotEqual(nid, rid, "cancelled code was recycled - native id reuse appeared; recheck the perturbation design")
        self.assertIn(rid, t.db.reservations, "cancelled code still occupies its slot, so it is never reused")

    def test_pool_exhaustion_raises_not_recycles(self):
        t = _tools()
        placeholder = next(iter(t.db.reservations.values()))
        for code in POOL:
            t.db.reservations[code] = placeholder
        with self.assertRaises(ValueError):
            t._get_new_reservation_id()


if __name__ == "__main__":
    unittest.main()
