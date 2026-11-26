"""Regression test for paper/forensic_recover.classify_recover on the recovery-suite run files.

The 12 recovery trajectories that do not restore the gold state split into 7 in
which the agent corrects the target record and then writes to the distractor
through the old identifier again (poisoned_recovery), 4 in which it corrects the
target record but leaves the distractor wrong (incomplete_recovery), and 1 in
which it never writes to the new identifier (single_no_recovery). Needs the run
files under wear/runs/.

Run from the repository root: tau2-bench/.venv/bin/python -m pytest wear/test_forensic_recover.py
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "paper"))

from forensic_recover import classify_recover  # noqa: E402

EXPECTED = {
    ("retail_recover_backpack", 300): "poisoned_recovery",
    ("retail_recover_backpack", 301): "incomplete_recovery",
    ("retail_recover_backpack", 302): "poisoned_recovery",
    ("retail_recover_backpack", 303): "poisoned_recovery",
    ("retail_recover_backpack", 304): "poisoned_recovery",
    ("retail_recover_desklamp", 302): "incomplete_recovery",
    ("retail_recover_desklamp", 304): "single_no_recovery",
    ("retail_recover_toothbrush", 300): "incomplete_recovery",
    ("retail_recover_toothbrush", 301): "poisoned_recovery",
    ("retail_recover_toothbrush", 302): "incomplete_recovery",
    ("retail_recover_toothbrush", 303): "poisoned_recovery",
    ("retail_recover_toothbrush", 304): "poisoned_recovery",
}


class TestForensicRecover(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = classify_recover()

    def test_totals_7_4_1(self):
        c = self.res["counts"]
        self.assertEqual(self.res["n_fail_wrong"], 12, "n_fail_wrong should be 12")
        self.assertEqual(
            (c["poisoned_recovery"], c["incomplete_recovery"], c["single_no_recovery"]),
            (7, 4, 1), "split should be 7 poisoned / 4 incomplete / 1 single")

    def test_per_seed_labels(self):
        got = {(t["task"], t["seed"]): t["label"] for t in self.res["per_traj"]}
        self.assertEqual(got, EXPECTED, "per-seed labels diverge from the archived forensic split")

    def test_no_errored_writes_invariant(self):
        errored = [(t["task"], t["seed"]) for t in self.res["per_traj"] if t["any_errored"]]
        self.assertEqual(errored, [], "errored write appeared - recheck the target-based criterion")


if __name__ == "__main__":
    unittest.main()
