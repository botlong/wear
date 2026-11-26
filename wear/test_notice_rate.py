"""Unit tests for notice_rate._contains_id, the whole-token identifier match.

Run from the repository root: tau2-bench/.venv/bin/python -m pytest wear/test_notice_rate.py
"""
import unittest

from notice_rate import _contains_id


class TestContainsId(unittest.TestCase):
    def test_whole_token_match(self):
        self.assertTrue(_contains_id('{"reservation_id": "NO6JO3"}', "NO6JO3"))
        self.assertTrue(_contains_id("current id NO6JO3 now", "NO6JO3"))

    def test_boundary_rejects_longer_token(self):
        self.assertFalse(_contains_id("ZZ00010", "ZZ0001"))
        self.assertFalse(_contains_id("XNO6JO3", "NO6JO3"))
        self.assertFalse(_contains_id("NO6JO3A", "NO6JO3"))

    def test_case_insensitive(self):
        self.assertTrue(_contains_id('{"id": "no6jo3"}', "NO6JO3"))
        self.assertTrue(_contains_id("No6Jo3", "NO6JO3"))

    def test_case_insensitive_boundary(self):
        self.assertFalse(_contains_id("no6jo3x", "NO6JO3"))
        self.assertTrue(_contains_id('found "no6jo3".', "NO6JO3"))

    def test_regex_special_id_escaped(self):
        self.assertTrue(_contains_id('order "#W0000001" here', "#W0000001"))
        self.assertFalse(_contains_id("#W00000012", "#W0000001"))

    def test_absent(self):
        self.assertFalse(_contains_id('{"reservation_id": "HKEG34"}', "NO6JO3"))


if __name__ == "__main__":
    unittest.main()
