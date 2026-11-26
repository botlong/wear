"""Unit tests for stats.py.

Run from the repository root: tau2-bench/.venv/bin/python -m pytest wear/test_stats.py
"""
import unittest

from stats import (
    cell_mean, paired_permutation, paired_sign_flip_exact, task_means,
    two_stage_bootstrap_ci, wilson,
)
from fractions import Fraction


class TestTaskAggregation(unittest.TestCase):
    def test_task_equal_weight_not_seed_weight(self):
        cell = {"A": [1.0] * 10, "B": [0.0]}
        self.assertAlmostEqual(cell_mean(cell), 0.5)
        self.assertEqual(task_means(cell), {"A": 1.0, "B": 0.0})

    def test_empty_task_dropped(self):
        self.assertEqual(task_means({"A": [1.0], "B": []}), {"A": 1.0})


class TestBootstrap(unittest.TestCase):
    def test_degenerate_all_same_collapses(self):
        point, lo, hi = two_stage_bootstrap_ci({"A": [1.0, 1.0], "B": [1.0]}, n_boot=500)
        self.assertEqual((point, lo, hi), (1.0, 1.0, 1.0))

    def test_ci_brackets_point_and_orders(self):
        cell = {"A": [1.0, 1.0, 0.0], "B": [0.0, 0.0], "C": [1.0], "D": [0.5, 0.5]}
        point, lo, hi = two_stage_bootstrap_ci(cell, n_boot=2000, seed=7)
        self.assertLessEqual(lo, point)
        self.assertLessEqual(point, hi)
        self.assertLess(lo, hi)

    def test_deterministic_given_seed(self):
        cell = {"A": [1.0, 0.0], "B": [0.5]}
        self.assertEqual(
            two_stage_bootstrap_ci(cell, n_boot=300, seed=3),
            two_stage_bootstrap_ci(cell, n_boot=300, seed=3),
        )


class TestWilson(unittest.TestCase):
    def test_saturated_10_of_10(self):
        lo, hi = wilson(10, 10)
        self.assertAlmostEqual(hi, 1.0)
        self.assertAlmostEqual(lo, 0.7225, places=3)

    def test_zero_of_10(self):
        lo, hi = wilson(0, 10)
        self.assertAlmostEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.2775, places=3)

    def test_half(self):
        lo, hi = wilson(5, 10)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            wilson(11, 10)
        with self.assertRaises(ValueError):
            wilson(0, 0)


class TestPairedPermutation(unittest.TestCase):
    def test_strong_effect_small_p(self):
        pairs = {f"t{i}": (0.0, 1.0) for i in range(10)}
        obs, p = paired_permutation(pairs, n_perm=4000, seed=1)
        self.assertAlmostEqual(obs, 1.0)
        self.assertLess(p, 0.01)

    def test_null_effect_large_p(self):
        pairs = {f"t{i}": (0.5, 0.5) for i in range(10)}
        obs, p = paired_permutation(pairs, n_perm=1000, seed=2)
        self.assertEqual(obs, 0.0)
        self.assertGreater(p, 0.9)

    def test_add_one_smoothing_no_zero_p(self):
        pairs = {f"t{i}": (0.0, 1.0) for i in range(20)}
        _, p = paired_permutation(pairs, n_perm=1000, seed=3)
        self.assertGreater(p, 0.0)


class TestExactSignFlip(unittest.TestCase):
    def test_saturated_ten_pairs_is_one_over_512(self):
        pairs = {str(i): (0.0, 1.0) for i in range(10)}
        obs, p = paired_sign_flip_exact(pairs)
        self.assertEqual(obs, 1.0)
        self.assertEqual(p, Fraction(2, 2 ** 10))

    def test_mixed_signs_counts_ties(self):
        # diffs +1, +1, -1: |mean| = 1/3; patterns with |sum| >= 1 are all 8
        obs, p = paired_sign_flip_exact({"a": (0, 1), "b": (0, 1), "c": (1, 0)})
        self.assertAlmostEqual(obs, 1 / 3)
        self.assertEqual(p, Fraction(1))

    def test_unequal_magnitudes(self):
        # diffs 3, 1: sums +-4, +-2 -> |sum| >= 4 in 2 of 4 patterns
        _, p = paired_sign_flip_exact({"a": (0, 3), "b": (0, 1)})
        self.assertEqual(p, Fraction(1, 2))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            paired_sign_flip_exact({})


if __name__ == "__main__":
    unittest.main()
