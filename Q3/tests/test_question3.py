# 本程序及代码是在AI工具辅助下完成的
"""问题3实体首次导通与统计汇总测试。"""

import os
import sys
import unittest

import numpy as np


Q3_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, Q3_DIR)
import run_question3 as q3  # noqa: E402
import solid_first_passage as passage  # noqa: E402


class TestFirstPassage(unittest.TestCase):

    def test_empty_sequence(self):
        empty = np.empty((0, 3))
        result = passage.first_contact_n(empty, empty, np.empty(0))
        self.assertFalse(result['conductive'])
        self.assertIsNone(result['n_contact'])

    def test_two_cylinder_chain_connects_at_two(self):
        c = np.array([[-2499.0, 0.0, 0.0],
                      [2499.0, 61.0, 0.0]])
        u = np.array([[1.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0]])
        h = np.array([2500.0, 2500.0])
        for mode in ('inscribed', 'circumscribed'):
            result = passage.first_contact_n(
                c, u, h, n_sides=64, mode=mode)
            self.assertTrue(result['conductive'])
            self.assertEqual(result['n_contact'], 2)

    def test_broken_chain_never_connects(self):
        c = np.array([[-2499.0, 0.0, 0.0],
                      [2499.0, 63.0, 0.0]])
        u = np.array([[1.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0]])
        h = np.array([2500.0, 2500.0])
        for mode in ('inscribed', 'circumscribed'):
            result = passage.first_contact_n(
                c, u, h, n_sides=64, mode=mode)
            self.assertFalse(result['conductive'])
            self.assertIsNone(result['n_contact'])

    def test_near_threshold_is_bracketed(self):
        c = np.array([[-2499.0, 0.0, 0.0],
                      [2499.0, 61.85, 0.0]])
        u = np.array([[1.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0]])
        h = np.array([2500.0, 2500.0])
        result = passage.paired_first_contacts(c, u, h, n_sides=64)
        self.assertEqual(result['outer']['n_contact'], 2)
        self.assertIsNone(result['inner']['n_contact'])
        self.assertTrue(result['bracket_valid'])

    def test_random_sample_obeys_bracket(self):
        rng = np.random.default_rng(7)
        c, u, h = passage.axis_geometry.generate_cylinders(40, rng)
        result = passage.paired_first_contacts(c, u, h, n_sides=32)
        self.assertTrue(result['bracket_valid'])


class TestStatistics(unittest.TestCase):

    def test_empirical_cdf_is_monotone(self):
        curve = q3.empirical_cdf([2, 3, 3, None], 4, target=0.5)
        np.testing.assert_allclose(curve['p_hat'], [0.0, 0.25, 0.75, 0.75])
        self.assertEqual(curve['n_hat'], 3)
        self.assertTrue(np.all(np.diff(curve['p_hat']) >= 0.0))

    def test_none_means_beyond_search_limit(self):
        curve = q3.empirical_cdf([None] * 10, 5, target=0.9)
        self.assertIsNone(curve['n_hat'])
        self.assertIsNone(curve['n_safe'])

    def test_half_up_and_conservative_rounding(self):
        self.assertEqual(q3.round_half_up_2(0.865), 0.87)
        self.assertEqual(q3.ceil_2(0.860001), 0.87)
        self.assertEqual(q3.ceil_2(0.86), 0.86)

    def test_phi_conversion(self):
        expected = 611 * np.pi * 30.0 ** 2 * 5000.0 / 10000.0 ** 3 * 100.0
        self.assertAlmostEqual(q3.phi_percent(611), expected)


if __name__ == '__main__':
    unittest.main()
