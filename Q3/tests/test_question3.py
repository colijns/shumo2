# 本程序及代码是在AI工具辅助下完成的
"""问题3网格、区间分类与二分搜索测试。"""

import os
import sys
import unittest


Q3_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, Q3_DIR)
import run_question3 as q3  # noqa: E402


class TestGrid(unittest.TestCase):

    def test_percent_precision(self):
        self.assertAlmostEqual(q3.GRID_STEP * 100.0, 0.01)

    def test_round_trip(self):
        for phi in [0.0, 0.005, 0.0087, 0.01]:
            self.assertAlmostEqual(
                q3.index_to_phi(q3.phi_to_index(phi)), phi)


class TestDecision(unittest.TestCase):

    def test_interval_above(self):
        row = {'p_hat': 0.95, 'ci_lower': 0.91, 'ci_upper': 0.98}
        self.assertEqual(q3.interval_classification(row), 'above')
        self.assertEqual(q3.point_decision(row), 'above')

    def test_interval_below(self):
        row = {'p_hat': 0.85, 'ci_lower': 0.80, 'ci_upper': 0.89}
        self.assertEqual(q3.interval_classification(row), 'below')
        self.assertEqual(q3.point_decision(row), 'below')

    def test_interval_uncertain_uses_point_estimate(self):
        row = {'p_hat': 0.91, 'ci_lower': 0.86, 'ci_upper': 0.94}
        self.assertEqual(q3.interval_classification(row), 'uncertain')
        self.assertEqual(q3.point_decision(row), 'above')


class TestSearch(unittest.TestCase):

    def test_first_feasible(self):
        threshold = 87

        def fake(index):
            return {'decision': 'above' if index >= threshold else 'below'}

        low, high = q3.find_first_feasible(fake, 0, 100)
        self.assertEqual(low, 86)
        self.assertEqual(high, 87)

    def test_select_confirmed_bracket(self):
        rows = [
            {'grid_index': 85, 'p_hat': 0.86},
            {'grid_index': 86, 'p_hat': 0.89},
            {'grid_index': 87, 'p_hat': 0.91},
            {'grid_index': 88, 'p_hat': 0.95},
        ]
        index, valid = q3.select_confirmed_bracket(rows)
        self.assertEqual(index, 87)
        self.assertTrue(valid)

    def test_first_confirmation_point_above_is_not_bracketed(self):
        rows = [
            {'grid_index': 86, 'p_hat': 0.91},
            {'grid_index': 87, 'p_hat': 0.94},
        ]
        index, valid = q3.select_confirmed_bracket(rows)
        self.assertEqual(index, 86)
        self.assertFalse(valid)


if __name__ == '__main__':
    unittest.main()
