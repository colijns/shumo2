"""Q4纯A严格统一复核的统计与几何测试。"""

import os
import sys
import unittest

import numpy as np


Q4_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if Q4_DIR not in sys.path:
    sys.path.insert(0, Q4_DIR)
import run_unified_a_audit as audit  # noqa: E402


class TestUnifiedAudit(unittest.TestCase):

    def test_empirical_curve_is_monotone(self):
        curve = audit.empirical_curve([2, 3, None, 3], 4)
        np.testing.assert_allclose(curve['p_hat'], [0.0, 0.25, 0.75, 0.75])
        self.assertTrue(np.all(np.diff(curve['p_hat']) >= 0.0))

    def test_strict_three_way_classification(self):
        outer = {
            'n': np.arange(1, 4),
            'p_hat': np.array([0.70, 0.88, 0.95]),
            'ci_lower': np.array([0.60, 0.80, 0.91]),
            'ci_upper': np.array([0.80, 0.93, 0.98]),
        }
        inner = {
            'n': np.arange(1, 4),
            'p_hat': np.array([0.65, 0.85, 0.93]),
            'ci_lower': np.array([0.55, 0.78, 0.90]),
            'ci_upper': np.array([0.75, 0.91, 0.97]),
        }
        self.assertEqual(audit.classify_at(1, outer, inner)['verdict'],
                         'reliably_insufficient')
        self.assertEqual(audit.classify_at(2, outer, inner)['verdict'],
                         'undetermined')
        self.assertEqual(audit.classify_at(3, outer, inner)['verdict'],
                         'reliably_feasible')

    def test_small_geometric_sample_obeys_bracket(self):
        rng = np.random.default_rng(7)
        c, u, h = audit.axis_geometry.generate_cylinders(30, rng)
        result = audit.solid_fp.paired_first_contacts(
            c, u, h, n_sides=16)
        self.assertTrue(result['bracket_valid'])


if __name__ == '__main__':
    unittest.main()
