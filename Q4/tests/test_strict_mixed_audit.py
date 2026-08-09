"""严格混合候选统计判定测试。"""

import os
import sys
import unittest


Q4_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if Q4_DIR not in sys.path:
    sys.path.insert(0, Q4_DIR)
import run_strict_mixed_audit as audit  # noqa: E402


class TestStrictVerdict(unittest.TestCase):

    def test_reliably_feasible_uses_inner_lower(self):
        verdict, _, _ = audit.strict_verdict(96, 98, 100)
        self.assertEqual(verdict, 'reliably_feasible')

    def test_reliably_insufficient_uses_outer_upper(self):
        verdict, _, _ = audit.strict_verdict(70, 80, 100)
        self.assertEqual(verdict, 'reliably_insufficient')

    def test_crossing_bounds_are_undetermined(self):
        verdict, _, _ = audit.strict_verdict(90, 92, 100)
        self.assertEqual(verdict, 'undetermined')

    def test_summary_does_not_overclaim_with_cheaper_unknown(self):
        candidates = ((1, 0), (2, 0))
        rows = []
        # 第一个点90/100仍待定；第二个点96/100可靠可行。
        for trial in range(100):
            rows.append({
                'trial_index': trial,
                'inner': [int(trial < 90), int(trial < 96)],
                'outer': [int(trial < 92), int(trial < 98)],
                'bracket_violations': 0,
                'inner_gjk': 0,
                'outer_gjk': 0,
            })
        summary = audit.summarize(
            candidates, rows, cyl_sides=32,
            ball_subdivisions=2, base_seed=42)
        self.assertEqual(
            summary['cheapest_reliably_feasible']['N_A'], 2)
        self.assertEqual(summary['cheaper_unresolved_count'], 1)
        self.assertFalse(summary['global_claim_ready_within_candidate_set'])


if __name__ == '__main__':
    unittest.main()
