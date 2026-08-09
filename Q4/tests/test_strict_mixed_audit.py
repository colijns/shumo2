"""严格混合候选统计判定测试。"""

import csv
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


Q4_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(Q4_DIR)
for path in (Q4_DIR, os.path.join(ROOT, 'Q2')):
    if path not in sys.path:
        sys.path.insert(0, path)
import geometry as axis_geometry  # noqa: E402
import run_strict_mixed_audit as audit  # noqa: E402


class TestStrictVerdict(unittest.TestCase):

    def test_result_files_are_atomically_writable(self):
        summary = {
            'config': {'trials': 1},
            'candidate_results': [
                {'N_A': 1, 'N_B': 2, 'strict_verdict': 'undetermined'},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            summary_path = os.path.join(directory, 'summary.json')
            csv_path = os.path.join(directory, 'candidates.csv')
            with mock.patch.object(audit, 'SUMMARY', summary_path), \
                    mock.patch.object(audit, 'CSV_RESULT', csv_path):
                audit._write_result_files(summary)
            with open(summary_path, encoding='utf-8') as handle:
                self.assertEqual(json.load(handle), summary)
            with open(csv_path, newline='', encoding='utf-8-sig') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]['N_A'], '1')
            self.assertEqual(rows[0]['N_B'], '2')

    def test_checkpoint_atomic_backup_and_corruption_fallback(self):
        config = {
            'candidates': [[1, 0]],
            'base_seed': 42,
            'cyl_sides': 8,
            'ball_subdivisions': 1,
            'a_sequence_length': 2,
        }
        first = {
            0: {'trial_index': 0, 'inner': [0], 'outer': [1],
                'bracket_violations': 0, 'inner_gjk': 1, 'outer_gjk': 2},
        }
        second = dict(first)
        second[1] = {
            'trial_index': 1, 'inner': [1], 'outer': [1],
            'bracket_violations': 0, 'inner_gjk': 3, 'outer_gjk': 4,
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, 'checkpoint.json')
            backup = checkpoint + '.bak'
            with mock.patch.object(audit, 'CHECKPOINT', checkpoint), \
                    mock.patch.object(audit, 'CHECKPOINT_BACKUP', backup):
                audit._save_checkpoint(config, first)
                audit._save_checkpoint(config, second)
                self.assertEqual(
                    sorted(audit._load_checkpoint(config, True)), [0, 1])
                with open(checkpoint, 'w', encoding='utf-8') as handle:
                    handle.write('{')
                # 主文件损坏时自动回退到上一版完整备份。
                self.assertEqual(
                    sorted(audit._load_checkpoint(config, True)), [0])

    def test_fixed_750_sequence_matches_q3_generation_prefix(self):
        seed = 20260808
        c, u, h, balls = audit._generate_trial_geometry(
            seed, n_a_used=619, n_b_used=7, a_sequence_length=750)
        rng = np.random.default_rng(seed)
        c_all, u_all, h_all = axis_geometry.generate_cylinders(750, rng)
        expected_balls = audit.fast_geometry.generate_balls(7, rng)
        np.testing.assert_allclose(c, c_all[:619])
        np.testing.assert_allclose(u, u_all[:619])
        np.testing.assert_allclose(h, h_all[:619])
        np.testing.assert_allclose(balls, expected_balls)

    def test_reliably_feasible_uses_inner_lower(self):
        verdict, _, _ = audit.strict_verdict(96, 98, 100)
        self.assertEqual(verdict, 'reliably_feasible')

    def test_reliably_insufficient_uses_outer_upper(self):
        verdict, _, _ = audit.strict_verdict(70, 80, 100)
        self.assertEqual(verdict, 'reliably_insufficient')

    def test_crossing_bounds_are_undetermined(self):
        verdict, _, _ = audit.strict_verdict(90, 92, 100)
        self.assertEqual(verdict, 'undetermined')

    def test_joint_interval_is_wider_than_pointwise_interval(self):
        _, point_inner, point_outer = audit.strict_verdict(92, 94, 100)
        _, joint_inner, joint_outer, alpha_each = \
            audit.joint_strict_verdict(92, 94, 100, n_candidates=11)
        self.assertAlmostEqual(alpha_each, 0.05 / 22)
        self.assertLessEqual(joint_inner[0], point_inner[0])
        self.assertGreaterEqual(joint_outer[1], point_outer[1])

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
            summary['stage_pointwise_cheapest_reliably_feasible']['N_A'], 2)
        self.assertEqual(summary['cheaper_unresolved_count'], 1)
        self.assertFalse(summary['global_claim_ready_within_candidate_set'])
        self.assertEqual(
            summary['formal_unified_recommendation']['N_A'], 619)


if __name__ == '__main__':
    unittest.main()
