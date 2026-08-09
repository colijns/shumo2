"""严格混合实体内外界测试。"""

import os
import sys
import unittest

import numpy as np


Q4_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(Q4_DIR)
for path in (Q4_DIR, os.path.join(ROOT, 'Q2'), os.path.join(ROOT, 'Q3')):
    if path not in sys.path:
        sys.path.insert(0, path)

import geometry as axis_geometry  # noqa: E402
import solid_first_passage as q3_solid  # noqa: E402
import solid_mix_geometry as smg  # noqa: E402


class TestBallPolyhedron(unittest.TestCase):

    def test_inner_vertices_and_outer_face_bound(self):
        vertices, faces, inradius = smg.unit_icosphere(2)
        np.testing.assert_allclose(
            np.linalg.norm(vertices, axis=1), 1.0, atol=1e-12)
        self.assertGreater(inradius, 0.0)
        _, outer_vertices, scale = smg.ball_polyhedron_faces(
            np.zeros(3), subdivisions=2, mode='circumscribed')
        self.assertGreaterEqual(inradius * scale, smg.R_B - 1e-10)
        self.assertTrue(np.all(
            np.linalg.norm(outer_vertices, axis=1) >= smg.R_B))
        self.assertEqual(len(faces), 320)

    def test_error_decreases_with_refinement(self):
        errors = [smg.ball_polyhedron_error(subdivisions=k)
                  for k in range(3)]
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])

    def test_cross_wall_fragments_are_clipped_inside_box(self):
        centers = np.array([[-4910.0, -4910.0, 0.0]])
        for mode in ('inscribed', 'circumscribed'):
            fragments = smg.wrap_ball_fragments(
                centers, subdivisions=1, mode=mode)
            self.assertGreaterEqual(len(fragments), 2)
            for fragment in fragments:
                self.assertTrue(np.all(fragment.lo >= -smg.HALF_L - 1e-7))
                self.assertTrue(np.all(fragment.hi <= smg.HALF_L + 1e-7))


class TestUnifiedGraph(unittest.TestCase):

    def test_analytic_contact_bounds_match_gjk_reference(self):
        rng = np.random.default_rng(314159)
        c, u, h = axis_geometry.generate_cylinders(18, rng)
        balls = rng.uniform(-smg.HALF_L, smg.HALF_L, size=(24, 3))
        for mode in ('inscribed', 'circumscribed'):
            fragments = (smg.wrap_cylinder_fragments(
                c, u, h, n_sides=16, mode=mode)
                + smg.wrap_ball_fragments(
                    balls, subdivisions=1, mode=mode))
            fast, _, _ = smg._contact_edges(
                fragments, cyl_sides=16, ball_subdivisions=1,
                mode=mode, use_analytic_bounds=True)
            reference, _, _ = smg._contact_edges(
                fragments, cyl_sides=16, ball_subdivisions=1,
                mode=mode, use_analytic_bounds=False)
            self.assertEqual(set(map(tuple, fast.tolist())),
                             set(map(tuple, reference.tolist())))

    def test_pure_a_matches_q3_trial_by_trial(self):
        c = np.array([[-2499.0, 0.0, 0.0],
                      [2499.0, 61.0, 0.0]])
        u = np.array([[1.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0]])
        h = np.array([2500.0, 2500.0])
        balls = np.empty((0, 3))
        for mode in ('inscribed', 'circumscribed'):
            prepared = smg.prepare_trial(
                c, u, h, balls, cyl_sides=64,
                ball_subdivisions=1, mode=mode)
            q3 = q3_solid.first_contact_n(
                c, u, h, n_sides=64, mode=mode)
            for n_a in (1, 2):
                expected = (q3['n_contact'] is not None
                            and q3['n_contact'] <= n_a)
                self.assertEqual(
                    smg.sample_prefix(prepared, n_a, 0), expected)

    def test_inner_never_stronger_than_outer_random(self):
        rng = np.random.default_rng(20260809)
        c, u, h = axis_geometry.generate_cylinders(25, rng)
        balls = rng.uniform(-smg.HALF_L, smg.HALF_L, size=(30, 3))
        inner, outer = smg.prepare_paired(
            c, u, h, balls, cyl_sides=16, ball_subdivisions=1)
        for n_a, n_b in ((0, 0), (10, 10), (25, 10), (10, 30), (25, 30)):
            y_inner, y_outer = smg.paired_prefix(
                inner, outer, n_a, n_b)
            self.assertLessEqual(int(y_inner), int(y_outer))

    def test_prefix_monotonicity(self):
        rng = np.random.default_rng(17)
        c, u, h = axis_geometry.generate_cylinders(20, rng)
        balls = rng.uniform(-smg.HALF_L, smg.HALF_L, size=(25, 3))
        prepared = smg.prepare_trial(
            c, u, h, balls, cyl_sides=16,
            ball_subdivisions=1, mode='inscribed')
        points = [(10, 10), (20, 10), (10, 25), (20, 25)]
        values = [smg.sample_prefix(prepared, *point) for point in points]
        for i, j in ((0, 1), (0, 2), (1, 3), (2, 3)):
            self.assertLessEqual(int(values[i]), int(values[j]))

    def test_dense_pure_b_chain_connects(self):
        c = np.empty((0, 3))
        u = np.empty((0, 3))
        h = np.empty(0)
        xs = np.linspace(-4900.0, 4900.0, 26)
        balls = np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)])
        inner, outer = smg.prepare_paired(
            c, u, h, balls, cyl_sides=16, ball_subdivisions=2)
        self.assertTrue(smg.sample_prefix(inner, 0, len(balls)))
        self.assertTrue(smg.sample_prefix(outer, 0, len(balls)))


if __name__ == '__main__':
    unittest.main()
