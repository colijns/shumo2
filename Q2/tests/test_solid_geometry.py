import os
import sys
import unittest

import numpy as np


Q2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, Q2_DIR)
import geometry as axis_geo  # noqa: E402
import solid_geometry as solid  # noqa: E402


class TestPrismConstruction(unittest.TestCase):

    def test_radial_bracket_and_error(self):
        inner = solid.polygon_radial_radius(30.0, 64, 'inscribed')
        outer = solid.polygon_radial_radius(30.0, 64, 'circumscribed')
        self.assertLessEqual(inner, 30.0)
        self.assertGreaterEqual(outer, 30.0)
        self.assertLess(solid.radial_error_bound(30.0, 64), 0.04)

    def test_uncut_prism_vertices(self):
        faces = solid.prism_faces(
            np.zeros(3), np.array([1.0, 0.0, 0.0]), 100.0,
            n_sides=32, mode='inscribed')
        _, vertices = solid.clip_faces_to_box(faces)
        self.assertEqual(len(vertices), 64)
        self.assertTrue(np.all(vertices[:, 0] <= 100.0 + 1e-8))
        self.assertTrue(np.all(vertices[:, 0] >= -100.0 - 1e-8))

    def test_radial_only_crossing_creates_wrapped_piece(self):
        c = np.array([[0.0, 4975.0, 0.0]])
        u = np.array([[1.0, 0.0, 0.0]])
        h = np.array([2500.0])
        axis_frag = axis_geo.clip_batch(c, u, h)
        solid_frag = solid.wrap_prism_fragments(c, u, h, n_sides=32)
        self.assertEqual(len(axis_frag['p1s']), 1)
        self.assertEqual(len(solid_frag), 2)
        for fragment in solid_frag:
            self.assertTrue(np.all(fragment.lo >= -axis_geo.HALF_L - 1e-7))
            self.assertTrue(np.all(fragment.hi <= axis_geo.HALF_L + 1e-7))


class TestSolidDistance(unittest.TestCase):

    def test_known_gap_between_parallel_prisms(self):
        c = np.array([[0.0, 0.0, 0.0], [0.0, 61.8, 0.0]])
        u = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        h = np.array([100.0, 100.0])
        fragments = solid.wrap_prism_fragments(c, u, h, n_sides=64)
        self.assertEqual(len(fragments), 2)
        distance = solid.gjk_polytope_distance(fragments[0], fragments[1])
        self.assertAlmostEqual(distance, 1.8, places=6)

    def test_single_wrapped_source_not_automatic_conductive(self):
        c = np.array([[-3500.0, 0.0, 0.0]])
        u = np.array([[1.0, 0.0, 0.0]])
        h = np.array([2500.0])
        result = solid.sample_conductive_solid(c, u, h, n_sides=32)
        self.assertFalse(result['conductive'])
        self.assertEqual(result['n_fragments'], 2)
        self.assertEqual(result['n_crossing'], 1)


class TestBracket(unittest.TestCase):

    def test_conductivity_is_bracketed_near_threshold(self):
        # 两根圆柱分别接左右电极，真实侧向表面间距1.85 nm。
        # 64边内接棱柱仍断开；外切棱柱的径向扩张使其连通。
        c = np.array([[-2499.0, 0.0, 0.0],
                      [2499.0, 61.85, 0.0]])
        u = np.array([[1.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0]])
        h = np.array([2500.0, 2500.0])
        inner = solid.sample_conductive_solid(
            c, u, h, n_sides=64, mode='inscribed')
        outer = solid.sample_conductive_solid(
            c, u, h, n_sides=64, mode='circumscribed')
        self.assertFalse(inner['conductive'])
        self.assertTrue(outer['conductive'])

    def test_far_and_near_cases_agree(self):
        u = np.array([[1.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0]])
        h = np.array([2500.0, 2500.0])
        for separation, expected in ((61.0, True), (63.0, False)):
            c = np.array([[-2499.0, 0.0, 0.0],
                          [2499.0, separation, 0.0]])
            states = [solid.sample_conductive_solid(
                c, u, h, n_sides=64, mode=mode)['conductive']
                      for mode in ('inscribed', 'circumscribed')]
            self.assertEqual(states, [expected, expected])

    def test_inscribed_vertices_inside_circumscribed_cross_section(self):
        c = np.array([[0.0, 0.0, 0.0]])
        u = np.array([[0.3, 0.4, np.sqrt(0.75)]])
        h = np.array([2500.0])
        inner = solid.wrap_prism_fragments(
            c, u, h, n_sides=32, mode='inscribed')[0]
        outer = solid.wrap_prism_fragments(
            c, u, h, n_sides=32, mode='circumscribed')[0]
        self.assertTrue(np.all(inner.lo >= outer.lo - 1e-7))
        self.assertTrue(np.all(inner.hi <= outer.hi + 1e-7))


if __name__ == '__main__':
    unittest.main()
