import importlib.util
import os
import unittest

import numpy as np


Q4_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location(
    'q4_geometry', os.path.join(Q4_DIR, 'geometry.py'))
geo = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(geo)


class TestClippedBall(unittest.TestCase):

    def test_wrap_crossing_x_creates_two_fragments(self):
        shapes = geo.wrap_spheres(np.array([[4900.0, 0.0, 0.0]]))
        self.assertEqual(len(shapes), 2)
        bounds = sorted((round(s.lo[0]), round(s.hi[0])) for s in shapes)
        self.assertEqual(bounds, [(-5000, -4900), (4700, 5000)])

    def test_support_is_inside_ball_and_box(self):
        shape = geo.wrap_spheres(np.array([[4900.0, 4900.0, 0.0]]))[0]
        for direction in ([1, 2, 3], [-2, 1, 0.5], [0, -1, 0]):
            point = shape.support(np.asarray(direction, dtype=float))
            self.assertTrue(np.all(point >= -geo.HALF_L - 1e-7))
            self.assertTrue(np.all(point <= geo.HALF_L + 1e-7))
            self.assertLessEqual(np.linalg.norm(point - shape.center),
                                 shape.radius + 1e-6)

    def test_support_dominates_random_feasible_points(self):
        """数值抽查支撑点确实使给定方向上的投影最大。"""
        rng = np.random.default_rng(20260808)
        shape = geo.wrap_spheres(np.array([[4900.0, 4900.0, 0.0]]))[0]
        directions = rng.normal(size=(12, 3))
        raw = rng.uniform(-1.0, 1.0, size=(50000, 3))
        raw = raw[np.linalg.norm(raw, axis=1) <= 1.0]
        points = shape.center + shape.radius * raw
        feasible = points[np.all(points >= -geo.HALF_L, axis=1)
                          & np.all(points <= geo.HALF_L, axis=1)]
        self.assertGreater(len(feasible), 100)
        for direction in directions:
            support = shape.support(direction)
            self.assertGreaterEqual(
                float(direction @ support) + 1e-7,
                float(np.max(feasible @ direction)))

    def test_single_wrapped_sphere_does_not_short_electrodes(self):
        result = geo.sample_conductive(
            np.empty((0, 3)), np.empty((0, 3)), np.empty(0),
            np.array([[4900.0, 0.0, 0.0]]))
        self.assertFalse(result['conductive'])
        self.assertEqual(result['n_B_fragments'], 2)
        self.assertEqual(result['n_B_crossing'], 1)


class TestDistances(unittest.TestCase):

    def test_sphere_sphere_gap(self):
        balls = geo.wrap_spheres(np.array([[0.0, 0.0, 0.0],
                                            [401.8, 0.0, 0.0]]))
        self.assertAlmostEqual(geo.shape_distance(balls[0], balls[1]),
                               1.8, places=7)

    def test_cylinder_sphere_side_gap(self):
        c = np.array([[0.0, 0.0, 0.0]])
        u = np.array([[1.0, 0.0, 0.0]])
        h = np.array([100.0])
        cylinder = geo.cylinder_shapes(c, u, h)[0]
        ball = geo.wrap_spheres(np.array([[0.0, 231.8, 0.0]]))[0]
        self.assertAlmostEqual(geo.shape_distance(cylinder, ball),
                               1.8, places=7)

    def test_clipped_ball_distance_symmetric(self):
        clipped = geo.wrap_spheres(np.array([[4900.0, 0.0, 0.0]]))
        other = geo.wrap_spheres(np.array([[-4600.0, 0.0, 0.0]]))[0]
        left_cap = min(clipped, key=lambda s: s.center[0])
        d1 = geo.shape_distance(left_cap, other)
        d2 = geo.shape_distance(other, left_cap)
        self.assertAlmostEqual(d1, d2, places=7)
        self.assertAlmostEqual(d1, 100.0, places=7)


class TestCandidateSweep(unittest.TestCase):

    def test_aabb_sweep_matches_brute_force(self):
        rng = np.random.default_rng(314159)
        centers = rng.uniform(-geo.HALF_L, geo.HALF_L, size=(40, 3))
        centers[:3, 0] = [4990.0, -4990.0, 4900.0]
        shapes = geo.wrap_spheres(centers)
        actual = set(geo.aabb_candidate_pairs(shapes))
        expected = set()
        for i in range(len(shapes)):
            for j in range(i + 1, len(shapes)):
                gaps = np.maximum(
                    np.maximum(shapes[i].lo - shapes[j].hi,
                               shapes[j].lo - shapes[i].hi), 0.0)
                if np.all(gaps <= geo.DELTA):
                    expected.add((i, j))
        self.assertEqual(actual, expected)


class TestGraph(unittest.TestCase):

    def test_pure_a_matches_question2_kernel(self):
        """混合内核在B=0时必须退化为已验证的问题2内核。"""
        rng = np.random.default_rng(20260808)
        for n_a in (1, 12, 40):
            c, u, h = geo.q2geo.generate_cylinders(n_a, rng)
            expected = geo.q2geo.sample_conductive(c, u, h)
            actual = geo.sample_conductive(
                c, u, h, np.empty((0, 3)))
            self.assertEqual(actual['conductive'], expected['conductive'])
            self.assertEqual(actual['n_A_fragments'], expected['n_fragments'])
            self.assertEqual(actual['n_edges'], expected['n_edges'])
            self.assertLessEqual(actual['n_A_crossing'], n_a)

    def test_sphere_chain_conducts(self):
        # 25个球从左右电极排成接触链；相邻中心距400 nm，球面相切。
        centers = np.column_stack([
            np.linspace(-4800.0, 4800.0, 25),
            np.zeros(25), np.zeros(25)])
        result = geo.sample_conductive(
            np.empty((0, 3)), np.empty((0, 3)), np.empty(0), centers)
        self.assertTrue(result['conductive'])

    def test_broken_sphere_chain_does_not_conduct(self):
        centers = np.column_stack([
            np.linspace(-4800.0, 4800.0, 25),
            np.zeros(25), np.zeros(25)])
        centers[12, 1] = 1000.0
        result = geo.sample_conductive(
            np.empty((0, 3)), np.empty((0, 3)), np.empty(0), centers)
        self.assertFalse(result['conductive'])


if __name__ == '__main__':
    unittest.main()
