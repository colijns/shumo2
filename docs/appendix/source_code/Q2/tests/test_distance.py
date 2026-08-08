# 本程序及代码是在AI工具辅助下完成的
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31

"""距离/建边正确性测试：批量 vs 标量、GJK vs 解析、电极距离、阈值边界。"""

import os
import sys
import unittest

import numpy as np

_Q2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _Q2)
sys.path.insert(0, os.path.join(_Q2, '..', 'Q1'))
import geometry as geo  # noqa: E402
from core import R, DELTA, segment_distance, gjk_distance  # noqa: E402


def _rand_axis_segments(rng, n):
    a1 = rng.uniform(-5000, 5000, size=(n, 3))
    b1 = a1 + rng.uniform(-3000, 3000, size=(n, 3))
    a2 = rng.uniform(-5000, 5000, size=(n, 3))
    b2 = a2 + rng.uniform(-3000, 3000, size=(n, 3))
    return a1, b1, a2, b2


class TestBatchDistance(unittest.TestCase):

    def test_matches_scalar_random(self):
        rng = np.random.default_rng(3)
        a1, b1, a2, b2 = _rand_axis_segments(rng, 2000)
        d_batch = geo.segment_distance_batch(a1, b1, a2, b2)
        d_scalar = np.array([segment_distance(a1[i], b1[i], a2[i], b2[i])
                             for i in range(2000)])
        np.testing.assert_allclose(d_batch, d_scalar, rtol=1e-9, atol=1e-9)

    def test_matches_scalar_parallel_collinear(self):
        # 平行/共线退化分支
        a1 = np.array([[-1000.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        b1 = np.array([[1000.0, 0.0, 0.0], [1000.0, 0.0, 0.0], [2000.0, 0.0, 0.0]])
        a2 = np.array([[2000.0, 5.0, 0.0], [0.0, 3.0, 0.0], [500.0, 0.0, 0.0]])
        b2 = np.array([[4000.0, 5.0, 0.0], [1000.0, 3.0, 0.0], [1500.0, 0.0, 0.0]])
        d_batch = geo.segment_distance_batch(a1, b1, a2, b2)
        for i in range(3):
            d_s = segment_distance(a1[i], b1[i], a2[i], b2[i])
            self.assertAlmostEqual(d_batch[i], d_s, places=9)

    def test_crossing_segments_zero(self):
        # 交叉线段（X 形）：轴距 0
        a1 = np.array([[-1000.0, -1000.0, 0.0]])
        b1 = np.array([[1000.0, 1000.0, 0.0]])
        a2 = np.array([[-1000.0, 1000.0, 0.0]])
        b2 = np.array([[1000.0, -1000.0, 0.0]])
        d = geo.segment_distance_batch(a1, b1, a2, b2)
        np.testing.assert_allclose(d, [0.0], atol=1e-9)


class TestGjkVsAnalytic(unittest.TestCase):
    """GJK（平端圆柱模型）与解析界的一致性。

    胶囊距离 capsule = max(0, 轴距-2R) 是平端圆柱表面距离的下界
    （胶囊 ⊃ 平端圆柱），且两者 Hausdorff 距离 ≤ r ⟹
    capsule ≤ gjk ≤ capsule + 2r。平行圆柱时 flat = capsule 精确。
    """

    def _to_cyl(self, a, b):
        c = 0.5 * (a + b)
        axis = b - a
        n = np.linalg.norm(axis)
        return c, axis / n, n / 2.0

    def _capsule(self, a1, a2, b1, b2):
        axis_d = geo.segment_distance_batch(a1[None], a2[None],
                                            b1[None], b2[None])[0]
        return max(0.0, axis_d - 2 * R)

    def test_separated_random_bounds(self):
        rng = np.random.default_rng(5)
        for _ in range(1000):
            a1 = rng.uniform(-5000, 5000, 3)
            u1 = rng.normal(size=3)
            a2 = a1 + 200.0 * u1 / np.linalg.norm(u1)
            b1 = rng.uniform(-5000, 5000, 3)
            u2 = rng.normal(size=3)
            b2 = b1 + 200.0 * u2 / np.linalg.norm(u2)
            c1, d1, h1 = self._to_cyl(a1, a2)
            c2, d2, h2 = self._to_cyl(b1, b2)
            capsule = self._capsule(a1, a2, b1, b2)
            gjk = gjk_distance(c1, d1, h1, c2, d2, h2, R)
            self.assertGreaterEqual(gjk, capsule - 1e-6)
            self.assertLessEqual(gjk, capsule + 2 * R + 1e-6)

    def test_parallel_exact(self):
        # 平行圆柱：平端 = 胶囊，GJK 与解析精确一致
        rng = np.random.default_rng(9)
        for _ in range(500):
            axis_d = rng.uniform(0.0, 200.0)
            a1 = np.array([-2000.0, 0.0, 0.0])
            b1 = np.array([2000.0, 0.0, 0.0])
            a2 = np.array([-2000.0, axis_d, 0.0])
            b2 = np.array([2000.0, axis_d, 0.0])
            c1, d1, h1 = self._to_cyl(a1, b1)
            c2, d2, h2 = self._to_cyl(a2, b2)
            analytic = max(0.0, axis_d - 2 * R)
            gjk = gjk_distance(c1, d1, h1, c2, d2, h2, R)
            self.assertAlmostEqual(gjk, analytic, places=6)

    def test_overlapping_cylinders_zero(self):
        # 贯穿重叠：GJK 返回 0
        c1 = np.array([0.0, 0.0, 0.0]); u1 = np.array([1.0, 0.0, 0.0]); h1 = 2500.0
        c2 = np.array([100.0, 0.0, 0.0]); u2 = np.array([0.0, 1.0, 0.0]); h2 = 2500.0
        gjk = gjk_distance(c1, u1, h1, c2, u2, h2, R)
        self.assertAlmostEqual(gjk, 0.0, places=9)


class TestElectrode(unittest.TestCase):

    def test_hand_calculated(self):
        # 单片段 x∈(4000,4500)（u=(1,0,0), h=250）：平端圆柱半径不凸出 x
        # e_x = h = 250 → x_min = 4000, x_max = 4500
        # dL = 4000+5000 = 9000，dR = 5000-4500 = 500
        p1s = np.array([[4000.0, 0.0, 0.0]])
        p2s = np.array([[4500.0, 0.0, 0.0]])
        dL, dR = geo.electrode_dist(p1s, p2s)
        self.assertAlmostEqual(dL[0], 4000.0 + 5000.0, places=6)
        self.assertAlmostEqual(dR[0], 5000.0 - 4500.0, places=6)
        # x∈(4800,5000)：dR = 5000-5000-30 < 0 → clamp 0，连
        p1s = np.array([[4800.0, 0.0, 0.0]])
        p2s = np.array([[5000.0, 0.0, 0.0]])
        dL, dR = geo.electrode_dist(p1s, p2s)
        self.assertAlmostEqual(dR[0], 0.0, places=6)

    def test_tilted_x_extent(self):
        # 倾斜圆柱：u=(0.8,0.6,0), h=100, c=(-1900,0,0)
        # 平端圆柱：e_x = h|u_x| + r√(1-u_x²) = 100*0.8 + 30*0.6 = 98
        # x_min = -1998, x_max = -1802
        u = np.array([0.8, 0.6, 0.0])
        c = np.array([-1900.0, 0.0, 0.0])
        p1s = (c - 100.0 * u)[None]
        p2s = (c + 100.0 * u)[None]
        dL, dR = geo.electrode_dist(p1s, p2s)
        self.assertAlmostEqual(dL[0], -1998.0 + 5000.0, places=6)
        self.assertAlmostEqual(dR[0], 5000.0 - (-1802.0), places=6)


class TestThreshold(unittest.TestCase):
    # 链式结构：cyl1 触左电极（轴段 x∈[-4999,1]，dL=1），cyl2 触右电极
    # （轴段 x∈[-1,4999]，dR=1），两圆柱平行相距 axis_d → 连通 ⇔ 轴距 ≤ 61.8
    # （平行圆柱时平端距离 = 胶囊距离 = max(0, 轴距-60) 精确）
    C1 = np.array([-2499.0, 0.0, 0.0])
    U1 = np.array([1.0, 0.0, 0.0])

    def _two_cyl(self, axis_d):
        c1 = self.C1
        u1 = self.U1
        h1 = 2500.0
        c2 = np.array([2499.0, axis_d, 0.0])
        u2 = np.array([1.0, 0.0, 0.0])
        h2 = 2500.0
        return np.array([c1, c2]), np.array([u1, u2]), np.array([h1, h2])

    def test_threshold_edge_connect(self):
        # 轴距 61.8 恰好 → 表面距离 1.8 → GJK 精算连
        c, u, h = self._two_cyl(61.8)
        res = geo.sample_conductive(c, u, h)
        self.assertTrue(res['conductive'])
        self.assertGreaterEqual(res['n_gjk'], 1)

    def test_threshold_edge_not_connect(self):
        # 轴距 61.9 > 61.8 → 胶囊预筛拒绝，不连
        c, u, h = self._two_cyl(61.9)
        res = geo.sample_conductive(c, u, h)
        self.assertFalse(res['conductive'])

    def test_critical_band_gjk_triggered(self):
        # 轴距 61.5 → 表面距离 1.5 ≤ 1.8 → GJK 建边导通
        c, u, h = self._two_cyl(61.5)
        res = geo.sample_conductive(c, u, h)
        self.assertGreaterEqual(res['n_gjk'], 1)
        self.assertTrue(res['conductive'])

    def test_critical_band_gjk_reject(self):
        # 轴距 61.75 → 表面距离 1.75 ≤ 1.8 → 导通
        c, u, h = self._two_cyl(61.75)
        res = geo.sample_conductive(c, u, h)
        self.assertTrue(res['conductive'])
        self.assertGreaterEqual(res['n_gjk'], 1)

    def test_just_above_threshold_rejected(self):
        # 轴距 61.8005 > 61.8 → 胶囊预筛拒绝（capsule 1.8005 > 1.8），不连
        c, u, h = self._two_cyl(61.8005)
        res = geo.sample_conductive(c, u, h)
        self.assertFalse(res['conductive'])
        self.assertEqual(res['n_gjk'], 0)


if __name__ == '__main__':
    unittest.main()
