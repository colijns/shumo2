# 本程序及代码是在AI工具辅助下完成的
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31

"""蒙特卡洛 + 边界截断片段独立语义 + 构造导通链测试。"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import geometry as geo  # noqa: E402
import monte_carlo as mc  # noqa: E402


class TestMonteCarloCore(unittest.TestCase):

    def test_n_cylinders(self):
        self.assertEqual(mc.n_cylinders(0.005), 354)
        self.assertEqual(mc.n_cylinders(0.006), 424)
        self.assertEqual(mc.n_cylinders(0.007), 495)
        self.assertEqual(mc.n_cylinders(0.01), 707)

    def test_reproducible_same_seed(self):
        r1 = mc.simulate_phi(0.01, m=3, seed=42)
        r2 = mc.simulate_phi(0.01, m=3, seed=42)
        self.assertEqual(r1['x'], r2['x'])
        self.assertEqual(r1['mean_fragments'], r2['mean_fragments'])

    def test_different_seed_different_rng(self):
        r1 = mc.simulate_phi(0.01, m=3, seed=42)
        r2 = mc.simulate_phi(0.01, m=3, seed=43)
        # 连续统计量（平均片段数）不同 seed 必不同
        self.assertNotEqual(r1['mean_fragments'], r2['mean_fragments'])

    def test_zero_phi(self):
        res = mc.simulate_phi(0.0, m=100, seed=1)
        self.assertEqual(res['x'], 0)
        self.assertEqual(res['p_hat'], 0.0)
        self.assertEqual(res['ci_lower'], 0.0)

    def test_wilson_bounds(self):
        for x, m in [(0, 2000), (2000, 2000), (100, 2000), (1, 2000)]:
            p, lo, hi = mc.wilson_ci(x, m)
            # 浮点容差（p̂=0/1 时 lo/hi 有 ~1e-19 残差）
            self.assertGreaterEqual(lo, -1e-12)
            self.assertLessEqual(hi, 1.0 + 1e-12)
            self.assertLessEqual(lo, p + 1e-9)
            self.assertLessEqual(p, hi + 1e-9)

    def test_smoke_small_m(self):
        # 4 个 φ × M=2 冒烟，全部跑通且数值在合理域
        for i, phi in enumerate([0.005, 0.006, 0.007, 0.01]):
            res = mc.simulate_phi(phi, m=2, seed=mc.BASE_SEED + i)
            self.assertGreaterEqual(res['x'], 0)
            self.assertLessEqual(res['x'], 2)
            self.assertGreater(res['mean_fragments'], 0.0)


class TestIndependentFragments(unittest.TestCase):
    """同源编号只记录来源，不使分处相对边界的片段自动电连接。"""

    def test_single_wall_crossing_cylinder_not_automatic_conductive(self):
        # 单根跨壁圆柱：p1=(-6000,0,0) → p2=(-1000,0,0)
        # 内部段 x∈[-5000,-1000] 触左电极；wrap 段 x∈[0,4000] 触右电极；
        # 两片段仅同源但空间分离，不建立不可见导线，因此不能单根短接 S-T。
        from core import build_cylinders
        ends = np.array([[[-6000.0, 0.0, 0.0], [-1000.0, 0.0, 0.0]]])
        c, u, h = build_cylinders(ends)
        res = geo.sample_conductive(c, u, h)
        self.assertFalse(res['conductive'])
        self.assertEqual(res['n_fragments'], 2)
        self.assertEqual(res['n_left'], 1)
        self.assertEqual(res['n_right'], 1)

    def test_single_fragment_only_touches_left(self):
        # 单根圆柱完全在内部且仅触左电极，显然不导通。
        from core import build_cylinders
        ends = np.array([[[-4999.0, 0.0, 0.0], [-2500.0, 0.0, 0.0]]])
        c, u, h = build_cylinders(ends)
        res = geo.sample_conductive(c, u, h)
        self.assertFalse(res['conductive'])
        self.assertEqual(res['n_fragments'], 1)
        self.assertEqual(res['n_left'], 1)
        self.assertEqual(res['n_right'], 0)


class TestConstructedChain(unittest.TestCase):
    # cyl1 触左（轴段 x∈[-4999,1]，dL=1），cyl2 触右（轴段 x∈[-1,4999]，dR=1），
    # 平行相距 axis_d：平端距离 = 胶囊距离 = max(0, axis_d-60) 精确

    def test_two_cylinder_chain(self):
        # 轴距 20 → 距离 0 → 连：S-cyl1-cyl2-T → Y=1
        c = np.array([[-2499.0, 0.0, 0.0], [2499.0, 20.0, 0.0]])
        u = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        h = np.array([2500.0, 2500.0])
        res = geo.sample_conductive(c, u, h)
        self.assertTrue(res['conductive'])
        self.assertGreaterEqual(res['n_edges'], 1)
        self.assertGreaterEqual(res['n_left'], 1)
        self.assertGreaterEqual(res['n_right'], 1)

    def test_chain_broken_by_gap(self):
        # 轴距 80 > 61.8 → 不导通
        c = np.array([[-2499.0, 0.0, 0.0], [2499.0, 80.0, 0.0]])
        u = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        h = np.array([2500.0, 2500.0])
        res = geo.sample_conductive(c, u, h)
        self.assertFalse(res['conductive'])

    def test_gjk_band_counts(self):
        # 轴距 61.5 链（距离 1.5）→ n_gjk ≥ 1 且连
        c = np.array([[-2499.0, 0.0, 0.0], [2499.0, 61.5, 0.0]])
        u = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        h = np.array([2500.0, 2500.0])
        res = geo.sample_conductive(c, u, h)
        self.assertTrue(res['conductive'])
        self.assertGreaterEqual(res['n_gjk'], 1)


if __name__ == '__main__':
    unittest.main()
