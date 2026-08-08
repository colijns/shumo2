# 本程序及代码是在AI工具辅助下完成的
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31

"""问题4 几何内核测试：球生成/截断、三类接触、电极距离、混合导通。"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'Q2'))
import geometry as geo      # noqa: E402
import geometry_mix as gm   # noqa: E402


class TestConstants(unittest.TestCase):
    """成本派生值与题目换算一致（相对 1% 容差）。"""

    def test_single_piece_cost(self):
        self.assertAlmostEqual(gm.c_A, 1.4844e-2, delta=1.4844e-4)
        self.assertAlmostEqual(gm.c_B, 1.6755e-3, delta=1.6755e-5)

    def test_volume_ratios(self):
        # V_B/V_A = (4/3 π r_B³)/(π r_A² l_A)
        ratio = gm.V_B_NM3 / gm.V_A_NM3
        expect = (4.0 / 3.0 * 200.0 ** 3) / (30.0 ** 2 * 5000.0)
        self.assertAlmostEqual(ratio, expect, places=9)

    def test_cost_function(self):
        na, nb = 3, 5
        self.assertAlmostEqual(gm.cost(na, nb), gm.c_A * na + gm.c_B * nb)


class TestBalls(unittest.TestCase):

    def test_generate_range(self):
        rng = np.random.default_rng(42)
        cs = gm.generate_balls(500, rng)
        self.assertEqual(cs.shape, (500, 3))
        self.assertTrue(np.all(cs >= -gm.HALF_L) and np.all(cs <= gm.HALF_L))

    def test_clip_inside_single(self):
        cs = np.array([[0.0, 0.0, 0.0]])
        r = gm.clip_balls(cs)
        self.assertEqual(len(r['c_img']), 1)
        self.assertEqual(r['ball_idx'].tolist(), [0])

    def test_clip_single_axis_cross(self):
        # x 负向越界：c_x - r_B < -5000
        cs = np.array([[-4910.0, 0.0, 0.0]])
        r = gm.clip_balls(cs)
        self.assertEqual(len(r['c_img']), 2)
        # 本体 + 平移 +L 片段
        moved = r['c_img'][1]
        np.testing.assert_allclose(moved, [-4910.0 + gm.L, 0.0, 0.0])

    def test_clip_two_axis_cross_three(self):
        # 双轴越界（角部）→ 3 片段，全落回扩张盒语义内
        cs = np.array([[-4910.0, -4910.0, 0.0]])
        r = gm.clip_balls(cs)
        self.assertEqual(len(r['c_img']), 3)
        self.assertEqual(sorted(r['ball_idx'].tolist()), [0, 0, 0])

    def test_clip_three_axis_cross_four(self):
        cs = np.array([[-4910.0, -4910.0, -4910.0]])
        r = gm.clip_balls(cs)
        self.assertEqual(len(r['c_img']), 4)
        # 完整球模型：平移片段球心可出盒，但每片段与盒相交（半径 200 部分进入）
        self.assertTrue(np.all(r['c_img'] - gm.R_B <= gm.HALF_L + 1e-9))
        self.assertTrue(np.all(r['c_img'] + gm.R_B >= -gm.HALF_L - 1e-9))

    def test_clip_max_four(self):
        # 2r_B=400 < L：同轴不可能双向越界，片段数 ≤ 4
        rng = np.random.default_rng(7)
        cs = gm.generate_balls(3000, rng)
        r = gm.clip_balls(cs)
        idx = r['ball_idx']
        counts = np.bincount(idx, minlength=len(cs))
        self.assertTrue(np.all(counts <= 4))
        self.assertEqual(counts.sum(), len(r['c_img']))


class TestDistances(unittest.TestCase):

    def test_sphere_electrode(self):
        dL, dR = gm.sphere_electrode_dist(np.array([[4500.0, 0.0, 0.0]]))
        self.assertAlmostEqual(dR[0], gm.HALF_L - 4500.0 - gm.R_B)   # 300
        self.assertAlmostEqual(dL[0], 4500.0 - gm.R_B + gm.HALF_L)   # 9300
        # 穿过电极面 → clamp 0
        dL, dR = gm.sphere_electrode_dist(np.array([[4900.0, 0.0, 0.0]]))
        self.assertEqual(dR[0], 0.0)
        self.assertGreater(dL[0], 0.0)

    def test_bb_threshold(self):
        # 球心距恰 401.8 = 2r_B + δ → 接触；402.0 → 不接触
        pairs = gm.bb_pairs(np.array([[0.0, 0.0, 0.0], [gm.THRESH_BB, 0.0, 0.0]]))
        self.assertEqual(len(pairs), 1)
        pairs = gm.bb_pairs(np.array([[0.0, 0.0, 0.0], [402.0, 0.0, 0.0]]))
        self.assertEqual(len(pairs), 0)

    def test_ab_distance_threshold(self):
        # 圆柱轴段 x 向 (0,0,0)-(5000,0,0)；球心正上方 431.8 = r_A+r_B+δ → 接触
        p1s = np.array([[0.0, 0.0, 0.0]])
        p2s = np.array([[5000.0, 0.0, 0.0]])
        a_lo = np.minimum(p1s, p2s) - gm.R_A
        a_hi = np.maximum(p1s, p2s) + gm.R_A
        ball = np.array([[0.0, 0.0, gm.THRESH_AB],      # 距离恰阈值 → 连
                         [0.0, 0.0, gm.THRESH_AB + 0.2],  # 超阈值 → 不连
                         [-200.0, 0.0, 0.0]])           # 端面外 200 → 连
        b_lo = ball - gm.R_B
        b_hi = ball + gm.R_B
        edges = gm._ab_edges_all(p1s, p2s, a_lo, a_hi, ball, b_lo, b_hi)
        got = set(map(tuple, edges.tolist()))
        self.assertIn((0, 0), got)      # (A 段 id, B 片段 id)
        self.assertIn((0, 2), got)
        self.assertNotIn((0, 1), got)

    def test_aa_gjk_threshold(self):
        # 两根平行 x 向圆柱，轴距 61.8 → 表面距离 1.8 = δ 接触；62 → 不接触
        p1s = np.array([[-2500.0, 0.0, 0.0], [-2500.0, 62.0, 0.0]])
        p2s = np.array([[2500.0, 0.0, 0.0], [2500.0, 62.0, 0.0]])
        lo = np.minimum(p1s, p2s) - gm.R_A
        hi = np.maximum(p1s, p2s) + gm.R_A
        edges = gm._aa_edges_all(p1s, p2s, lo, hi)
        self.assertEqual(len(edges), 0)   # 62 - 60 = 2 > 1.8

        # 轴距 61.8：c2 中心 (0, 61.8, 0) → 表面距 1.8 = δ → GJK 精算连边
        p1s = np.array([[-2500.0, 0.0, 0.0], [-2500.0, 61.8, 0.0]])
        p2s = np.array([[2500.0, 0.0, 0.0], [2500.0, 61.8, 0.0]])
        lo = np.minimum(p1s, p2s) - gm.R_A
        hi = np.maximum(p1s, p2s) + gm.R_A
        edges = gm._aa_edges_all(p1s, p2s, lo, hi)
        self.assertEqual(len(edges), 1)


class TestCrossingRatio(unittest.TestCase):

    def test_ball_crossing_ratio(self):
        rng = np.random.default_rng(42)
        cs = gm.generate_balls(3000, rng)
        r = gm.ball_crossing_ratio(cs)
        # 理论 1-(1-2r_B/L)³ ≈ 11.53%，容差 2pp
        theory = 1.0 - (1.0 - 2.0 * gm.R_B / gm.L) ** 3
        self.assertLess(abs(r - theory), 0.02)


class TestConductivity(unittest.TestCase):

    def test_sample_mixed_reproducible(self):
        rng = np.random.default_rng(42)
        c, u, h = geo.generate_cylinders(60, rng)
        balls = gm.generate_balls(100, rng)
        r1 = gm.sample_mixed(c, u, h, balls)
        r2 = gm.sample_mixed(c, u, h, balls)
        self.assertEqual(r1['conductive'], r2['conductive'])

    def test_prefix_monotonic_small(self):
        """共同随机数下 Y_s 对任一维 +1 单调不减（20 点小样本逐 trial 断言）。"""
        rng = np.random.default_rng(123)
        c, u, h = geo.generate_cylinders(80, rng)
        balls = gm.generate_balls(120, rng)
        pr = gm.prepare_trial(c, u, h, balls, 120)
        pts = [(40, 60), (60, 60), (40, 100), (60, 100)]
        y = [gm.sample_prefix(pr, *p) for p in pts]
        # y[0] ≤ y[1]（na+1）、y[0] ≤ y[2]（nb+1）、y[1] ≤ y[3]（nb+1）、y[2] ≤ y[3]
        for i, j in ((0, 1), (0, 2), (1, 3), (2, 3)):
            self.assertLessEqual(int(y[i]), int(y[j]))

    def test_first_contact_nb_never_conductive_small(self):
        # 50 个球不足以导通（理论连通密度远低于临界）
        rng = np.random.default_rng(7)
        nc = gm.first_contact_nb(gm.generate_balls(50, rng))
        self.assertIsNone(nc)


if __name__ == '__main__':
    unittest.main(verbosity=2)
