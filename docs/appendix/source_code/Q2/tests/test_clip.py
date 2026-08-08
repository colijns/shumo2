# 本程序及代码是在AI工具辅助下完成的
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31

"""clip_cylinder / clip_batch 截断正确性测试。"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import geometry as geo  # noqa: E402


class TestClip(unittest.TestCase):

    def test_inside_single_fragment(self):
        p1 = np.array([-1000.0, 0.0, 0.0])
        p2 = np.array([3000.0, 0.0, 0.0])
        frags = geo.clip_cylinder(p1, p2)
        self.assertEqual(len(frags), 1)
        np.testing.assert_allclose(frags[0][0], p1)
        np.testing.assert_allclose(frags[0][1], p2)

    def test_right_wall_wrap(self):
        # p1=(6000,0,0) → p2=(4000,0,0)：外部段平移后 x∈(-5000,-4000)
        p1 = np.array([6000.0, 0.0, 0.0])
        p2 = np.array([4000.0, 0.0, 0.0])
        frags = geo.clip_cylinder(p1, p2)
        self.assertEqual(len(frags), 2)
        # 内部段 (5000,0,0)-(4000,0,0)
        # 内部段两端点 x ≥ -5000；wrap 段有端点 x ≤ -5000（平移后 {-5000,-4000}）
        inside = [f for f in frags
                  if min(f[0][0], f[1][0]) >= -geo.HALF_L + 1e-9][0]
        wrapped = [f for f in frags
                   if min(f[0][0], f[1][0]) <= -geo.HALF_L + 1e-9][0]
        np.testing.assert_allclose(inside[0], [5000.0, 0.0, 0.0])
        np.testing.assert_allclose(inside[1], [4000.0, 0.0, 0.0])
        seg_xs = np.sort([wrapped[0][0], wrapped[1][0]])
        np.testing.assert_allclose(seg_xs, [-5000.0, -4000.0])

    def test_two_plane_crossing_three_fragments(self):
        # 穿越两个轴平面（x=5000 与 y=5000）→ 3 片段，全部落回 box 内且长度守恒
        p1 = np.array([6000.0, 0.0, 0.0])
        p2 = np.array([0.0, 6000.0, 0.0])
        frags = geo.clip_cylinder(p1, p2)
        self.assertEqual(len(frags), 3)
        total = 0.0
        for f1, f2 in frags:
            self.assertTrue(np.all(f1 >= -geo.HALF_L - geo.BOX_TOL) and
                            np.all(f1 <= geo.HALF_L + geo.BOX_TOL))
            self.assertTrue(np.all(f2 >= -geo.HALF_L - geo.BOX_TOL) and
                            np.all(f2 <= geo.HALF_L + geo.BOX_TOL))
            total += np.linalg.norm(f2 - f1)
        # 原始轴段长度 √(6000²+6000²)（非 CYL_LEN，本例轴段端点差为 (6000,6000,0)）
        self.assertAlmostEqual(total, np.linalg.norm(p2 - p1), places=5)

    def test_four_fragment_case(self):
        # 数值验证过的 4 片段案例
        c = np.array([4274.2, 4679.3, -4852.9])
        u = np.array([-0.625, 0.405, 0.667])
        u = u / np.linalg.norm(u)
        h = geo.CYL_HALF_LEN
        frags = geo.clip_cylinder(c - h * u, c + h * u)
        self.assertEqual(len(frags), 4)

    def test_fully_outside_single_wrapped(self):
        # 整段界外 → 1 片段平移回界内
        p1 = np.array([6000.0, 0.0, 0.0])
        p2 = np.array([9000.0, 0.0, 0.0])
        frags = geo.clip_cylinder(p1, p2)
        self.assertEqual(len(frags), 1)
        f1, f2 = frags[0]
        self.assertTrue(f1[0] >= -geo.HALF_L - geo.BOX_TOL and f2[0] <= geo.HALF_L + geo.BOX_TOL)
        self.assertAlmostEqual(f2[0] - f1[0], 3000.0, places=6)

    def test_endpoint_on_plane_no_crossing(self):
        # 端点恰在平面（t=0）不穿越
        p1 = np.array([5000.0, 0.0, 0.0])
        p2 = np.array([3000.0, 0.0, 0.0])
        frags = geo.clip_cylinder(p1, p2)
        self.assertEqual(len(frags), 1)

    def test_random_10000_invariants(self):
        # 随机压力：每根 ≤4 片段、片段全在 box 内、长度守恒 5000
        rng = np.random.default_rng(7)
        c, u, h = geo.generate_cylinders(10000, rng)
        for i in range(10000):
            frags = geo.clip_cylinder(c[i] - h[i] * u[i], c[i] + h[i] * u[i])
            self.assertLessEqual(len(frags), 4, msg=f'cyl {i}: {len(frags)} 片段')
            total = 0.0
            for f1, f2 in frags:
                self.assertTrue(np.all(f1 >= -geo.HALF_L - geo.BOX_TOL) and
                                np.all(f1 <= geo.HALF_L + geo.BOX_TOL))
                self.assertTrue(np.all(f2 >= -geo.HALF_L - geo.BOX_TOL) and
                                np.all(f2 <= geo.HALF_L + geo.BOX_TOL))
                total += np.linalg.norm(f2 - f1)
            self.assertAlmostEqual(total, geo.CYL_LEN, places=5,
                                   msg=f'cyl {i} 长度不守恒')


class TestClipBatch(unittest.TestCase):

    def test_batch_matches_scalar(self):
        rng = np.random.default_rng(11)
        c, u, h = geo.generate_cylinders(200, rng)
        frag = geo.clip_batch(c, u, h)
        # 逐根对拍
        n_scalar = 0
        for i in range(200):
            frags = geo.clip_cylinder(c[i] - h[i] * u[i], c[i] + h[i] * u[i])
            n_scalar += len(frags)
        self.assertEqual(len(frag['p1s']), n_scalar)
        self.assertEqual(frag['cyl_idx'].shape, (n_scalar,))
        # 同源索引正确性：idx 单调非降且总数 = 每根片段数
        counts = np.bincount(frag['cyl_idx'], minlength=200)
        for i in range(200):
            self.assertEqual(counts[i], len(geo.clip_cylinder(
                c[i] - h[i] * u[i], c[i] + h[i] * u[i])))


if __name__ == '__main__':
    unittest.main()
