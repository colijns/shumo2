"""问题1 单元测试：几何（球/圆柱/GJK/线段/支撑）、周期边界、电极、图连通。

对应规格 docs/问题1.md 第 14 节验证方法 1-9 条。
参考值全部用解析公式手算，不依赖被测实现。
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (R, DELTA, L, HALF_L,
                  gjk_distance, support_cylinder, segment_distance,
                  cylinder_aabb, pbc_candidate_k, aabb_pbc_overlap,
                  electrode_contact, analyze_group, UnionFind,
                  bfs_witness_path)


def cyl(c, u, h, r=R):
    """构造圆柱参数（c, u, h），u 自动归一化。"""
    u = np.asarray(u, dtype=float)
    return np.asarray(c, dtype=float), u / np.linalg.norm(u), float(h)


def ref_point_cylinder_dist(p, c, u, h, r):
    """点到有限平端圆柱体集合的解析距离。

    投影到轴线段（clamp），轴向在段内用侧面 max(rho-r, 0)；
    轴向超出端点时用端面圆盘距离 sqrt(dx^2 + max(rho-r,0)^2)。
    """
    p = np.asarray(p, dtype=float)
    pc = p - c
    t_un = float(pc @ u)
    rho = np.linalg.norm(pc - t_un * u)  # 真径向分量（⊥u）
    if -h <= t_un <= h:  # 投影在轴段内 → 侧面
        return max(rho - r, 0.0)
    dx = abs(t_un) - h  # 超出端面的轴向量
    return np.sqrt(dx * dx + max(rho - r, 0.0) ** 2)


class TestSphereSphere(unittest.TestCase):
    """第 1 条：球—球距离与解析公式 |c1-c2|-2r 一致（h<=0 走球分支）。"""

    def test_separated_axis(self):
        d = gjk_distance(np.zeros(3), np.array([1., 0, 0]), 0.0,
                         np.array([200., 0, 0]), np.array([1., 0, 0]), 0.0)
        self.assertAlmostEqual(d, 200.0 - 2 * R, places=6)

    def test_separated_diagonal(self):
        c2 = np.array([100., 100., 100.])
        d = gjk_distance(np.zeros(3), np.array([1., 0, 0]), 0.0,
                         c2, np.array([0., 1, 0]), 0.0)
        self.assertAlmostEqual(d, np.linalg.norm(c2) - 2 * R, places=6)

    def test_overlapping(self):
        d = gjk_distance(np.zeros(3), np.array([1., 0, 0]), 0.0,
                         np.array([50., 0, 0]), np.array([1., 0, 0]), 0.0)
        self.assertEqual(d, 0.0)

    def test_concentric(self):
        d = gjk_distance(np.zeros(3), np.array([1., 0, 0]), 0.0,
                         np.zeros(3), np.array([0., 1, 0]), 0.0)
        self.assertEqual(d, 0.0)


class TestPointCylinderGeometry(unittest.TestCase):
    """第 2 条：点到有限平端圆柱的侧面/端面距离（解析对照）。"""

    def test_point_to_cylinder_distances(self):
        c, u, h = cyl([0, 0, 0], [1, 0, 0], 100.0)
        cases = [
            ([0., 80., 0.], 50.0),          # 侧面：径向 80-30
            ([150., 0., 0.], 50.0),         # 端面正上方：轴向 50
            ([150., 80., 0.], np.sqrt(50**2 + 50**2)),  # 角：端面圆盘边缘
            ([0., 0., 0.], 0.0),            # 内部
            ([-100., 30., 0.], 0.0),        # 侧面贴边
        ]
        for p, expect in cases:
            got = ref_point_cylinder_dist(np.asarray(p, float), c, u, h, R)
            self.assertAlmostEqual(got, expect, places=9, msg=f"p={p}")

    def test_support_matches_definition(self):
        """支撑点满足规格 §6.1：v·S(v) == v·c + h|v·u| + r||v-(v·u)u||。
        随机 20 个方向核对（含与轴平行/垂直的退化方向）。"""
        rng = np.random.default_rng(7)
        c, u, h = cyl([10, -20, 30], [1, 2, -1], 500.0)
        # 与 u 正交的径向基（合法取样方向）
        e1 = np.array([u[1], -u[0], 0.0])
        if np.linalg.norm(e1) < 1e-9:
            e1 = np.array([0.0, u[2], -u[1]])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        radials = [np.zeros(3), R * e1, -R * e1, R * e2, -R * e2]
        dirs = [np.array([1., 0, 0]), np.array([0., 1, 0]),
                np.array([1., 1, 1.]), np.array([0., 0, 1.])]
        dirs += [rng.normal(size=3) for _ in range(16)]
        for v in dirs:
            v = v / np.linalg.norm(v)
            s = support_cylinder(v, c, u, h)
            h_expect = (v @ c + h * abs(v @ u)
                        + R * np.linalg.norm(v - (v @ u) * u))
            self.assertAlmostEqual(v @ s, h_expect, places=6, msg=f"v={v}")
            # 支撑性质：对任意取样点 x∈C，v·x <= v·S(v)
            for t in (-h, 0.0, h):
                for q in radials:
                    x = c + t * u + q
                    self.assertLessEqual(v @ x, v @ s + 1e-9)


class TestCylinderSphere(unittest.TestCase):
    """第 3 条：圆柱—球距离 = max(球心到圆柱体距离 - r, 0)，侧/端/角三类。"""

    def setUp(self):
        self.c, self.u, self.h = cyl([0, 0, 0], [1, 0, 0], 100.0)

    def _dist(self, center):
        return gjk_distance(self.c, self.u, self.h,
                            np.asarray(center, float), np.array([0., 1, 0]), 0.0)

    def test_side(self):
        # 球心径向 100（投影在轴段内）：圆柱体距离 70，再减球半径 → 40
        self.assertAlmostEqual(self._dist([0., 100., 0.]), 40.0, places=6)

    def test_flat_end(self):
        # 球心在轴线上、端面外 100：圆柱体距离 100，减球半径 → 70
        self.assertAlmostEqual(self._dist([200., 0., 0.]), 70.0, places=6)

    def test_corner(self):
        # 球心 (150, 80, 0)：端面圆盘边缘最近点 (100, 30, 0)
        expect = np.sqrt(50.0**2 + 50.0**2) - R
        self.assertAlmostEqual(self._dist([150., 80., 0.]), expect, places=6)

    def test_overlapping(self):
        # 球心在圆柱内 → 0
        self.assertEqual(self._dist([0., 20., 0.]), 0.0)
        # 球心在端面内（轴向超出但径向贴面）→ 0
        self.assertEqual(self._dist([120., 0., 0.]), 0.0)


class TestParallelCylinders(unittest.TestCase):
    """第 4 条：平行/斜交有限圆柱的 GJK 距离与已知结果一致。"""

    def test_parallel_axis_gap(self):
        d = gjk_distance(*cyl([0, 0, 0], [1, 0, 0], 100.0),
                         *cyl([0, 200, 0], [1, 0, 0], 100.0))
        self.assertAlmostEqual(d, 200.0 - 2 * R, places=6)

    def test_perpendicular_skew(self):
        # 轴线垂直交错，最近点均在轴段内部：轴线距 200 → 表面距 140
        d = gjk_distance(*cyl([0, 0, 0], [1, 0, 0], 100.0),
                         *cyl([0, 0, 200], [0, 1, 0], 100.0))
        self.assertAlmostEqual(d, 200.0 - 2 * R, places=6)

    def test_coaxial_end_gap(self):
        # 共轴端面相对，端面距 50（< 60 半径和）：同心圆盘最近点即圆心
        d = gjk_distance(*cyl([0, 0, 0], [1, 0, 0], 100.0),
                         *cyl([250, 0, 0], [1, 0, 0], 100.0))
        self.assertAlmostEqual(d, 50.0, places=6)

    def test_gap_below_delta_is_zero_or_small(self):
        # 平行、轴向重叠、轴线距 58（表面距 -2 → 相交/重叠）→ 0
        d = gjk_distance(*cyl([0, 0, 0], [1, 0, 0], 100.0),
                         *cyl([0, 58, 0], [1, 0, 0], 100.0))
        self.assertEqual(d, 0.0)


class TestIntersectingCylinders(unittest.TestCase):
    """第 5 条：两个圆柱相交时距离为 0。"""

    def test_axis_crossing(self):
        d = gjk_distance(*cyl([0, 0, 0], [1, 0, 0], 100.0),
                         *cyl([0, 0, 0], [0, 1, 0], 100.0))
        self.assertEqual(d, 0.0)

    def test_shared_endpoint(self):
        # 组1 构型：A1 段 [-5000,-2588]，A12 平移 (-10000,0,0) 后段
        # [-7412,-5000]，共享端点 x=-5000 → 相交 → 0
        d = gjk_distance(*cyl([-3794, 0, 0], [1, 0, 0], 1206.0),
                         *cyl([-6206, 0, 0], [1, 0, 0], 1206.0))
        self.assertEqual(d, 0.0)

    def test_contained(self):
        # 小圆柱完全位于大圆柱内部
        d = gjk_distance(*cyl([0, 0, 0], [1, 0, 0], 20.0),
                         *cyl([0, 0, 0], [1, 0, 0], 100.0))
        self.assertEqual(d, 0.0)


class TestPeriodicNeighbors(unittest.TestCase):
    """第 6 条：相对边界两侧的介质在 PBC 下识别为近邻（文档 §5.3 构型）。"""

    def setUp(self):
        # A1：(-5000,0,0)→(-2588,0,0)；A12 原样 (5000,0,0)→(2412,0,0)，
        # 镜像平移 (-10000,0,0) 后左端 (-5000,0,0) 与 A1 左端完全重合。
        self.c = np.array([[-3794., 0, 0], [3794., 0, 0]])
        self.u = np.tile([1., 0, 0], (2, 1))
        self.h = np.array([1206., 1206.])

    def test_pbc_contact_distance_zero(self):
        res = analyze_group(self.c, self.u, self.h, pbc=True)
        self.assertTrue(res['conductive'])
        self.assertEqual(len(res['contact_edges']), 1)
        i, j, k, d = res['contact_edges'][0]
        self.assertEqual((i, j), (0, 1))
        self.assertEqual(tuple(k), (-1, 0, 0))
        self.assertAlmostEqual(d, 0.0, places=6)
        self.assertEqual(res['witness_path'], ['S', 'A1', 'A2', 'T'])

    def test_pbc_candidate_k_contains_minus_x(self):
        # 约定 dc = c_i - c_j（analyze_group 内部一致）：c0 - c1 = (-7588,0,0)
        kset = pbc_candidate_k(self.c[0] - self.c[1])
        self.assertIn((-1, 0, 0), {tuple(k) for k in kset})

    def test_aabb_pbc_overlap(self):
        lo, hi = cylinder_aabb(self.c, self.u, self.h)
        self.assertTrue(aabb_pbc_overlap(lo[0], hi[0], lo[1], hi[1], (-1, 0, 0)))
        self.assertFalse(aabb_pbc_overlap(lo[0], hi[0], lo[1], hi[1], (0, 0, 0)))


class TestPBCOff(unittest.TestCase):
    """第 7 条：关闭 PBC 后不会错误连边。"""

    def setUp(self):
        self.c = np.array([[-3794., 0, 0], [3794., 0, 0]])
        self.u = np.tile([1., 0, 0], (2, 1))
        self.h = np.array([1206., 1206.])

    def test_pbc_off_no_edges(self):
        # 统一主口径默认关闭周期自动接触。
        res = analyze_group(self.c, self.u, self.h)
        self.assertEqual(len(res['contact_edges']), 0)
        self.assertFalse(res['conductive'])

    def test_pbc_off_baseline_edges(self):
        # 普通近邻关 PBC 仍连边：轴线距 60 → 表面距 0（临界接触）
        c = np.array([[0., 0, 0], [0., 60, 0]])
        u = np.tile([1., 0, 0], (2, 1))
        h = np.full(2, 100.0)
        res = analyze_group(c, u, h, pbc=False)
        self.assertEqual(len(res['contact_edges']), 1)


class TestUnionFindBFS(unittest.TestCase):
    """第 8 条：并查集结果与 BFS 路径一致。"""

    def _check(self, n, edges, left, right, conductive):
        uf = UnionFind(n + 2)
        s, t = n, n + 1
        for i in left:
            uf.union(s, i)
        for i in right:
            uf.union(t, i)
        adj = {}
        for a, b in edges:
            uf.union(a, b)
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        self.assertEqual(uf.connected(s, t), conductive)

        path = bfs_witness_path(n, adj, left, right)
        if conductive:
            self.assertIsNotNone(path)
            self.assertEqual(path[0], 'S')
            self.assertEqual(path[-1], 'T')
            nodes = path[1:-1]
            for prev, nxt in zip(nodes, nodes[1:]):
                a, b = int(prev[1:]) - 1, int(nxt[1:]) - 1
                self.assertIn(b, adj.get(a, set()))
        else:
            self.assertIsNone(path)

    def test_connected_with_cycle(self):
        self._check(5, [(0, 1), (1, 2), (2, 3), (3, 1), (3, 4)], [0], [4], True)

    def test_disconnected(self):
        self._check(4, [(0, 1), (2, 3)], [0], [3], False)

    def test_single_medium_touches_both(self):
        # 一个介质同时接触左右两面
        self._check(1, [], [0], [0], True)

    def test_no_edges(self):
        self._check(3, [], [0], [2], False)


class TestElectrodeContact(unittest.TestCase):
    """第 9 条：左右电极接触判定正确。"""

    def test_left_contact(self):
        # 贴左面：x_min = -5090 <= -5000+1.8
        c, u, h = cyl([-4990, 0, 0], [1, 0, 0], 100.0)
        self.assertTrue(electrode_contact(c, u, h, side='L'))
        self.assertFalse(electrode_contact(c, u, h, side='R'))

    def test_exactly_at_delta(self):
        # x_min 恰为 -4998.2：<= 成立（含等号）
        c = np.array([-4898.2, 0, 0.])
        self.assertTrue(electrode_contact(c, np.array([1., 0, 0]), 100.0, side='L'))
        # 右面：x_max 恰为 4998.2
        c = np.array([4898.2, 0, 0.])
        self.assertTrue(electrode_contact(c, np.array([1., 0, 0]), 100.0, side='R'))

    def test_inside_not_contact(self):
        c, u, h = cyl([0, 0, 0], [1, 0, 0], 100.0)
        self.assertFalse(electrode_contact(c, u, h, side='L'))
        self.assertFalse(electrode_contact(c, u, h, side='R'))

    def test_parallel_to_electrode(self):
        # 轴线垂直于 x（沿 y）：投影半宽 e_x = r，c_x = -5028 → x_min = -5058
        c, u, h = cyl([-5028, 0, 0], [0, 1, 0], 100.0)
        self.assertTrue(electrode_contact(c, u, h, side='L'))

    def test_slightly_inside(self):
        # x_min = -4997.5 > -4998.2：不接触（1.5 nm 间隙）
        c, u, h = cyl([-4897.5, 0, 0], [1, 0, 0], 100.0)
        self.assertFalse(electrode_contact(c, u, h, side='L'))


class TestSegmentDistance(unittest.TestCase):
    """线段距离解析对照（Ericson 5.1.9 已知构型）。"""

    def test_crossing(self):
        d = segment_distance(np.array([0., 0, 0]), np.array([10., 0, 0]),
                             np.array([5., -5, 0]), np.array([5., 5, 0]))
        self.assertAlmostEqual(d, 0.0, places=9)

    def test_parallel_offset(self):
        d = segment_distance(np.array([0., 0, 0]), np.array([10., 0, 0]),
                             np.array([0., 7, 0]), np.array([10., 7, 0]))
        self.assertAlmostEqual(d, 7.0, places=9)

    def test_endpoint_gap(self):
        # 端点相对：最近距离为两点间距
        d = segment_distance(np.array([0., 0, 0]), np.array([10., 0, 0]),
                             np.array([30., 0, 0]), np.array([40., 0, 0]))
        self.assertAlmostEqual(d, 20.0, places=9)

    def test_point_above_end(self):
        d = segment_distance(np.array([0., 0, 0]), np.array([10., 0, 0]),
                             np.array([15., 4, 0]), np.array([15., 4, 0]))
        self.assertAlmostEqual(d, np.hypot(5.0, 4.0), places=9)


if __name__ == '__main__':
    unittest.main(verbosity=2)
