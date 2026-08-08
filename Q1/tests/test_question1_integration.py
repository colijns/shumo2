"""问题1 集成测试：附件三组固定结果、PBC 敏感性、组1 穷举对照。

对应规格 docs/问题1.md 第 14 节验证方法 10-12 条。
期望值全部来自文档第 11/13 节结果表（唯一权威）。
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (L, R, DELTA, gjk_distance, analyze_group, load_cylinders)

XLSX = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'attachment', '附件.xlsx')

# 文档第 11 节：三组固定结果
EXPECT = {
    0: dict(name='组1', n=12, left=3, right=4, aabb=11, edges=5,
            maxcomp=8, path=['S', 'A1', 'A12', 'T'], conductive=True),
    1: dict(name='组2', n=49, left=11, right=11, aabb=154, edges=39,
            maxcomp=34, path=['S', 'A1', 'A47', 'T'], conductive=True),
    2: dict(name='组3', n=535, left=92, right=90, aabb=4688, edges=344,
            maxcomp=308, path=['S', 'A2', 'A505', 'T'], conductive=True),
}
# 文档第 13 节：PBC 敏感性（关 PBC 接触边数/导电状态）
EXPECT_OFF = {0: (2, False), 1: (27, True), 2: (165, True)}


class TestAttachmentFixedResults(unittest.TestCase):
    """第 10 条：三组介质数、接触边数和导电状态与固定结果一致。"""

    def test_group1(self):
        self._check(0)

    def test_group2(self):
        self._check(1)

    def test_group3(self):
        self._check(2)

    def _check(self, idx):
        exp = EXPECT[idx]
        c, u, h, names = load_cylinders(XLSX, idx)
        res = analyze_group(c, u, h)
        self.assertEqual(res['n_cylinders'], exp['n'], exp['name'])
        self.assertEqual(res['n_left_contact'], exp['left'], exp['name'])
        self.assertEqual(res['n_right_contact'], exp['right'], exp['name'])
        self.assertEqual(res['aabb_candidate_pairs'], exp['aabb'], exp['name'])
        self.assertEqual(len(res['contact_edges']), exp['edges'], exp['name'])
        self.assertEqual(res['max_component_size'], exp['maxcomp'], exp['name'])
        self.assertEqual(res['conductive'], exp['conductive'], exp['name'])
        self.assertEqual(res['witness_path'], exp['path'], exp['name'])
        # 见证路径逐边必须真实合法（并查集/BFS 交叉检查）：
        # S 只连左接触介质，T 只连右接触介质，介质间边必须在接触边集合
        edges = {(e[0], e[1]) for e in res['contact_edges']}
        left = set(res['left_contacts'])
        right = set(res['right_contacts'])
        path = res['witness_path']
        for a, b in zip(path, path[1:]):
            if a == 'S':
                self.assertIn(int(b[1:]) - 1, left,
                              f"{exp['name']} 路径起点 {b} 不在左接触集合")
            elif b == 'T':
                self.assertIn(int(a[1:]) - 1, right,
                              f"{exp['name']} 路径终点 {a} 不在右接触集合")
            else:
                ia, ib = int(a[1:]) - 1, int(b[1:]) - 1
                self.assertTrue((ia, ib) in edges or (ib, ia) in edges,
                                f"{exp['name']} 路径边 {a}-{b} 不在接触边集合")


class TestPBCSensitivity(unittest.TestCase):
    """第 11 条：三组 PBC 开关敏感性结果一致。"""

    def test_pbc_off(self):
        for idx, (n_off, conductive_off) in EXPECT_OFF.items():
            c, u, h, _ = load_cylinders(XLSX, idx)
            res_off = analyze_group(c, u, h, pbc=False)
            self.assertEqual(len(res_off['contact_edges']), n_off,
                             f"{EXPECT[idx]['name']} 关 PBC 边数")
            self.assertEqual(res_off['conductive'], conductive_off,
                             f"{EXPECT[idx]['name']} 关 PBC 导电状态")

    def test_group1_requires_pbc(self):
        """组1 是关键样本：关 PBC 后不导电（文档第 13 节）。"""
        c, u, h, _ = load_cylinders(XLSX, 0)
        self.assertTrue(analyze_group(c, u, h)['conductive'])
        self.assertFalse(analyze_group(c, u, h, pbc=False)['conductive'])


class TestExhaustiveVsAccelerated(unittest.TestCase):
    """第 12 条：组1 穷举全部 66×27 个"介质对—周期镜像"组合，
    纯 GJK 结果与 AABB 加速管线完全一致（粗筛不漏）。"""

    def test_group1_exhaustive(self):
        c, u, h, _ = load_cylinders(XLSX, 0)
        n = len(c)
        res = analyze_group(c, u, h)
        accel = {(e[0], e[1]) for e in res['contact_edges']}

        # 穷举：每对 27 个镜像，取最小周期距离，<= 1.8 视为接触
        ks = [(kx, ky, kz) for kx in (-1, 0, 1) for ky in (-1, 0, 1)
              for kz in (-1, 0, 1)]
        brute = set()
        for i in range(n):
            for j in range(i + 1, n):
                dmin = np.inf
                for k in ks:
                    c2k = c[j] + L * np.asarray(k, dtype=float)
                    d = gjk_distance(c[i], u[i], h[i], c2k, u[j], h[j])
                    dmin = min(dmin, d)
                if dmin <= DELTA:
                    brute.add((i, j))
        self.assertEqual(brute, accel,
                         f"穷举 {n * (n - 1) // 2 * 27} 组合与加速管线不一致")


if __name__ == '__main__':
    unittest.main(verbosity=2)
