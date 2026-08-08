"""问题1集成测试：片段独立主结果、周期接触敏感性、组1穷举对照。

固定期望值来自统一口径复算，并由组1全量穷举交叉验证。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import DELTA, gjk_distance, analyze_group, load_cylinders

XLSX = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'attachment', '附件.xlsx')

# 统一口径主结果：只按片段在当前微构体内的实际位置判断接触。
EXPECT = {
    0: dict(name='组1', n=12, left=3, right=4, aabb=5, edges=2,
            maxcomp=4, path=None, conductive=False),
    1: dict(name='组2', n=49, left=11, right=11, aabb=129, edges=27,
            maxcomp=34, path=['S', 'A2', 'A12', 'A24', 'A39', 'T'],
            conductive=True),
    2: dict(name='组3', n=535, left=92, right=90, aabb=3926, edges=165,
            maxcomp=258, path=['S', 'A63', 'A264', 'A216', 'A351', 'T'],
            conductive=True),
}
# 周期镜像自动接触仅作敏感性对照。
EXPECT_PERIODIC = {0: (5, True), 1: (39, True), 2: (344, True)}


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
        res = analyze_group(c, u, h, pbc=False)
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
        if not res['conductive']:
            self.assertIsNone(path, f"{exp['name']} 不导通时不应存在见证路径")
            return
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
    """周期镜像接触只作为对照，不参与统一主结论。"""

    def test_periodic_contact_sensitivity(self):
        for idx, (n_edges, conductive) in EXPECT_PERIODIC.items():
            c, u, h, _ = load_cylinders(XLSX, idx)
            res = analyze_group(c, u, h, pbc=True)
            self.assertEqual(len(res['contact_edges']), n_edges,
                             f"{EXPECT[idx]['name']} 周期对照边数")
            self.assertEqual(res['conductive'], conductive,
                             f"{EXPECT[idx]['name']} 周期对照导电状态")

    def test_group1_changes_under_periodic_contact(self):
        """组1是口径敏感样本：主口径不导通，周期对照导通。"""
        c, u, h, _ = load_cylinders(XLSX, 0)
        self.assertFalse(analyze_group(c, u, h, pbc=False)['conductive'])
        self.assertTrue(analyze_group(c, u, h, pbc=True)['conductive'])


class TestExhaustiveVsAccelerated(unittest.TestCase):
    """组1穷举当前盒内全部66个介质对，与加速管线完全一致。"""

    def test_group1_exhaustive(self):
        c, u, h, _ = load_cylinders(XLSX, 0)
        n = len(c)
        res = analyze_group(c, u, h, pbc=False)
        accel = {(e[0], e[1]) for e in res['contact_edges']}

        # 穷举：只计算当前盒内实际位置，不加入周期镜像。
        brute = set()
        for i in range(n):
            for j in range(i + 1, n):
                d = gjk_distance(c[i], u[i], h[i], c[j], u[j], h[j])
                if d <= DELTA:
                    brute.add((i, j))
        self.assertEqual(brute, accel,
                         f"穷举 {n * (n - 1) // 2} 个介质对与加速管线不一致")


if __name__ == '__main__':
    unittest.main(verbosity=2)
