# 本程序及代码是在AI工具辅助下完成的
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31

"""问题4 搜索层测试：成本/配比辅助、Wilson 判定、N90 估计、剪枝、分块评估一致性。"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_question4 as rq4   # noqa: E402
import geometry_mix as gm     # noqa: E402


class TestCostHelpers(unittest.TestCase):

    def test_b_lim_floor(self):
        # c0 = 3c_A + 5c_B：na=3 恰整除 → 5；na=2 剩 c_A + 5c_B → floor(c_A/c_B + 5)
        c0 = 3 * gm.c_A + 5 * gm.c_B
        self.assertEqual(rq4.b_lim(3, c0), 5)
        expect2 = int(np.floor(gm.c_A / gm.c_B + 5))
        self.assertEqual(rq4.b_lim(2, c0), expect2)

    def test_b_lim_negative(self):
        c0 = 2 * gm.c_A + 3 * gm.c_B
        self.assertLess(rq4.b_lim(10, c0), 0)

    def test_b_lim_exact_divisibility(self):
        # 整除无浮点噪声（COST_TOL 保护）
        c0 = 100 * gm.c_A + 200 * gm.c_B
        self.assertEqual(rq4.b_lim(100, c0), 200)


class TestWilsonVerdict(unittest.TestCase):

    def test_reliable(self):
        v, lo, hi = rq4.wilson_verdict(195, 200)
        self.assertEqual(v, 'reliable')
        self.assertGreaterEqual(lo, rq4.P_TARGET)

    def test_insufficient(self):
        v, lo, hi = rq4.wilson_verdict(0, 20)
        self.assertEqual(v, 'insufficient')
        self.assertLess(hi, rq4.P_TARGET)

    def test_crossing(self):
        v, lo, hi = rq4.wilson_verdict(9, 10)
        self.assertEqual(v, 'crossing')
        self.assertLess(lo, rq4.P_TARGET)
        self.assertGreaterEqual(hi, rq4.P_TARGET)


class TestN90(unittest.TestCase):

    def test_none_if_never(self):
        self.assertEqual(rq4.n90_from_contacts([None] * 5, 50, 5), (None, None))

    def test_hat_and_safe(self):
        # arr = [10]*50 + [50]*50, m=100：p(10)=0.5, p(50)=1.0
        contacts = [10] * 50 + [50] * 50
        n_hat, n_safe = rq4.n90_from_contacts(contacts, 60, 100)
        self.assertEqual(n_hat, 50)
        self.assertEqual(n_safe, 50)   # x=100/100 → Wilson lo ≈ 0.963 ≥ 0.90

    def test_safe_none_when_only_point_est(self):
        # m=20, p(40)=0.9 点估计刚过线，但 lo < 0.90
        contacts = [40] * 18 + [80] * 2
        n_hat, n_safe = rq4.n90_from_contacts(contacts, 80, 20)
        self.assertEqual(n_hat, 40)
        self.assertIsNone(n_safe)


class TestBandPoints(unittest.TestCase):

    def test_divisibility_correction(self):
        c_star = 3 * gm.c_A + 5 * gm.c_B
        pts = rq4.band_points(c_star, na_cap=10)
        # na=3 成本恰 = C* → b_lim 修正减 1；其余 na 严格便宜
        self.assertIn((3, 4), pts)
        self.assertNotIn((3, 5), pts)
        for na, nb in pts:
            self.assertLess(gm.c_A * na + gm.c_B * nb, c_star - rq4.COST_TOL)

    def test_pure_b_endpoint_included(self):
        pts = rq4.band_points(3 * gm.c_A + 5 * gm.c_B, na_cap=10)
        self.assertEqual(pts[0][0], 0)
        self.assertGreater(pts[0][1], 0)


class TestFeaPruning(unittest.TestCase):
    """剪枝分支不触发真实评估；块内扫描用极小样本。workers=1 省 spawn。"""

    @classmethod
    def setUpClass(cls):
        cls.store = rq4.EvalStore(1, 50, 60, 99)

    @classmethod
    def tearDownClass(cls):
        cls.store.close()

    def test_pure_a_prune(self):
        n_a_hat = 10
        c0 = gm.c_A * (n_a_hat + 5)
        ok, witness = self.store.fea(c0, n_a_hat, 10 ** 9, m_search=8)
        self.assertTrue(ok)
        self.assertEqual(witness[0], n_a_hat)

    def test_pure_b_prune(self):
        # 禁纯A剪枝（n_a_hat 极大），c0 内首块 na 的 b_lim 即超过 n_b_hat
        c0 = 3 * gm.c_A + 40 * gm.c_B
        ok, witness = self.store.fea(c0, 10 ** 9, 30, m_search=8)
        self.assertTrue(ok)
        self.assertGreaterEqual(witness[1], 30)

    def test_infeasible_small_layer(self):
        # 3 个球 + 0 根 A：m=8 次试验几乎必然不导通 → 层不可行（真实评估路径）
        ok, witness = self.store.fea(3 * gm.c_B, 10 ** 9, 10 ** 9, m_search=8)
        self.assertFalse(ok)
        self.assertIsNone(witness)


class TestEvalConsistency(unittest.TestCase):
    """分块合并正确性：跨 CHUNK_T/CHUNK_P 边界的点-试验矩阵与整批结果一致。"""

    @classmethod
    def setUpClass(cls):
        cls.store = rq4.EvalStore(1, 60, 80, 7)

    @classmethod
    def tearDownClass(cls):
        cls.store.close()

    def test_chunk_boundaries(self):
        # 47 点（> CHUNK_P=32）与 40 试验（> CHUNK_T=32）→ 分块 + 合并
        pts = [(i, i + 5) for i in range(47)]
        x1 = self.store.eval(pts, 40)
        x2 = rq4._eval_job((np.asarray(pts, dtype=np.int64), 60, 80, 7, 0, 40))
        np.testing.assert_array_equal(x1, x2)

    def test_incremental_append(self):
        # 分两段追加（trial 从现有计数继续）与一次跑完等价
        pts = [(10, 20), (30, 40), (50, 60)]
        x_full = self.store.eval(pts, 16)
        x_a = self.store.eval(pts, 10)
        x_b = self.store.eval(pts, 6, trial_start=10)
        np.testing.assert_array_equal(x_full, x_a + x_b)


class _StubStore:
    """内存 store 桩：ensure 只提升 m，get 返回 (x, m)。"""

    def __init__(self):
        self.data = {}

    def ensure(self, pts, target):
        for p in pts:
            d = self.data.get(p, [0, 0])
            if d[1] < target:
                self.data[p] = [d[0], target]

    def get(self, p):
        d = self.data.get(p)
        return (0, 0) if d is None else tuple(d)


class TestNaMaxCoverage(unittest.TestCase):

    def test_covers_safe(self):
        # n_safe > n_hat 时序列必须覆盖可靠纯A 点（日志失败根因）
        c_ub = gm.c_A * 611
        na_max = rq4.compute_na_max(c_ub, 611, 617, 900)
        self.assertEqual(na_max, 619)
        self.assertLessEqual(617, na_max)

    def test_safe_none(self):
        c_ub = gm.c_A * 611
        na_max = rq4.compute_na_max(c_ub, 611, None, 900)
        self.assertEqual(na_max, 613)

    def test_env_too_low_raises(self):
        with self.assertRaises(SystemExit):
            rq4.compute_na_max(gm.c_A * 611, 611, 617, 600)


class TestAnchors(unittest.TestCase):

    def test_anchors_for(self):
        self.assertEqual(rq4.anchors_for(611, 617), [(611, 0), (617, 0)])
        self.assertEqual(rq4.anchors_for(611, None), [(611, 0)])

    def test_seed_from_baseline(self):
        # 手造首次导通样本：50 个 trial 在 n=10 导通、50 个 trial 未导通
        store = rq4.EvalStore(1, 60, 80, 7)
        try:
            contacts = [10] * 50 + [None] * 50
            rq4.seed_from_baseline(store, contacts, 100, 50, None)
            self.assertEqual(store.get((50, 0)), (50, 100))
            # ensure 从 m_base 续跑补差：不重复、样本量累计
            store.ensure([(50, 0)], 120)
            self.assertEqual(store.get((50, 0)), (50, 120))
        finally:
            store.close()


class TestEscalationLadder(unittest.TestCase):
    """escalate：成本升序、cost_cut 剪枝、K 传播（stub 纯逻辑，无蒙特卡洛）。"""

    def setUp(self):
        self.store = _StubStore()
        self.pts = [(i, 0) for i in range(1, 71)]   # 成本递增，跨两批

    def test_all_escalated_when_no_reliable(self):
        def fake(x, m):
            return ('insufficient', 0.8, 0.89)
        orig = rq4.wilson_verdict
        rq4.wilson_verdict = fake
        try:
            stage = {p: ('crossing', 0.85, 0.95) for p in self.pts}
            _, K = rq4.escalate(self.store, self.pts, stage, 600)
        finally:
            rq4.wilson_verdict = orig
        self.assertIsNone(K)
        for p in self.pts:
            self.assertEqual(self.store.get(p)[1], 600)
            self.assertEqual(stage[p][0], 'insufficient')

    def test_prune_after_reliable(self):
        # 批1 首个点可靠 → 批1 剩余与尾批全部剪枝（成本 ≥ K 不追加）
        calls = {'n': 0}

        def fake(x, m):
            calls['n'] += 1
            return ('reliable', 0.91, 0.95) if calls['n'] == 1 \
                else ('insufficient', 0.8, 0.89)
        orig = rq4.wilson_verdict
        rq4.wilson_verdict = fake
        try:
            stage = {p: ('crossing', 0.85, 0.95) for p in self.pts}
            _, K = rq4.escalate(self.store, self.pts, stage, 600)
        finally:
            rq4.wilson_verdict = orig
        self.assertEqual(K, gm.c_A)
        self.assertEqual(stage[(1, 0)][0], 'reliable')
        # 尾批 6 点被剪：仍 crossing、样本未追加
        for p in [(65, 0), (66, 0), (70, 0)]:
            self.assertEqual(stage[p][0], 'crossing')
            self.assertEqual(self.store.get(p)[1], 0)

    def test_cost_cut_prescreen(self):
        def fake(x, m):
            return ('insufficient', 0.8, 0.89)
        orig = rq4.wilson_verdict
        rq4.wilson_verdict = fake
        try:
            stage = {p: ('crossing', 0.85, 0.95) for p in self.pts}
            _, K = rq4.escalate(self.store, self.pts, stage, 600,
                                cost_cut=gm.c_A * 3)
        finally:
            rq4.wilson_verdict = orig
        # 预剪：cost ≥ 3c_A 的点不追加
        self.assertEqual(self.store.get((3, 0))[1], 0)
        self.assertEqual(self.store.get((2, 0))[1], 600)


class TestVerifyBand(unittest.TestCase):
    """verify_band：K ≤ c_layer 早停（单调引理）；锚点兜底。"""

    def test_stops_when_K_below_layer(self):
        # 首层（c_min = 3c_A）内出现可靠点 → 不再评估更高层
        store = _StubStore()
        anchors = [(20, 0), (25, 0)]

        def fake(x, m):
            return ('reliable', 0.91, 0.95)
        orig = rq4.wilson_verdict
        rq4.wilson_verdict = fake
        try:
            stage, v_pts, K = rq4.verify_band(
                store, c_min=3 * gm.c_A, c_ub_eff=gm.c_A * 30,
                na_cap=30, m0=20, m_final=40, anchor_m=80,
                anchors=anchors, K_init=None)
        finally:
            rq4.wilson_verdict = orig
        self.assertIsNotNone(K)
        # 首层 band 全部 reliable → K = 首层最便宜点成本
        band = rq4.band_points(3 * gm.c_A, na_cap=30)
        self.assertEqual(K, min(rq4.point_cost(p) for p in band))
        # 早停：只评估了首层点，锚点未触发兜底
        self.assertNotIn((20, 0), stage)
        self.assertEqual(store.get((20, 0))[1], 0)

    def test_anchor_fallback_when_never_reliable(self):
        store = _StubStore()
        anchors = [(20, 0), (25, 0)]

        def fake(x, m):
            return ('insufficient', 0.8, 0.89)
        orig = rq4.wilson_verdict
        rq4.wilson_verdict = fake
        try:
            stage, v_pts, K = rq4.verify_band(
                store, c_min=3 * gm.c_A, c_ub_eff=gm.c_A * 30,
                na_cap=30, m0=20, m_final=40, anchor_m=80,
                anchors=anchors, K_init=None)
        finally:
            rq4.wilson_verdict = orig
        self.assertIsNone(K)
        # 全不足 → 锚点兜底先 m_final=40，仍不足再 anchor_m=80
        self.assertEqual(store.get((20, 0))[1], 80)
        self.assertIn((20, 0), stage)


if __name__ == '__main__':
    unittest.main(verbosity=2)
