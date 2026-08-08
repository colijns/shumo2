# 本程序及代码是在AI工具辅助下完成的
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31

"""问题3 核心逻辑测试：首次导通数量、单调性、口径对照、跨壁理论值。"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import first_passage as fp  # noqa: E402


class TestFirstContact:
    def test_no_cylinder_not_conductive(self):
        c = np.empty((0, 3))
        u = np.empty((0, 3))
        h = np.empty(0)
        res = fp.first_contact_n(c, u, h)
        assert res['n_contact'] is None
        assert not res['conductive']

    def test_single_cross_wall_h1_immediate(self):
        """敏感性口径（假设一）：单根跨壁圆柱立即导通 N_c=1。"""
        ends = np.array([[[-6000.0, 0.0, 0.0], [-1000.0, 0.0, 0.0]]])
        from core import build_cylinders
        c, u, h = build_cylinders(ends)
        res = fp.first_contact_n(c, u, h, same_source=True)
        assert res['n_contact'] == 1
        assert res['conductive']

    def test_single_cross_wall_h2_not_conductive(self):
        """假设二：同源片段不自动电连续，单根跨壁圆柱不导通。"""
        ends = np.array([[[-6000.0, 0.0, 0.0], [-1000.0, 0.0, 0.0]]])
        from core import build_cylinders
        c, u, h = build_cylinders(ends)
        res = fp.first_contact_n(c, u, h, same_source=False)
        assert res['n_contact'] is None
        assert not res['conductive']

    def test_monotone_nested_samples(self):
        """嵌套共同样本单调性：Y_s(N) 单调不减（增量早停正确性）。"""
        rng = np.random.default_rng(7)
        c, u, h = fp.geo.generate_cylinders(60, rng)
        # 逐前缀检查导通状态单调
        prev = False
        for k in range(5, 61, 5):
            res = fp.first_contact_n(c[:k], u[:k], h[:k])
            cur = res['conductive']
            # 一旦导通，更高前缀必须仍导通
            assert cur or not prev, '导通状态不应回退'
            prev = prev or cur

    def test_n_contact_consistent_with_prefix(self):
        """N_c 与直接判前缀导通一致：k < N_c 不导通，k ≥ N_c 导通。"""
        rng = np.random.default_rng(11)
        c, u, h = fp.geo.generate_cylinders(400, rng)
        res = fp.first_contact_n(c, u, h)
        if res['n_contact'] is not None:
            nc = res['n_contact']
            below = fp.first_contact_n(c[:nc - 1], u[:nc - 1], h[:nc - 1])
            at = fp.first_contact_n(c[:nc], u[:nc], h[:nc])
            assert not below['conductive']
            assert at['conductive']


class TestCrossing:
    def test_crossing_ratio_near_theoretical(self):
        """跨壁比例 ≈ 60.8% 理论基准（10^4 根 MC，容差 ±1.5pp）。"""
        rng = np.random.default_rng(42)
        c, u, h = fp.geo.generate_cylinders(10_000, rng)
        n_cross, n_cyl = fp.crossing_count(c, u, h)
        ratio = n_cross / n_cyl
        assert 0.593 <= ratio <= 0.623, f'跨壁比例 {ratio:.4f} 偏离理论基准'

    def test_crossing_count_zero_for_interior(self):
        """中心原点、轴向 x 的圆柱不跨壁。"""
        c = np.zeros((1, 3))
        u = np.array([[1.0, 0.0, 0.0]])
        h = np.array([100.0])
        n_cross, n_cyl = fp.crossing_count(c, u, h)
        assert n_cross == 0 and n_cyl == 1


class TestEstimators:
    def test_n90_quantile_definition(self):
        """N̂90 = min{N: P̂(N) ≥ 0.90}，即经验 90% 分位数（higher）。"""
        import run_question3 as rq
        # P̂(10) = 1800/2000 = 0.90 ≥ 0.90 → N̂90 = 10（定义含等号）
        contacts = [10] * 1800 + [20] * 200
        n_hat, n_safe = rq.n_90_estimates(contacts, n_max=30, m=2000)
        assert n_hat == 10
        # P̂(10) = 1799/2000 = 0.8995 < 0.90 → N̂90 = 20
        contacts2 = [10] * 1799 + [20] * 201
        n_hat2, _ = rq.n_90_estimates(contacts2, n_max=30, m=2000)
        assert n_hat2 == 20

    def test_empirical_cdf_monotone(self):
        import run_question3 as rq
        rng = np.random.default_rng(5)
        contacts = list(rng.integers(1, 100, size=500))
        Ns, p_hat, lo, hi, x = rq.empirical_cdf(contacts, n_max=100, m=500)
        assert np.all(np.diff(p_hat) >= 0)
        assert np.all(lo <= p_hat) and np.all(p_hat <= hi)

    def test_wilson_bound_at_p90(self):
        """p̂=0.90、m=2000 时 Wilson 区间不与 0.90 线退化相撞。"""
        import run_question3 as rq
        from monte_carlo import wilson_ci
        p, lo, hi = wilson_ci(1800, 2000)
        assert lo < p < hi
        assert lo > 0.88  # 保证区间有分辨率
