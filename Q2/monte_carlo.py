# 本程序及代码是在AI工具辅助下完成的
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31

"""问题2 蒙特卡洛：给定体积分数 φ 估计微构体导通概率（95% Wilson CI）。

- N_A = round(φ·V_BOX/V_A)，体积分数按完整圆柱计量（截断不改变计量体积）；
- 单一 rng + 固定生成顺序，seed 可复现；
- 每样本返回导通标志与片段统计，聚合后输出结果 dict。
"""

import time

import numpy as np

import geometry as geo

BASE_SEED = 42          # 批次入口派生：BASE_SEED + 100000*phi_index + batch_index
Z_WILSON = 1.959964     # 95% 正态分位数
THEORETICAL_CROSSING_RATE = 1.0 - (0.25 + 15.0 / (32.0 * np.pi))


def n_cylinders(phi):
    """体积分数 φ → 圆柱数量（四舍五入）。"""
    value = float(phi) * geo.V_BOX / geo.V_A
    if value < 0.0:
        raise ValueError('体积分数不能为负数')
    # Python round 使用“银行家舍入”；题目要求普通四舍五入。
    return int(np.floor(value + 0.5))


def wilson_ci(x, m, z=Z_WILSON):
    """Wilson 95% 置信区间。p̂=0/1 不退化（区间非对称）。

    返回 (p_hat, ci_lower, ci_upper)。
    """
    p_hat = x / m
    denom = 1.0 + z * z / m
    center = p_hat + z * z / (2.0 * m)
    half = z * np.sqrt(p_hat * (1.0 - p_hat) / m + z * z / (4.0 * m * m))
    lo = (center - half) / denom
    hi = (center + half) / denom
    # 概率界 clip（p̂=0/1 时公式两端浮点可略出界）
    return p_hat, max(0.0, lo), min(1.0, hi)


def simulate_phi(phi, m=2000, seed=BASE_SEED):
    """对单一体积分数跑 M 次生成并估计导通概率。

    返回 dict: {phi, n_A, m, x, p_hat, ci_lower, ci_upper,
                mean_fragments, mean_edges, mean_gjk, elapsed_s}。
    """
    n_A = n_cylinders(phi)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    x = 0
    frag_sum = crossing_sum = edges_sum = gjk_sum = 0
    for _ in range(m):
        c, u, h = geo.generate_cylinders(n_A, rng)
        res = geo.sample_conductive(c, u, h)
        x += int(res['conductive'])
        frag_sum += res['n_fragments']
        crossing_sum += res['n_crossing']
        edges_sum += res['n_edges']
        gjk_sum += res['n_gjk']
    elapsed = time.perf_counter() - t0
    p_hat, lo, hi = wilson_ci(x, m)
    return {
        'phi': phi,
        'n_A': n_A,
        'm': m,
        'x': x,
        'p_hat': p_hat,
        'ci_lower': lo,
        'ci_upper': hi,
        'mean_fragments': frag_sum / m,
        'mean_crossing': crossing_sum / m,
        'mean_crossing_rate': crossing_sum / (m * n_A) if n_A else 0.0,
        'mean_edges': edges_sum / m,
        'mean_gjk': gjk_sum / m,
        'elapsed_s': elapsed,
    }


if __name__ == '__main__':
    # 冒烟：φ=1.00%，M=50
    import json
    res = simulate_phi(0.01, m=50, seed=42)
    print(json.dumps(res, indent=2, ensure_ascii=False))
