# 本程序及代码是在AI工具辅助下完成的
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31



import time

import numpy as np

import geometry as geo

BASE_SEED = 42
Z_WILSON = 1.959964
THEORETICAL_CROSSING_RATE = 1.0 - (0.25 + 15.0 / (32.0 * np.pi))


def n_cylinders(phi):
    value = float(phi) * geo.V_BOX / geo.V_A
    if value < 0.0:
        raise ValueError('体积分数不能为负数')

    return int(np.floor(value + 0.5))


def wilson_ci(x, m, z=Z_WILSON):
    p_hat = x / m
    denom = 1.0 + z * z / m
    center = p_hat + z * z / (2.0 * m)
    half = z * np.sqrt(p_hat * (1.0 - p_hat) / m + z * z / (4.0 * m * m))
    lo = (center - half) / denom
    hi = (center + half) / denom

    return p_hat, max(0.0, lo), min(1.0, hi)


def simulate_phi(phi, m=2000, seed=BASE_SEED):
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

    import json
    res = simulate_phi(0.01, m=50, seed=42)
    print(json.dumps(res, indent=2, ensure_ascii=False))
