# 本程序及代码是在AI工具辅助下完成的
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31

import importlib.util
import os
import time

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    'q4_geometry', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'geometry.py'))
geo = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(geo)


Z_WILSON = 1.959964


def wilson_ci(x, m, z=Z_WILSON):
    if m <= 0:
        raise ValueError('样本数必须为正')
    p = x / m
    denom = 1.0 + z * z / m
    center = p + z * z / (2.0 * m)
    half = z * np.sqrt(p * (1.0 - p) / m + z * z / (4.0 * m * m))
    return p, max(0.0, (center - half) / denom), min(1.0, (center + half) / denom)


def total_cost(n_a, n_b):
    return int(n_a) * geo.COST_PER_A + int(n_b) * geo.COST_PER_B


def volume_fractions(n_a, n_b):
    return (int(n_a) * geo.V_A / geo.V_BOX,
            int(n_b) * geo.V_B / geo.V_BOX)


def simulate_pair(n_a, n_b, m=20, seed=20260808):
    n_a, n_b, m = int(n_a), int(n_b), int(m)
    rng = np.random.default_rng(seed)
    x = 0
    totals = {key: 0.0 for key in (
        'n_fragments', 'n_A_fragments', 'n_B_fragments',
        'n_A_crossing', 'n_B_crossing', 'n_edges', 'n_candidates',
        'n_exact')}
    t0 = time.perf_counter()
    for _ in range(m):
        result = geo.generate_sample(n_a, n_b, rng)
        x += int(result['conductive'])
        for key in totals:
            totals[key] += result[key]
    elapsed = time.perf_counter() - t0
    p, lo, hi = wilson_ci(x, m)
    phi_a, phi_b = volume_fractions(n_a, n_b)
    return {
        'n_A': n_a, 'n_B': n_b, 'm': m, 'x': x,
        'p_hat': p, 'ci_lower': lo, 'ci_upper': hi,
        'phi_A': phi_a, 'phi_B': phi_b,
        'cost': total_cost(n_a, n_b),
        **{f'mean_{key}': value / m for key, value in totals.items()},
        'elapsed_s': elapsed,
    }


def common_random_trial(pairs, seed):
    pairs = [(int(n_a), int(n_b)) for n_a, n_b in pairs]
    if not pairs:
        return []
    if any(n_a < 0 or n_b < 0 for n_a, n_b in pairs):
        raise ValueError('介质数量不能为负数')
    rng = np.random.default_rng(seed)
    max_a = max(n_a for n_a, _ in pairs)
    max_b = max(n_b for _, n_b in pairs)
    c_a, u_a, h_a = geo.q2geo.generate_cylinders(max_a, rng)
    c_b = geo.generate_spheres(max_b, rng)
    return [geo.sample_conductive(
        c_a[:n_a], u_a[:n_a], h_a[:n_a], c_b[:n_b])
            for n_a, n_b in pairs]


def aggregate_common_trials(pairs, trial_results):
    pairs = [(int(n_a), int(n_b)) for n_a, n_b in pairs]
    m = len(trial_results)
    if m <= 0:
        raise ValueError('至少需要一次共同随机数试验')
    keys = ('n_fragments', 'n_A_fragments', 'n_B_fragments',
            'n_A_crossing', 'n_B_crossing', 'n_edges', 'n_candidates',
            'n_exact')
    rows = []
    for index, (n_a, n_b) in enumerate(pairs):
        values = [trial[index] for trial in trial_results]
        x = sum(int(value['conductive']) for value in values)
        p, lo, hi = wilson_ci(x, m)
        phi_a, phi_b = volume_fractions(n_a, n_b)
        rows.append({
            'n_A': n_a, 'n_B': n_b, 'm': m, 'x': x,
            'p_hat': p, 'ci_lower': lo, 'ci_upper': hi,
            'phi_A': phi_a, 'phi_B': phi_b,
            'cost': total_cost(n_a, n_b),
            **{f'mean_{key}': sum(value[key] for value in values) / m
               for key in keys},
        })
    return rows


def equal_cost_frontier(reference_n_a, a_counts):
    return cost_frontier(total_cost(reference_n_a, 0), a_counts)


def cost_frontier(budget, a_counts):
    budget = float(budget)
    if budget < 0.0:
        raise ValueError('成本上限不能为负数')
    rows = []
    for n_a in sorted(set(int(v) for v in a_counts)):
        if n_a < 0 or total_cost(n_a, 0) > budget:
            continue
        remaining = budget - total_cost(n_a, 0)
        n_b = max(0, int(np.floor(remaining / geo.COST_PER_B + 1e-12)))
        rows.append((n_a, n_b))
    return rows
