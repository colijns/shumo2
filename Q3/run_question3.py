# 本程序及代码是在AI工具辅助下完成的
"""问题3：用完整实体首次导通分布估计90%最低填充量。

核心输出不是单个体积分数上的独立概率，而是每次共同随机样本的首次导通
根数。外切多棱柱给出真实临界根数下界，内接多棱柱给出上界；两者随正多
边形边数增加而收敛。
"""

from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal, ROUND_HALF_UP
import csv
import json
import math
import os
import sys
import time

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
Q2_DIR = os.path.join(ROOT, 'Q2')
for path in (HERE, Q2_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import monte_carlo as mc  # noqa: E402
import solid_first_passage as first_passage  # noqa: E402
import solid_geometry  # noqa: E402


TARGET = float(os.environ.get('SHUMO_Q3_TARGET', '0.90'))
N_MAX = int(os.environ.get('SHUMO_Q3_NMAX', '900'))
TRIALS = int(os.environ.get('SHUMO_Q3_TRIALS', '20'))
N_SIDES = int(os.environ.get('SHUMO_Q3_SIDES', '64'))
WORKERS = int(os.environ.get(
    'SHUMO_Q3_WORKERS', str(min(8, os.cpu_count() or 1))))
BATCH_SIZE = int(os.environ.get('SHUMO_Q3_BATCH_SIZE', '1'))
BASE_SEED = int(os.environ.get('SHUMO_Q3_BASE_SEED', '20260808'))
BOOTSTRAP_REPEATS = int(os.environ.get('SHUMO_Q3_BOOTSTRAP', '2000'))

RESULT_DIR = os.path.join(HERE, 'results')
SUMMARY_JSON = os.path.join(RESULT_DIR, 'question3_solid_summary.json')
CURVE_CSV = os.path.join(RESULT_DIR, 'question3_solid_curve.csv')


def empirical_cdf(contacts, n_max, target=TARGET):
    """首次导通根数样本转为单调经验CDF及逐点Wilson区间。"""
    m = len(contacts)
    if m == 0:
        raise ValueError('首次导通样本不能为空')
    values = np.asarray(
        [value if value is not None else n_max + 1 for value in contacts],
        dtype=int,
    )
    if np.any(values < 1) or np.any(values > n_max + 1):
        raise ValueError('首次导通根数超出允许范围')
    histogram = np.bincount(values, minlength=n_max + 2)
    x = np.cumsum(histogram)[1:n_max + 1]
    p_hat = x / m
    lower = np.empty(n_max, dtype=float)
    upper = np.empty(n_max, dtype=float)
    for index, successes in enumerate(x):
        _, lower[index], upper[index] = mc.wilson_ci(int(successes), m)
    indices = np.arange(1, n_max + 1, dtype=int)
    point_hit = np.nonzero(p_hat >= target)[0]
    safe_hit = np.nonzero(lower >= target)[0]
    return {
        'n': indices,
        'x': x,
        'p_hat': p_hat,
        'ci_lower': lower,
        'ci_upper': upper,
        'n_hat': int(indices[point_hit[0]]) if len(point_hit) else None,
        'n_safe': int(indices[safe_hit[0]]) if len(safe_hit) else None,
    }


def bootstrap_n90(contacts, n_max, repeats=BOOTSTRAP_REPEATS,
                  target=TARGET, seed=99173):
    """首次导通根数90%分位数的可复现Bootstrap区间。"""
    if repeats <= 0:
        return None
    values = np.asarray(
        [value if value is not None else n_max + 1 for value in contacts],
        dtype=int,
    )
    rng = np.random.default_rng(seed)
    quantiles = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sample = values[rng.integers(0, len(values), size=len(values))]
        quantiles[index] = np.quantile(sample, target, method='higher')
    return {
        'n_lower': float(np.percentile(quantiles, 2.5)),
        'n_upper': float(np.percentile(quantiles, 97.5)),
    }


def phi_percent(n_cylinders):
    if n_cylinders is None:
        return None
    return float(n_cylinders) * mc.geo.V_A / mc.geo.V_BOX * 100.0


def round_half_up_2(value):
    if value is None:
        return None
    return float(Decimal(str(value)).quantize(
        Decimal('0.01'), rounding=ROUND_HALF_UP))


def ceil_2(value):
    if value is None:
        return None
    return math.ceil(value * 100.0 - 1e-12) / 100.0


def threshold_summary(curve, contacts, n_max, bootstrap_seed):
    n_hat = curve['n_hat']
    n_safe = curve['n_safe']

    def at(n_value):
        if n_value is None:
            return None
        index = n_value - 1
        return {
            'n': n_value,
            'x': int(curve['x'][index]),
            'p_hat': float(curve['p_hat'][index]),
            'ci_lower': float(curve['ci_lower'][index]),
            'ci_upper': float(curve['ci_upper'][index]),
            'phi_percent_raw': phi_percent(n_value),
            'phi_percent_rounded': round_half_up_2(phi_percent(n_value)),
        }

    bootstrap = bootstrap_n90(
        contacts, n_max, seed=bootstrap_seed)
    if bootstrap is not None:
        bootstrap['phi_lower_percent'] = phi_percent(bootstrap['n_lower'])
        bootstrap['phi_upper_percent'] = phi_percent(bootstrap['n_upper'])
        bootstrap['width_percentage_point'] = (
            bootstrap['phi_upper_percent'] - bootstrap['phi_lower_percent'])
    return {
        'point': at(n_hat),
        'safe': at(n_safe),
        'bootstrap': bootstrap,
        'n_never': sum(value is None for value in contacts),
    }


def _batch_worker(args):
    n_max, trial_indices, base_seed, n_sides = args
    outer_contacts = []
    inner_contacts = []
    stats = {
        'm': 0,
        'bracket_violations': 0,
        'uncertain_samples': 0,
        'outer_gjk': 0,
        'inner_gjk': 0,
        'outer_crossing': 0,
        'inner_crossing': 0,
    }
    for trial_index in trial_indices:
        seed = base_seed + int(trial_index)
        outer, inner, one = first_passage.simulate_paired_trials(
            n_max, m=1, seed=seed, n_sides=n_sides)
        outer_contacts.extend(outer)
        inner_contacts.extend(inner)
        stats['m'] += 1
        for key in stats:
            if key != 'm':
                stats[key] += one[key]
    return outer_contacts, inner_contacts, stats


def run_trials(n_max=N_MAX, trials=TRIALS, workers=WORKERS,
               batch_size=BATCH_SIZE, base_seed=BASE_SEED,
               n_sides=N_SIDES):
    """并行运行实体上下界首次导通试验。"""
    if trials <= 0 or workers <= 0 or batch_size <= 0:
        raise ValueError('试验数、进程数和批大小必须为正整数')
    chunks = [list(range(start, min(start + batch_size, trials)))
              for start in range(0, trials, batch_size)]
    jobs = [(n_max, chunk, base_seed, n_sides) for chunk in chunks]
    outer_contacts = []
    inner_contacts = []
    totals = {
        'm': 0,
        'bracket_violations': 0,
        'uncertain_samples': 0,
        'outer_gjk': 0,
        'inner_gjk': 0,
        'outer_crossing': 0,
        'inner_crossing': 0,
    }
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        for outer, inner, stats in executor.map(_batch_worker, jobs):
            outer_contacts.extend(outer)
            inner_contacts.extend(inner)
            for key in totals:
                totals[key] += stats[key]
    return outer_contacts, inner_contacts, totals


def _curve_rows(outer_curve, inner_curve):
    for index, n_value in enumerate(outer_curve['n']):
        yield {
            'n': int(n_value),
            'phi_percent': phi_percent(n_value),
            'outer_p_hat': float(outer_curve['p_hat'][index]),
            'outer_ci_lower': float(outer_curve['ci_lower'][index]),
            'outer_ci_upper': float(outer_curve['ci_upper'][index]),
            'inner_p_hat': float(inner_curve['p_hat'][index]),
            'inner_ci_lower': float(inner_curve['ci_lower'][index]),
            'inner_ci_upper': float(inner_curve['ci_upper'][index]),
        }


def main():
    if not (0.0 < TARGET < 1.0):
        raise ValueError('目标概率必须位于0和1之间')
    if N_MAX <= 0 or N_SIDES < 8:
        raise ValueError('N_MAX必须为正，横截面边数至少为8')
    print(
        f'Q3实体首次导通: N_max={N_MAX}, M={TRIALS}, sides={N_SIDES}, '
        f'workers={min(WORKERS, TRIALS)}, target={TARGET:.2f}',
        flush=True,
    )
    started = time.perf_counter()
    outer_contacts, inner_contacts, stats = run_trials()
    outer_curve = empirical_cdf(outer_contacts, N_MAX)
    inner_curve = empirical_cdf(inner_contacts, N_MAX)
    outer_summary = threshold_summary(
        outer_curve, outer_contacts, N_MAX, bootstrap_seed=99173)
    inner_summary = threshold_summary(
        inner_curve, inner_contacts, N_MAX, bootstrap_seed=99174)

    outer_n = outer_curve['n_hat']
    inner_n = inner_curve['n_hat']
    bracket_closed = (
        outer_n is not None and inner_n is not None and outer_n <= inner_n)
    empirical_n_interval = [outer_n, inner_n] if bracket_closed else None
    empirical_phi_interval = (
        [phi_percent(outer_n), phi_percent(inner_n)]
        if bracket_closed else None)
    same_reported_value = (
        bracket_closed
        and round_half_up_2(empirical_phi_interval[0])
        == round_half_up_2(empirical_phi_interval[1]))
    bootstrap_widths = [
        summary_item['bootstrap']['width_percentage_point']
        for summary_item in (outer_summary, inner_summary)
        if summary_item['bootstrap'] is not None
    ]
    bootstrap_precision_met = (
        len(bootstrap_widths) == 2 and max(bootstrap_widths) <= 0.01)
    conservative_safe_n = inner_curve['n_safe']

    summary = {
        'problem': 'A题问题3',
        'method': '完整实体内接/外切多棱柱首次导通上下界',
        'config': {
            'target_probability': TARGET,
            'n_max': N_MAX,
            'trials': TRIALS,
            'n_sides': N_SIDES,
            'radial_error_bound_nm': solid_geometry.radial_error_bound(
                mc.geo.R, N_SIDES),
            'workers': min(WORKERS, TRIALS),
            'batch_size': BATCH_SIZE,
            'base_seed': BASE_SEED,
            'trial_seed_formula': 'base_seed + trial_index',
            'same_source_auto_connection': False,
        },
        'outer_lower_threshold': outer_summary,
        'inner_upper_threshold': inner_summary,
        'empirical_geometry_bracket_n': empirical_n_interval,
        'empirical_geometry_bracket_phi_percent': empirical_phi_interval,
        'same_value_after_two_decimal_rounding': same_reported_value,
        'conservative_point_n': inner_n,
        'conservative_point_phi_percent_raw': phi_percent(inner_n),
        'conservative_95_n': conservative_safe_n,
        'conservative_95_phi_percent_raw': phi_percent(conservative_safe_n),
        'conservative_95_phi_percent_ceiling_2': ceil_2(
            phi_percent(conservative_safe_n)),
        'bootstrap_precision_target_percentage_point': 0.01,
        'bootstrap_precision_met': bootstrap_precision_met,
        'diagnostics': stats,
        'wall_elapsed_s': time.perf_counter() - started,
        'formal_result_ready': bool(
            bracket_closed and stats['bracket_violations'] == 0
            and conservative_safe_n is not None
            and bootstrap_precision_met),
    }

    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(SUMMARY_JSON, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    fields = [
        'n', 'phi_percent', 'outer_p_hat', 'outer_ci_lower',
        'outer_ci_upper', 'inner_p_hat', 'inner_ci_lower', 'inner_ci_upper',
    ]
    with open(CURVE_CSV, 'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_curve_rows(outer_curve, inner_curve))

    print('\n===== Q3实体上下界阶段结果 =====')
    print(f'本轮外切经验临界（几何下界）: {outer_n}')
    print(f'本轮内接经验临界（几何上界）: {inner_n}')
    if empirical_phi_interval is not None:
        print('本轮经验临界体积分数的几何夹逼: '
              f'[{empirical_phi_interval[0]:.6f}%, '
              f'{empirical_phi_interval[1]:.6f}%]')
    print(f'上下界违例: {stats["bracket_violations"]}, '
          f'样本级上下界有间隙: {stats["uncertain_samples"]}/{TRIALS}')
    print(f'Bootstrap精度达标: {bootstrap_precision_met}, '
          f'正式结果就绪: {summary["formal_result_ready"]}')
    print(f'结果: {SUMMARY_JSON}')
    print(f'曲线: {CURVE_CSV}')


if __name__ == '__main__':
    main()
