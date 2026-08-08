"""问题2：轴线截断与完整实体裁剪的同样本对照试验。

对每个随机微结构同时计算：
1. 现有轴线截断模型；
2. 圆柱内接正多棱柱（真实圆柱导通事件的下界）；
3. 圆柱外切正多棱柱（真实圆柱导通事件的上界）。

内外多棱柱均先按周期边界平移，再与基本盒精确裁剪。二者使用完全相同的
随机样本，因此差异来自边界几何，而不是蒙特卡洛抽样波动。
"""

from concurrent.futures import ProcessPoolExecutor
import json
import os
import sys
import time

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import geometry as axis_geometry  # noqa: E402
import monte_carlo as mc  # noqa: E402
import solid_geometry as solid_geometry  # noqa: E402


PHIS = tuple(float(value) for value in os.environ.get(
    'SHUMO_Q2_SOLID_PHIS', '0.005,0.006,0.007,0.01').split(','))
TRIALS = int(os.environ.get('SHUMO_Q2_SOLID_TRIALS', '20'))
N_SIDES = int(os.environ.get('SHUMO_Q2_SOLID_SIDES', '64'))
WORKERS = int(os.environ.get(
    'SHUMO_Q2_SOLID_WORKERS', str(min(8, os.cpu_count() or 1))))
BASE_SEED = int(os.environ.get('SHUMO_Q2_SOLID_SEED', '20260808'))
OUTPUT = os.path.join(HERE, 'results', 'solid_boundary_comparison.json')


def _one_sample(args):
    phi_index, trial_index, phi = args
    seed = BASE_SEED + phi_index * 100_000 + trial_index
    rng = np.random.default_rng(seed)
    n_cylinders = mc.n_cylinders(phi)
    c, u, h = axis_geometry.generate_cylinders(n_cylinders, rng)

    t0 = time.perf_counter()
    axis = axis_geometry.sample_conductive(c, u, h)
    axis_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    inner = solid_geometry.sample_conductive_solid(
        c, u, h, n_sides=N_SIDES, mode='inscribed')
    inner_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    outer = solid_geometry.sample_conductive_solid(
        c, u, h, n_sides=N_SIDES, mode='circumscribed')
    outer_s = time.perf_counter() - t0

    return {
        'phi_index': phi_index,
        'trial_index': trial_index,
        'seed': seed,
        'n_A': n_cylinders,
        'axis': int(axis['conductive']),
        'inner': int(inner['conductive']),
        'outer': int(outer['conductive']),
        'bracket_violation': int(inner['conductive'] and not outer['conductive']),
        'axis_s': axis_s,
        'inner_s': inner_s,
        'outer_s': outer_s,
        'axis_crossing': axis['n_crossing'],
        'inner_crossing': inner['n_crossing'],
        'outer_crossing': outer['n_crossing'],
        'inner_fragments': inner['n_fragments'],
        'outer_fragments': outer['n_fragments'],
        'inner_gjk': inner['n_gjk'],
        'outer_gjk': outer['n_gjk'],
    }


def _summary(phi, samples):
    m = len(samples)
    row = {'phi': phi, 'n_A': samples[0]['n_A'], 'm': m}
    for key in ('axis', 'inner', 'outer'):
        x = sum(sample[key] for sample in samples)
        p_hat, lower, upper = mc.wilson_ci(x, m)
        row[key] = {
            'x': x,
            'p_hat': p_hat,
            'ci_lower': lower,
            'ci_upper': upper,
        }
    for key in ('axis_s', 'inner_s', 'outer_s', 'axis_crossing',
                'inner_crossing', 'outer_crossing', 'inner_fragments',
                'outer_fragments', 'inner_gjk', 'outer_gjk'):
        row['mean_' + key] = sum(sample[key] for sample in samples) / m
    row['bracket_violations'] = sum(
        sample['bracket_violation'] for sample in samples)
    row['axis_vs_inner_disagreements'] = sum(
        sample['axis'] != sample['inner'] for sample in samples)
    row['axis_vs_outer_disagreements'] = sum(
        sample['axis'] != sample['outer'] for sample in samples)
    row['inner_outer_uncertain_samples'] = sum(
        sample['inner'] != sample['outer'] for sample in samples)
    return row


def main():
    if TRIALS <= 0 or N_SIDES < 8 or WORKERS <= 0:
        raise ValueError('试验数、棱柱边数和进程数必须为正，且棱柱边数至少为 8')
    jobs = [(phi_index, trial_index, phi)
            for phi_index, phi in enumerate(PHIS)
            for trial_index in range(TRIALS)]
    started = time.perf_counter()
    grouped = [[] for _ in PHIS]
    with ProcessPoolExecutor(max_workers=min(WORKERS, len(jobs))) as executor:
        for sample in executor.map(_one_sample, jobs):
            grouped[sample['phi_index']].append(sample)

    rows = []
    for phi_index, phi in enumerate(PHIS):
        samples = sorted(grouped[phi_index], key=lambda item: item['trial_index'])
        row = _summary(phi, samples)
        rows.append(row)
        print(
            f'phi={100 * phi:.2f}% N={row["n_A"]} M={row["m"]}: '
            f'axis={row["axis"]["x"]}, inner={row["inner"]["x"]}, '
            f'outer={row["outer"]["x"]}, '
            f'inner!=outer={row["inner_outer_uncertain_samples"]}, '
            f'violations={row["bracket_violations"]}', flush=True)

    payload = {
        'experiment': 'Q2 solid-cylinder boundary comparison',
        'interpretation': {
            'axis': '原轴线截断近似，仅作历史基线，不属于实体上下界',
            'inner': '内接正多棱柱，导通概率下界',
            'outer': '外切正多棱柱，导通概率上界',
            'same_random_samples': True,
            'same_source_fragments_auto_connected': False,
        },
        'config': {
            'phis': PHIS,
            'trials_per_phi': TRIALS,
            'n_sides': N_SIDES,
            'radial_error_bound_nm': solid_geometry.radial_error_bound(
                axis_geometry.R, N_SIDES),
            'workers': min(WORKERS, len(jobs)),
            'base_seed': BASE_SEED,
            'seed_formula': 'base_seed + 100000*phi_index + trial_index',
        },
        'rows': rows,
        'wall_elapsed_s': time.perf_counter() - started,
    }
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f'written: {OUTPUT}')


if __name__ == '__main__':
    main()
