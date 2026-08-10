# 本程序及代码是在AI工具辅助下完成的
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31


import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (os.path.join(ROOT, 'Q3'), os.path.join(ROOT, 'Q2')):
    if path not in sys.path:
        sys.path.insert(0, path)

import geometry as axis_geometry
import solid_first_passage as solid_fp
from monte_carlo import wilson_ci


TARGET = 0.90
DEFAULT_SEED = 20260808
RESULT_DIR = os.path.join(HERE, 'results')
CHECKPOINT = os.path.join(RESULT_DIR, 'question4_unified_a_checkpoint.json')
SUMMARY = os.path.join(RESULT_DIR, 'question4_unified_a_summary.json')
CURVE = os.path.join(RESULT_DIR, 'question4_unified_a_curve.csv')


def empirical_curve(contacts, n_max):
    m = len(contacts)
    if m == 0:
        raise ValueError('contacts不能为空')
    values = np.asarray(
        [value if value is not None else n_max + 1 for value in contacts],
        dtype=int,
    )
    hist = np.bincount(values, minlength=n_max + 2)
    x = np.cumsum(hist)[1:n_max + 1]
    p = x / m
    lo = np.empty(n_max, dtype=float)
    hi = np.empty(n_max, dtype=float)
    for i, successes in enumerate(x):
        _, lo[i], hi[i] = wilson_ci(int(successes), m)
    return {'n': np.arange(1, n_max + 1), 'x': x,
            'p_hat': p, 'ci_lower': lo, 'ci_upper': hi}


def first_at(values, predicate):
    hit = np.nonzero(predicate(values))[0]
    return int(hit[0] + 1) if len(hit) else None


def classify_at(n, outer_curve, inner_curve, target=TARGET):
    if n < 1 or n > len(outer_curve['n']):
        raise ValueError('n超出曲线范围')
    i = n - 1
    if outer_curve['ci_upper'][i] < target:
        verdict = 'reliably_insufficient'
    elif inner_curve['ci_lower'][i] >= target:
        verdict = 'reliably_feasible'
    else:
        verdict = 'undetermined'
    return {
        'N_A': n,
        'verdict': verdict,
        'outer_p_hat': float(outer_curve['p_hat'][i]),
        'outer_ci_lower': float(outer_curve['ci_lower'][i]),
        'outer_ci_upper': float(outer_curve['ci_upper'][i]),
        'inner_p_hat': float(inner_curve['p_hat'][i]),
        'inner_ci_lower': float(inner_curve['ci_lower'][i]),
        'inner_ci_upper': float(inner_curve['ci_upper'][i]),
    }


def _one_trial(args):
    trial_index, n_max, base_seed, n_sides = args
    rng = np.random.default_rng(base_seed + trial_index)
    c, u, h = axis_geometry.generate_cylinders(n_max, rng)
    result = solid_fp.paired_first_contacts(c, u, h, n_sides=n_sides)
    return {
        'trial_index': trial_index,
        'outer': result['outer']['n_contact'],
        'inner': result['inner']['n_contact'],
        'bracket_valid': result['bracket_valid'],
        'uncertain_width_n': result['uncertain_width_n'],
    }


def _load_checkpoint(config, resume):
    if not resume or not os.path.exists(CHECKPOINT):
        return {}
    with open(CHECKPOINT, encoding='utf-8') as handle:
        saved = json.load(handle)
    saved_config = saved.get('config', {})
    if any(saved_config.get(key) != value for key, value in config.items()):
        raise ValueError('检查点参数与本次运行不同；请取消--resume或恢复原参数')
    return {int(row['trial_index']): row for row in saved.get('trials', [])}


def _save_checkpoint(config, rows):
    payload = {'config': config,
               'trials': [rows[i] for i in sorted(rows)]}
    with open(CHECKPOINT, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def run_trials(n_max, trials, n_sides, base_seed, workers, resume=True):

    config = {'n_max': n_max, 'n_sides': n_sides,
              'base_seed': base_seed}
    rows = _load_checkpoint(config, resume)
    missing = [i for i in range(trials) if i not in rows]
    jobs = [(i, n_max, base_seed, n_sides) for i in missing]
    completed_since_save = 0
    if jobs:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            for row in executor.map(_one_trial, jobs, chunksize=1):
                rows[int(row['trial_index'])] = row
                completed_since_save += 1
                if completed_since_save >= max(10, workers * 2):
                    _save_checkpoint(config, rows)
                    completed_since_save = 0
        _save_checkpoint(config, rows)
    ordered = [rows[i] for i in range(trials)]
    return ([row['outer'] for row in ordered],
            [row['inner'] for row in ordered], ordered)


def phi_percent(n):
    return n * axis_geometry.V_A / axis_geometry.V_BOX * 100.0


def main():
    parser = argparse.ArgumentParser(description='Q4纯A退化情形严格统一复核')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args()

    n_max = int(os.environ.get('SHUMO_Q4_UNIFIED_NMAX', '750'))
    trials = int(os.environ.get('SHUMO_Q4_UNIFIED_TRIALS', '10000'))
    n_sides = int(os.environ.get('SHUMO_Q4_UNIFIED_SIDES', '32'))
    base_seed = int(os.environ.get('SHUMO_Q4_UNIFIED_SEED', str(DEFAULT_SEED)))
    workers = int(os.environ.get(
        'SHUMO_Q4_UNIFIED_WORKERS', str(min(8, os.cpu_count() or 1))))
    if args.smoke:
        trials = min(trials, 20)

    if min(n_max, trials, n_sides, workers) <= 0 or n_sides < 8:
        raise ValueError('N_max、M、workers必须为正，n_sides至少为8')
    os.makedirs(RESULT_DIR, exist_ok=True)
    started = time.perf_counter()
    print(f'Q4统一复核：N_max={n_max}, M={trials}, K={n_sides}, '
          f'workers={workers}, seed={base_seed}', flush=True)

    outer, inner, trial_rows = run_trials(
        n_max, trials, n_sides, base_seed, workers,
        resume=not args.no_resume)
    outer_curve = empirical_curve(outer, n_max)
    inner_curve = empirical_curve(inner, n_max)

    outer_hat = first_at(outer_curve['p_hat'], lambda x: x >= TARGET)
    inner_hat = first_at(inner_curve['p_hat'], lambda x: x >= TARGET)
    inner_safe = first_at(inner_curve['ci_lower'], lambda x: x >= TARGET)
    outer_fail = first_at(outer_curve['ci_upper'], lambda x: x >= TARGET)
    candidates = sorted(set(
        [n for n in range(max(1, (outer_hat or 617) - 5),
                          min(n_max, (inner_safe or 619) + 3) + 1)]
        + [611, 613, 616, 617, 618, 619]
    ))
    candidates = [n for n in candidates if n <= n_max]
    audit = [classify_at(n, outer_curve, inner_curve) for n in candidates]

    summary = {
        'method': 'Q3严格实体内接/外切夹逼，统一Q4的N_B=0退化情形',
        'config': {'n_max': n_max, 'trials': trials, 'n_sides': n_sides,
                   'base_seed': base_seed, 'workers': workers,
                   'same_source_auto_connection': False},
        'outer_point_threshold_lower_bound': outer_hat,
        'inner_point_threshold_upper_bound': inner_hat,
        'inner_wilson_safe_recommendation': inner_safe,
        'first_not_reliably_insufficient_outer': outer_fail,
        'recommended_N_A': inner_safe,
        'recommended_N_B': 0 if inner_safe is not None else None,
        'recommended_phi_A_percent': (phi_percent(inner_safe)
                                      if inner_safe is not None else None),
        'bracket_violations': sum(not row['bracket_valid']
                                  for row in trial_rows),
        'uncertain_geometry_samples': sum(row['uncertain_width_n'] > 0
                                          for row in trial_rows),
        'candidate_audit': audit,
        'scope_note': ('本结论严格统一纯A；含B候选仍需实体球冠混合几何。'
                       '在该工作完成前，不把含B近似点称为严格最优。'),
        'elapsed_s': time.perf_counter() - started,
    }
    with open(SUMMARY, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(CURVE, 'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['N_A', 'phi_A_percent',
                         'outer_x', 'outer_p_hat', 'outer_ci_lower', 'outer_ci_upper',
                         'inner_x', 'inner_p_hat', 'inner_ci_lower', 'inner_ci_upper',
                         'strict_verdict'])
        for n in range(1, n_max + 1):
            row = classify_at(n, outer_curve, inner_curve)
            i = n - 1
            writer.writerow([
                n, phi_percent(n), int(outer_curve['x'][i]),
                row['outer_p_hat'], row['outer_ci_lower'], row['outer_ci_upper'],
                int(inner_curve['x'][i]), row['inner_p_hat'],
                row['inner_ci_lower'], row['inner_ci_upper'], row['verdict'],
            ])

    print('\n===== 纯A统一复核 =====')
    print(f'经验临界夹逼：[{outer_hat}, {inner_hat}]')
    if inner_safe is None:
        print('严格保守推荐：当前样本量下尚未形成Wilson可靠可行点')
    else:
        print(f'严格保守推荐：N_A={inner_safe}, N_B=0')
    if inner_safe is not None:
        print(f'体积分数：{phi_percent(inner_safe):.6f}%')
    print(f'上下界违例：{summary["bracket_violations"]}')
    print(f'结果：{SUMMARY}')
    print(f'曲线：{CURVE}')


if __name__ == '__main__':
    main()
