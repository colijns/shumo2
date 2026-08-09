"""Q4低成本临界候选的严格混合实体复核。

快速Q4程序负责定位候选；本程序对候选集合共享同一随机介质前缀，同时计算
内接和外切实体模型。严格判定规则：

* inner的Wilson下限 >= 0.90：可靠可行；
* outer的Wilson上限 < 0.90：可靠不足；
* 其他：待追加样本或提高几何分辨率。

默认候选包含原Q4搜索点、7个低成本跨线点、617附近纯A点以及统一推荐619。
程序按trial保存检查点，M可从小样本直接扩展到正式样本而不重复已完成试验。
"""

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
for path in (HERE, os.path.join(ROOT, 'Q2')):
    if path not in sys.path:
        sys.path.insert(0, path)

import geometry as axis_geometry  # noqa: E402
import geometry_mix as fast_geometry  # noqa: E402
import solid_mix_geometry as strict_geometry  # noqa: E402
from monte_carlo import wilson_ci  # noqa: E402


TARGET = 0.90
DEFAULT_CANDIDATES = (
    (598, 62),
    (609, 5), (608, 14), (610, 4), (611, 0), (610, 12),
    (616, 0), (617, 0), (617, 1), (618, 0), (619, 0),
)
RESULT_DIR = os.path.join(HERE, 'results')
CHECKPOINT = os.path.join(RESULT_DIR, 'question4_strict_mixed_checkpoint.json')
SUMMARY = os.path.join(RESULT_DIR, 'question4_strict_mixed_summary.json')
CSV_RESULT = os.path.join(RESULT_DIR, 'question4_strict_mixed_candidates.csv')


def strict_verdict(inner_x, outer_x, m, target=TARGET):
    """根据概率下界/上界作严格三分类。"""
    _, inner_lo, inner_hi = wilson_ci(int(inner_x), int(m))
    _, outer_lo, outer_hi = wilson_ci(int(outer_x), int(m))
    if inner_lo >= target:
        verdict = 'reliably_feasible'
    elif outer_hi < target:
        verdict = 'reliably_insufficient'
    else:
        verdict = 'undetermined'
    return verdict, (inner_lo, inner_hi), (outer_lo, outer_hi)


def _trial_job(args):
    (trial_index, candidates, base_seed, cyl_sides,
     ball_subdivisions) = args
    n_a_max = max(point[0] for point in candidates)
    n_b_max = max(point[1] for point in candidates)
    rng = np.random.default_rng(base_seed + trial_index)
    c, u, h = axis_geometry.generate_cylinders(n_a_max, rng)
    balls = fast_geometry.generate_balls(n_b_max, rng)
    inner, outer = strict_geometry.prepare_paired(
        c, u, h, balls, n_a_max=n_a_max, n_b_max=n_b_max,
        cyl_sides=cyl_sides, ball_subdivisions=ball_subdivisions)
    inner_y = []
    outer_y = []
    violations = 0
    for n_a, n_b in candidates:
        yi, yo = strict_geometry.paired_prefix(inner, outer, n_a, n_b)
        inner_y.append(int(yi))
        outer_y.append(int(yo))
        violations += int(yi and not yo)
    return {
        'trial_index': int(trial_index),
        'inner': inner_y,
        'outer': outer_y,
        'bracket_violations': violations,
        'inner_gjk': int(inner['n_gjk']),
        'outer_gjk': int(outer['n_gjk']),
    }


def _checkpoint_config(candidates, base_seed, cyl_sides, ball_subdivisions):
    return {
        'candidates': [list(point) for point in candidates],
        'base_seed': int(base_seed),
        'cyl_sides': int(cyl_sides),
        'ball_subdivisions': int(ball_subdivisions),
    }


def _load_checkpoint(config, resume):
    if not resume or not os.path.exists(CHECKPOINT):
        return {}
    with open(CHECKPOINT, encoding='utf-8') as handle:
        payload = json.load(handle)
    if payload.get('config') != config:
        raise ValueError('检查点参数与本次不同，请取消--resume或恢复原参数')
    return {int(row['trial_index']): row for row in payload.get('trials', [])}


def _save_checkpoint(config, rows):
    with open(CHECKPOINT, 'w', encoding='utf-8') as handle:
        json.dump({'config': config,
                   'trials': [rows[i] for i in sorted(rows)]},
                  handle, ensure_ascii=False, indent=2)


def run_trials(candidates, trials, base_seed, cyl_sides,
               ball_subdivisions, workers, resume=True):
    config = _checkpoint_config(
        candidates, base_seed, cyl_sides, ball_subdivisions)
    rows = _load_checkpoint(config, resume)
    missing = [i for i in range(trials) if i not in rows]
    jobs = [(i, candidates, base_seed, cyl_sides, ball_subdivisions)
            for i in missing]
    since_save = 0
    if jobs:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            for row in executor.map(_trial_job, jobs, chunksize=1):
                rows[int(row['trial_index'])] = row
                since_save += 1
                if since_save >= max(4, workers):
                    _save_checkpoint(config, rows)
                    since_save = 0
        _save_checkpoint(config, rows)
    return [rows[i] for i in range(trials)]


def summarize(candidates, rows, cyl_sides, ball_subdivisions, base_seed):
    m = len(rows)
    inner = np.asarray([row['inner'] for row in rows], dtype=np.int64)
    outer = np.asarray([row['outer'] for row in rows], dtype=np.int64)
    inner_x = inner.sum(axis=0)
    outer_x = outer.sum(axis=0)
    result_rows = []
    for index, (n_a, n_b) in enumerate(candidates):
        verdict, inner_ci, outer_ci = strict_verdict(
            inner_x[index], outer_x[index], m)
        result_rows.append({
            'N_A': n_a,
            'N_B': n_b,
            'cost_yuan': fast_geometry.cost(n_a, n_b),
            'inner_x': int(inner_x[index]),
            'inner_p_hat': float(inner_x[index] / m),
            'inner_ci_lower': float(inner_ci[0]),
            'inner_ci_upper': float(inner_ci[1]),
            'outer_x': int(outer_x[index]),
            'outer_p_hat': float(outer_x[index] / m),
            'outer_ci_lower': float(outer_ci[0]),
            'outer_ci_upper': float(outer_ci[1]),
            'strict_verdict': verdict,
        })
    result_rows.sort(key=lambda row: (row['cost_yuan'], row['N_A'], row['N_B']))
    feasible = [row for row in result_rows
                if row['strict_verdict'] == 'reliably_feasible']
    cheapest = min(feasible, key=lambda row: row['cost_yuan'], default=None)
    cheaper_unresolved = []
    if cheapest is not None:
        cheaper_unresolved = [row for row in result_rows
                              if row['cost_yuan'] < cheapest['cost_yuan']
                              and row['strict_verdict'] != 'reliably_insufficient']
    return {
        'method': 'A圆柱+B球体内接/外切多面体周期实体夹逼',
        'config': {
            'trials': m,
            'base_seed': base_seed,
            'cyl_sides': cyl_sides,
            'ball_subdivisions': ball_subdivisions,
            'ball_radial_error_nm': strict_geometry.ball_polyhedron_error(
                subdivisions=ball_subdivisions),
            'target_probability': TARGET,
            'same_source_auto_connection': False,
        },
        'bracket_violations': int(sum(
            row['bracket_violations'] for row in rows)),
        'mean_inner_gjk': float(np.mean([row['inner_gjk'] for row in rows])),
        'mean_outer_gjk': float(np.mean([row['outer_gjk'] for row in rows])),
        'candidate_results': result_rows,
        'cheapest_reliably_feasible': cheapest,
        'cheaper_unresolved_count': len(cheaper_unresolved),
        'cheaper_unresolved': cheaper_unresolved,
        'global_claim_ready_within_candidate_set': bool(
            cheapest is not None and not cheaper_unresolved),
    }


def main():
    parser = argparse.ArgumentParser(description='Q4严格混合实体候选复核')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args()

    trials = int(os.environ.get('SHUMO_Q4_STRICT_TRIALS', '4000'))
    workers = int(os.environ.get(
        'SHUMO_Q4_STRICT_WORKERS', str(min(8, os.cpu_count() or 1))))
    base_seed = int(os.environ.get('SHUMO_Q4_STRICT_SEED', '20260808'))
    cyl_sides = int(os.environ.get('SHUMO_Q4_STRICT_A_SIDES', '32'))
    ball_subdivisions = int(os.environ.get('SHUMO_Q4_STRICT_B_SUBDIV', '3'))
    if args.smoke:
        trials = min(trials, 8)
        cyl_sides = min(cyl_sides, 16)
        ball_subdivisions = min(ball_subdivisions, 1)
    if trials <= 0 or workers <= 0:
        raise ValueError('M和workers必须为正整数')
    candidates = tuple(DEFAULT_CANDIDATES)
    os.makedirs(RESULT_DIR, exist_ok=True)
    started = time.perf_counter()
    print(f'Q4严格混合复核：M={trials}, workers={workers}, '
          f'K_A={cyl_sides}, B_subdiv={ball_subdivisions}, '
          f'候选={len(candidates)}', flush=True)
    rows = run_trials(
        candidates, trials, base_seed, cyl_sides,
        ball_subdivisions, workers, resume=not args.no_resume)
    summary = summarize(
        candidates, rows, cyl_sides, ball_subdivisions, base_seed)
    summary['elapsed_s'] = time.perf_counter() - started

    with open(SUMMARY, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(CSV_RESULT, 'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(summary['candidate_results'][0].keys()))
        writer.writeheader()
        writer.writerows(summary['candidate_results'])

    print('\n===== 严格混合候选结果 =====')
    for row in summary['candidate_results']:
        print(f"({row['N_A']:3d},{row['N_B']:3d}) "
              f"C={row['cost_yuan']:.6f} "
              f"inner={row['inner_p_hat']:.4f} "
              f"outer={row['outer_p_hat']:.4f} "
              f"{row['strict_verdict']}")
    best = summary['cheapest_reliably_feasible']
    if best is None:
        print('当前样本下尚无严格可靠可行候选。')
    else:
        print(f"最低严格可靠候选：({best['N_A']},{best['N_B']})，"
              f"成本 {best['cost_yuan']:.6f} 元")
        print(f"其下仍待定候选：{summary['cheaper_unresolved_count']} 个")
    print(f"上下界违例：{summary['bracket_violations']}")
    print(f'结果：{SUMMARY}')


if __name__ == '__main__':
    main()
