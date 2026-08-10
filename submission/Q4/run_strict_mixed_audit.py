# 本程序及代码是在AI工具辅助下完成的
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31


import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from statistics import NormalDist

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (HERE, os.path.join(ROOT, 'Q2')):
    if path not in sys.path:
        sys.path.insert(0, path)

import geometry as axis_geometry
import geometry_mix as fast_geometry
import solid_mix_geometry as strict_geometry
from monte_carlo import wilson_ci


TARGET = 0.90
FAMILY_ALPHA = 0.05
FORMAL_RECOMMENDATION = (619, 0)
FORMAL_TRIALS = 4000
A_SEQUENCE_LENGTH_DEFAULT = 750
MIXED_CANDIDATES = (
    (598, 62),
    (609, 5), (608, 14), (610, 4),
    (611, 3), (610, 12), (609, 21),
    (617, 1),
)
DEFAULT_CANDIDATES = MIXED_CANDIDATES
RESULT_DIR = os.path.join(HERE, 'results')
CHECKPOINT = os.path.join(
    RESULT_DIR, 'question4_strict_mixed_checkpoint_v2_nA750.json')
CHECKPOINT_BACKUP = CHECKPOINT + '.bak'
SUMMARY = os.path.join(RESULT_DIR, 'question4_strict_mixed_summary.json')
CSV_RESULT = os.path.join(RESULT_DIR, 'question4_strict_mixed_candidates.csv')


def strict_verdict(inner_x, outer_x, m, target=TARGET):
    _, inner_lo, inner_hi = wilson_ci(int(inner_x), int(m))
    _, outer_lo, outer_hi = wilson_ci(int(outer_x), int(m))
    if inner_lo >= target:
        verdict = 'reliably_feasible'
    elif outer_hi < target:
        verdict = 'reliably_insufficient'
    else:
        verdict = 'undetermined'
    return verdict, (inner_lo, inner_hi), (outer_lo, outer_hi)


def wilson_ci_alpha(x, m, alpha):
    if m <= 0 or not (0.0 < alpha < 1.0):
        raise ValueError('m必须为正且alpha必须位于0与1之间')
    p = float(x) / float(m)
    z = NormalDist().inv_cdf(1.0 - float(alpha) / 2.0)
    z2 = z * z
    denominator = 1.0 + z2 / m
    center = (p + z2 / (2.0 * m)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / m + z2 / (4.0 * m * m)) \
        / denominator
    return p, max(0.0, center - half), min(1.0, center + half)


def joint_strict_verdict(inner_x, outer_x, m, n_candidates,
                         target=TARGET, family_alpha=FAMILY_ALPHA):
    alpha_each = float(family_alpha) / (2 * int(n_candidates))
    _, inner_lo, inner_hi = wilson_ci_alpha(inner_x, m, alpha_each)
    _, outer_lo, outer_hi = wilson_ci_alpha(outer_x, m, alpha_each)
    if inner_lo >= target:
        verdict = 'joint_reliably_feasible'
    elif outer_hi < target:
        verdict = 'joint_reliably_insufficient'
    else:
        verdict = 'joint_undetermined'
    return verdict, (inner_lo, inner_hi), (outer_lo, outer_hi), alpha_each


def _generate_trial_geometry(seed, n_a_used, n_b_used, a_sequence_length):
    if a_sequence_length < n_a_used:
        raise ValueError('A固定序列长度不能小于实际使用数量')
    rng = np.random.default_rng(seed)
    c_all, u_all, h_all = axis_geometry.generate_cylinders(
        a_sequence_length, rng)
    balls = fast_geometry.generate_balls(n_b_used, rng)
    return (c_all[:n_a_used], u_all[:n_a_used], h_all[:n_a_used], balls)


def _trial_job(args):
    (trial_index, candidates, base_seed, cyl_sides,
     ball_subdivisions, a_sequence_length) = args
    n_a_max = max(point[0] for point in candidates)
    n_b_max = max(point[1] for point in candidates)



    c, u, h, balls = _generate_trial_geometry(
        base_seed + trial_index, n_a_max, n_b_max, a_sequence_length)
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


def _checkpoint_config(candidates, base_seed, cyl_sides, ball_subdivisions,
                       a_sequence_length):
    return {
        'candidates': [list(point) for point in candidates],
        'base_seed': int(base_seed),
        'cyl_sides': int(cyl_sides),
        'ball_subdivisions': int(ball_subdivisions),
        'a_sequence_length': int(a_sequence_length),
    }


def _load_checkpoint(config, resume):
    if not resume:
        return {}
    payload = None
    load_errors = []
    for path in (CHECKPOINT, CHECKPOINT_BACKUP):
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding='utf-8') as handle:
                payload = json.load(handle)
            break
        except (OSError, json.JSONDecodeError) as exc:
            load_errors.append(f'{path}: {exc}')
    if payload is None:
        if load_errors:
            raise ValueError('主检查点与备份均无法读取：' + '; '.join(load_errors))
        return {}
    if payload.get('config') != config:
        raise ValueError('检查点参数与本次不同，请取消--resume或恢复原参数')
    return {int(row['trial_index']): row for row in payload.get('trials', [])}


def _save_checkpoint(config, rows):
    temporary = CHECKPOINT + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump({'config': config,
                   'trials': [rows[i] for i in sorted(rows)]},
                  handle, ensure_ascii=False, separators=(',', ':'))
        handle.flush()
        os.fsync(handle.fileno())
    if os.path.exists(CHECKPOINT):
        os.replace(CHECKPOINT, CHECKPOINT_BACKUP)
    os.replace(temporary, CHECKPOINT)


def run_trials(candidates, trials, base_seed, cyl_sides,
               ball_subdivisions, workers,
               a_sequence_length=A_SEQUENCE_LENGTH_DEFAULT, resume=True,
               checkpoint_every=None):
    config = _checkpoint_config(
        candidates, base_seed, cyl_sides, ball_subdivisions,
        a_sequence_length)
    rows = _load_checkpoint(config, resume)
    missing = [i for i in range(trials) if i not in rows]
    jobs = [(i, candidates, base_seed, cyl_sides, ball_subdivisions,
             a_sequence_length)
            for i in missing]
    if checkpoint_every is None:
        checkpoint_every = int(os.environ.get(
            'SHUMO_Q4_CHECKPOINT_EVERY', str(max(50, 4 * workers))))
    checkpoint_every = max(1, int(checkpoint_every))
    since_save = 0
    if jobs:
        try:
            with ProcessPoolExecutor(
                    max_workers=min(workers, len(jobs))) as executor:
                for row in executor.map(_trial_job, jobs, chunksize=1):
                    rows[int(row['trial_index'])] = row
                    since_save += 1
                    if since_save >= checkpoint_every:
                        _save_checkpoint(config, rows)
                        since_save = 0
        finally:
            if since_save:
                _save_checkpoint(config, rows)
    return [rows[i] for i in range(trials)]


def summarize(candidates, rows, cyl_sides, ball_subdivisions, base_seed,
              a_sequence_length=A_SEQUENCE_LENGTH_DEFAULT):
    m = len(rows)
    inner = np.asarray([row['inner'] for row in rows], dtype=np.int64)
    outer = np.asarray([row['outer'] for row in rows], dtype=np.int64)
    inner_x = inner.sum(axis=0)
    outer_x = outer.sum(axis=0)
    result_rows = []
    n_candidates = len(candidates)
    for index, (n_a, n_b) in enumerate(candidates):
        verdict, inner_ci, outer_ci = strict_verdict(
            inner_x[index], outer_x[index], m)
        joint_verdict, joint_inner_ci, joint_outer_ci, alpha_each = \
            joint_strict_verdict(
                inner_x[index], outer_x[index], m, n_candidates)
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
            'joint_inner_ci_lower': float(joint_inner_ci[0]),
            'joint_inner_ci_upper': float(joint_inner_ci[1]),
            'joint_outer_ci_lower': float(joint_outer_ci[0]),
            'joint_outer_ci_upper': float(joint_outer_ci[1]),
            'joint_strict_verdict': joint_verdict,
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
    joint_feasible = [row for row in result_rows
                      if row['joint_strict_verdict']
                      == 'joint_reliably_feasible']
    joint_cheapest = min(
        joint_feasible, key=lambda row: row['cost_yuan'], default=None)
    joint_cheaper_unresolved = []
    if joint_cheapest is not None:
        joint_cheaper_unresolved = [
            row for row in result_rows
            if row['cost_yuan'] < joint_cheapest['cost_yuan']
            and row['joint_strict_verdict']
            != 'joint_reliably_insufficient']
    joint_insufficient = [
        row for row in result_rows
        if row['joint_strict_verdict'] == 'joint_reliably_insufficient']
    joint_undetermined = [
        row for row in result_rows
        if row['joint_strict_verdict'] == 'joint_undetermined']
    joint_mixed_feasible = [
        row for row in result_rows
        if row['joint_strict_verdict'] == 'joint_reliably_feasible']
    return {
        'method': 'A圆柱+B球体内接/外切多面体周期实体夹逼',
        'config': {
            'trials': m,
            'base_seed': base_seed,
            'cyl_sides': cyl_sides,
            'ball_subdivisions': ball_subdivisions,
            'a_sequence_length': a_sequence_length,
            'ball_radial_error_nm': strict_geometry.ball_polyhedron_error(
                subdivisions=ball_subdivisions),
            'target_probability': TARGET,
            'family_confidence': 1.0 - FAMILY_ALPHA,
            'bonferroni_alpha_each': alpha_each,
            'same_source_auto_connection': False,
        },
        'bracket_violations': int(sum(
            row['bracket_violations'] for row in rows)),
        'mean_inner_gjk': float(np.mean([row['inner_gjk'] for row in rows])),
        'mean_outer_gjk': float(np.mean([row['outer_gjk'] for row in rows])),
        'candidate_results': result_rows,
        'stage_pointwise_cheapest_reliably_feasible': cheapest,
        'cheaper_unresolved_count': len(cheaper_unresolved),
        'cheaper_unresolved': cheaper_unresolved,
        'stage_joint_cheapest_reliably_feasible': joint_cheapest,
        'joint_cheaper_unresolved_count': len(joint_cheaper_unresolved),
        'joint_cheaper_unresolved': joint_cheaper_unresolved,
        'screened_mixed_status_counts': {
            'reliably_insufficient': len(joint_insufficient),
            'undetermined': len(joint_undetermined),
            'reliably_feasible': len(joint_mixed_feasible),
        },
        'formal_unified_recommendation': {
            'N_A': FORMAL_RECOMMENDATION[0],
            'N_B': FORMAL_RECOMMENDATION[1],
            'source': 'Q3 M=10000 strict solid bound',
            'candidate_audit': None,
            'note': '纯A不纳入Q4混合候选重复检验',
        },
        'formal_recommendation_supported_within_screened_mixed_set': bool(
            len(joint_insufficient) == len(result_rows)),
        'global_claim_ready_within_candidate_set': bool(
            not joint_undetermined),
    }


def _write_result_files(summary):
    summary_temporary = SUMMARY + '.tmp'
    with open(summary_temporary, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(summary_temporary, SUMMARY)

    csv_temporary = CSV_RESULT + '.tmp'
    with open(csv_temporary, 'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(summary['candidate_results'][0].keys()))
        writer.writeheader()
        writer.writerows(summary['candidate_results'])
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(csv_temporary, CSV_RESULT)


def _configure_windows_runtime_limit():
    affinity_text = os.environ.get('SHUMO_Q4_AFFINITY_CORES', '').strip()
    affinity_mask_text = os.environ.get('SHUMO_Q4_AFFINITY_MASK', '').strip()
    below_normal = os.environ.get(
        'SHUMO_Q4_BELOW_NORMAL_PRIORITY', '').strip().lower() in {
            '1', 'true', 'yes', 'on',
        }
    if os.name != 'nt' or (
            not affinity_text and not affinity_mask_text and not below_normal):
        return None

    import ctypes

    logical_cores = os.cpu_count() or 1
    affinity_cores = logical_cores
    affinity_mask = (1 << min(logical_cores, 63)) - 1
    if affinity_mask_text:
        affinity_mask = int(affinity_mask_text, 0) & affinity_mask
        if affinity_mask == 0:
            raise ValueError('SHUMO_Q4_AFFINITY_MASK不得为空掩码')
        affinity_cores = affinity_mask.bit_count()
    elif affinity_text:
        affinity_cores = max(1, min(int(affinity_text), logical_cores, 63))
        affinity_mask = (1 << affinity_cores) - 1
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetProcessAffinityMask.argtypes = (
        ctypes.c_void_p, ctypes.c_size_t)
    kernel32.SetProcessAffinityMask.restype = ctypes.c_int
    kernel32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    kernel32.SetPriorityClass.restype = ctypes.c_int
    process = kernel32.GetCurrentProcess()
    if affinity_text or affinity_mask_text:
        if not kernel32.SetProcessAffinityMask(process, affinity_mask):
            raise ctypes.WinError(ctypes.get_last_error())
    priority = 'normal'
    if below_normal:
        below_normal_priority_class = 0x00004000
        if not kernel32.SetPriorityClass(
                process, below_normal_priority_class):
            raise ctypes.WinError(ctypes.get_last_error())
        priority = 'below_normal'
    return {
        'affinity_cores': affinity_cores,
        'logical_cores': logical_cores,
        'affinity_mask': hex(affinity_mask),
        'priority': priority,
    }


def main():
    runtime_limit = _configure_windows_runtime_limit()
    parser = argparse.ArgumentParser(description='Q4严格混合实体候选复核')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--status', action='store_true',
                        help='只查看检查点状态，不启动模拟')
    args = parser.parse_args()

    trials = int(os.environ.get(
        'SHUMO_Q4_STRICT_TRIALS', str(FORMAL_TRIALS)))
    workers = int(os.environ.get(
        'SHUMO_Q4_STRICT_WORKERS', str(min(8, os.cpu_count() or 1))))
    base_seed = int(os.environ.get('SHUMO_Q4_STRICT_SEED', '20260808'))
    cyl_sides = int(os.environ.get('SHUMO_Q4_STRICT_A_SIDES', '32'))
    ball_subdivisions = int(os.environ.get('SHUMO_Q4_STRICT_B_SUBDIV', '3'))
    a_sequence_length = int(os.environ.get(
        'SHUMO_Q4_STRICT_A_SEQUENCE', str(A_SEQUENCE_LENGTH_DEFAULT)))
    if args.smoke:
        trials = min(trials, 8)
        cyl_sides = min(cyl_sides, 16)
        ball_subdivisions = min(ball_subdivisions, 1)
    if trials <= 0 or workers <= 0:
        raise ValueError('M和workers必须为正整数')
    candidates = tuple(DEFAULT_CANDIDATES)
    os.makedirs(RESULT_DIR, exist_ok=True)
    checkpoint_config = _checkpoint_config(
        candidates, base_seed, cyl_sides, ball_subdivisions,
        a_sequence_length)
    if args.status:
        rows = _load_checkpoint(checkpoint_config, resume=True)
        print(f'严格混合检查点：saved={len(rows)}, target={trials}, '
              f'workers={workers}, K_A={cyl_sides}, '
              f'B_subdiv={ball_subdivisions}, '
              f'A_sequence={a_sequence_length}')
        return
    started = time.perf_counter()
    print(f'Q4严格混合复核：M={trials}, workers={workers}, '
          f'K_A={cyl_sides}, B_subdiv={ball_subdivisions}, '
          f'候选={len(candidates)}', flush=True)
    if runtime_limit:
        print(f"Windows资源限制：affinity={runtime_limit['affinity_cores']}/"
              f"{runtime_limit['logical_cores']} logical cores, "
              f"mask={runtime_limit['affinity_mask']}, "
              f"priority={runtime_limit['priority']}", flush=True)
    rows = run_trials(
        candidates, trials, base_seed, cyl_sides,
        ball_subdivisions, workers, a_sequence_length,
        resume=not args.no_resume)
    summary = summarize(
        candidates, rows, cyl_sides, ball_subdivisions, base_seed,
        a_sequence_length)
    summary['elapsed_s'] = time.perf_counter() - started
    summary['predeclared_formal_trials'] = FORMAL_TRIALS
    summary['run_status'] = (
        'formal_complete_M4000'
        if trials == FORMAL_TRIALS and not args.smoke
        else 'nonformal_custom_or_smoke_run')

    _write_result_files(summary)

    print('\n===== 严格混合候选结果 =====')
    for row in summary['candidate_results']:
        print(f"({row['N_A']:3d},{row['N_B']:3d}) "
              f"C={row['cost_yuan']:.6f} "
              f"inner={row['inner_p_hat']:.4f} "
              f"outer={row['outer_p_hat']:.4f} "
              f"{row['strict_verdict']} / {row['joint_strict_verdict']}")
    best = summary['stage_pointwise_cheapest_reliably_feasible']
    if best is None:
        print('当前样本下尚无逐点95%严格可靠可行候选。')
    else:
        print(f"本批M={trials}逐点95%阶段最低可靠候选："
              f"({best['N_A']},{best['N_B']})，"
              f"成本 {best['cost_yuan']:.6f} 元")
        print(f"其下仍待定候选：{summary['cheaper_unresolved_count']} 个")
    joint_best = summary['stage_joint_cheapest_reliably_feasible']
    if joint_best is None:
        print('候选族联合95%口径下尚无可靠可行候选。')
    else:
        print(f"候选族联合95%阶段最低可靠候选："
              f"({joint_best['N_A']},{joint_best['N_B']})；"
              f"其下仍待定 {summary['joint_cheaper_unresolved_count']} 个")
    counts = summary['screened_mixed_status_counts']
    print('含B候选联合判定：'
          f"不足 {counts['reliably_insufficient']}，"
          f"待定 {counts['undetermined']}，"
          f"可行 {counts['reliably_feasible']}")
    print('纯A保守基准（来自Q3 M=10000）：(619,0)；'
          'Q4只检验含B方案能否以更低成本替代该基准。')
    print(f"上下界违例：{summary['bracket_violations']}")
    print(f'结果：{SUMMARY}')


if __name__ == '__main__':
    main()
