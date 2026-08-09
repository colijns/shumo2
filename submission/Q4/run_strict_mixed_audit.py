"""Q4低成本临界候选的严格混合实体复核。

快速Q4程序负责定位候选；本程序对候选集合共享同一随机介质前缀，同时计算
内接和外切实体模型。严格判定规则：

* inner的Wilson下限 >= 0.90：可靠可行；
* outer的Wilson上限 < 0.90：可靠不足；
* 其他：待追加样本或提高几何分辨率。

默认候选包含原Q4搜索点、7个低成本跨线点、617附近纯A点以及统一推荐619。
程序按trial保存检查点，M可从小样本直接扩展到正式样本而不重复已完成试验。

常用环境变量：SHUMO_Q4_STRICT_TRIALS、SHUMO_Q4_STRICT_WORKERS、
SHUMO_Q4_STRICT_A_SEQUENCE（统一口径固定为750）、SHUMO_Q4_CHECKPOINT_EVERY。
Windows可选SHUMO_Q4_AFFINITY_CORES或SHUMO_Q4_AFFINITY_MASK限制处理器亲和性，
并以SHUMO_Q4_BELOW_NORMAL_PRIORITY=1降低优先级。使用--status只查看检查点。
"""

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

import geometry as axis_geometry  # noqa: E402
import geometry_mix as fast_geometry  # noqa: E402
import solid_mix_geometry as strict_geometry  # noqa: E402
from monte_carlo import wilson_ci  # noqa: E402


TARGET = 0.90
FAMILY_ALPHA = 0.05
FORMAL_RECOMMENDATION = (619, 0)
A_SEQUENCE_LENGTH_DEFAULT = 750
DEFAULT_CANDIDATES = (
    (598, 62),
    (609, 5), (608, 14), (610, 4), (611, 0), (610, 12),
    (616, 0), (617, 0), (617, 1), (618, 0), (619, 0),
)
RESULT_DIR = os.path.join(HERE, 'results')
CHECKPOINT = os.path.join(
    RESULT_DIR, 'question4_strict_mixed_checkpoint_nA750.json')
CHECKPOINT_BACKUP = CHECKPOINT + '.bak'
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


def wilson_ci_alpha(x, m, alpha):
    """任意显著性水平的Wilson区间，用于多候选联合置信校正。"""
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
    """Bonferroni联合置信下的严格三分类。

    每个候选同时包含内界与外界两个概率参数，因此总计有
    ``2 * n_candidates`` 个区间需要联合覆盖。
    """
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
    """按Q3固定A序列长度生成一次试验，再截取实际使用前缀。"""
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
    # 必须先生成与Q3相同长度的完整A序列，再取候选前缀。generate_cylinders
    # 分别批量抽取中心、方向和长度；若直接生成619而Q3生成750，方向/长度
    # 的随机数位置会错位，导致“同seed”并非同一个微构体样本。
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
    formal_row = next(
        (row for row in result_rows
         if (row['N_A'], row['N_B']) == FORMAL_RECOMMENDATION), None)
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
        'formal_unified_recommendation': {
            'N_A': FORMAL_RECOMMENDATION[0],
            'N_B': FORMAL_RECOMMENDATION[1],
            'source': 'Q3 M=10000 strict solid bound',
            'candidate_audit': formal_row,
        },
        'global_claim_ready_within_candidate_set': bool(
            joint_cheapest is not None and not joint_cheaper_unresolved),
    }


def _write_result_files(summary):
    """原子写出汇总JSON与候选CSV，避免中断后留下半文件。"""
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
    """Optionally cap Windows CPU affinity before the process pool is created.

    Worker processes inherit the parent's affinity and priority.  This permits a
    hard machine-wide CPU ceiling without changing trial seeds or model logic.
    """
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

    trials = int(os.environ.get('SHUMO_Q4_STRICT_TRIALS', '4000'))
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
    print('最终统一保守推荐（来自Q3 M=10000）：(619,0)，'
          '不由本批阶段最低点覆盖。')
    print(f"上下界违例：{summary['bracket_violations']}")
    print(f'结果：{SUMMARY}')


if __name__ == '__main__':
    main()
