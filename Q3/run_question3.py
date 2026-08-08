# 本程序及代码是在AI工具辅助下完成的
"""问题3：搜索导通概率不低于90%的介质A最低体积分数。

复用 Q2 的随机生成、边界截断、有限圆柱距离和图连通内核。搜索分两阶段：
1. 在 0.01 个百分点的离散网格上自适应蒙特卡洛二分搜索；
2. 对临界点及其相邻网格点使用独立随机种子做固定样本确认。

同源片段仅用于来源与体积追踪，不自动建立电学连接。
"""

from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
import os
import sys
import time

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q2_DIR = os.path.join(ROOT, 'Q2')
sys.path.insert(0, Q2_DIR)

import monte_carlo as mc  # noqa: E402


TARGET = float(os.environ.get('SHUMO_Q3_TARGET', '0.90'))
# 0.01 个百分点 = 0.0001 的体积分数。
GRID_STEP = float(os.environ.get('SHUMO_Q3_GRID_STEP', '0.0001'))
START_HIGH = float(os.environ.get('SHUMO_Q3_START_HIGH', '0.0100'))
MAX_PHI = float(os.environ.get('SHUMO_Q3_MAX_PHI', '0.0200'))
MIN_TRIALS = int(os.environ.get('SHUMO_Q3_MIN_TRIALS', '200'))
MAX_TRIALS = int(os.environ.get('SHUMO_Q3_MAX_TRIALS', '1000'))
CONFIRM_TRIALS = int(os.environ.get('SHUMO_Q3_CONFIRM_TRIALS', '2000'))
CONFIRM_RADIUS = int(os.environ.get('SHUMO_Q3_CONFIRM_RADIUS', '2'))
BATCH_SIZE = int(os.environ.get('SHUMO_Q3_BATCH_SIZE', '25'))
MAX_WORKERS = int(os.environ.get(
    'SHUMO_Q3_WORKERS', str(min(8, os.cpu_count() or 1))))
BASE_SEED = int(os.environ.get('SHUMO_Q3_BASE_SEED', '20260808'))

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
HISTORY_CSV = os.path.join(RESULT_DIR, 'question3_search_history.csv')
SUMMARY_JSON = os.path.join(RESULT_DIR, 'question3_summary.json')


def phi_to_index(phi):
    """体积分数转为0.01个百分点网格编号。"""
    return int(round(float(phi) / GRID_STEP))


def index_to_phi(index):
    """网格编号转体积分数。"""
    return int(index) * GRID_STEP


def _batch_worker(args):
    """进程池任务：完成一个体积分数的一批独立样本。"""
    phi, m, seed = args
    return mc.simulate_phi(phi, m=m, seed=seed)


def _coupled_batch_worker(args):
    """共同随机样本确认：同一最大构型的圆柱前缀对应不同填充量。

    对每个随机微构体先生成 MAX_PHI 对应的最大圆柱集合；较低填充量取其
    前 n 根圆柱。由于增加圆柱不会删除已有接触边，单个样本的导通状态随
    填充量单调不减，从而消除相邻体积分数独立抽样造成的反常非单调。
    """
    grid_indices, max_grid_index, m, seed = args
    indices = sorted(int(i) for i in grid_indices)
    phis = {i: index_to_phi(i) for i in indices}
    counts = {i: mc.n_cylinders(phis[i]) for i in indices}
    max_n = mc.n_cylinders(index_to_phi(max_grid_index))
    rng = np.random.default_rng(seed)
    stats = {i: {'x': 0} for i in indices}
    t0 = time.perf_counter()
    for _ in range(m):
        c, u, h = mc.geo.generate_cylinders(max_n, rng)
        previous_y = False
        for i in indices:
            # 图连通性对新增圆柱单调不减：较低填充量已经导通时，
            # 后续更高填充量无需重复截断、距离计算和构图。
            if previous_y:
                stats[i]['x'] += 1
                continue
            n = counts[i]
            res = mc.geo.sample_conductive(c[:n], u[:n], h[:n])
            current_y = bool(res['conductive'])
            previous_y = current_y
            stats[i]['x'] += int(current_y)
    elapsed = time.perf_counter() - t0
    return {
        i: {
            'phi': phis[i],
            'n_A': counts[i],
            'm': m,
            'x': stats[i]['x'],
            'p_hat': stats[i]['x'] / m,
            'ci_lower': 0.0,
            'ci_upper': 1.0,
            # 共同确认启用了单调提前终止，不再汇总非必要的几何性能指标。
            'mean_fragments': float('nan'),
            'mean_edges': float('nan'),
            'mean_gjk': float('nan'),
            'elapsed_s': elapsed,
        }
        for i in indices
    }


def merge_batches(phi, batches):
    """合并同一体积分数的批次统计结果。"""
    total_m = sum(r['m'] for r in batches)
    total_x = sum(r['x'] for r in batches)
    p_hat, lo, hi = mc.wilson_ci(total_x, total_m)
    return {
        'phi': float(phi),
        'phi_percent': 100.0 * float(phi),
        'n_A': batches[0]['n_A'],
        'm': total_m,
        'x': total_x,
        'p_hat': p_hat,
        'ci_lower': lo,
        'ci_upper': hi,
        'mean_fragments': sum(r['mean_fragments'] * r['m'] for r in batches) / total_m,
        'mean_edges': sum(r['mean_edges'] * r['m'] for r in batches) / total_m,
        'mean_gjk': sum(r['mean_gjk'] * r['m'] for r in batches) / total_m,
        'worker_elapsed_s': sum(r['elapsed_s'] for r in batches),
    }


def interval_classification(result, target=TARGET):
    """根据置信区间判断概率是否明确位于目标值一侧。"""
    if result['ci_lower'] >= target:
        return 'above'
    if result['ci_upper'] < target:
        return 'below'
    return 'uncertain'


def point_decision(result, target=TARGET):
    """搜索最终使用点估计作二元决策，同时保留区间是否明确的信息。"""
    return 'above' if result['p_hat'] >= target else 'below'


def select_confirmed_bracket(rows, target=TARGET):
    """从按网格升序排列的共同样本结果中选择首次达到目标的点。"""
    ordered = sorted(rows, key=lambda r: r['grid_index'])
    for pos, row in enumerate(ordered):
        if row['p_hat'] >= target:
            if pos == 0:
                return row['grid_index'], False
            return row['grid_index'], ordered[pos - 1]['p_hat'] < target
    return None, False


def evaluate_coupled_confirmation(executor, grid_indices, trials,
                                  workers, batch_size, base_seed,
                                  max_grid_index, target=TARGET):
    """并行执行临界邻域的共同随机样本确认。"""
    indices = sorted(set(int(i) for i in grid_indices))
    jobs = []
    remaining = trials
    batch_index = 0
    while remaining:
        m = min(batch_size, remaining)
        seed = base_seed + 200_000_000 + batch_index
        jobs.append((indices, max_grid_index, m, seed))
        remaining -= m
        batch_index += 1
    t0 = time.perf_counter()
    raw_batches = list(executor.map(_coupled_batch_worker, jobs))
    wall_elapsed = time.perf_counter() - t0
    rows = []
    for index in indices:
        batches = [batch[index] for batch in raw_batches]
        row = merge_batches(index_to_phi(index), batches)
        row['grid_index'] = index
        row['phase'] = 'confirm_coupled'
        row['interval_classification'] = interval_classification(row, target)
        row['decision'] = point_decision(row, target)
        row['decisive'] = row['interval_classification'] != 'uncertain'
        row['wall_elapsed_s'] = wall_elapsed
        rows.append(row)
    return rows


class ParallelEstimator:
    """复用同一进程池完成多个候选体积分数的自适应估计。"""

    def __init__(self, executor, workers=MAX_WORKERS, batch_size=BATCH_SIZE,
                 base_seed=BASE_SEED, target=TARGET):
        self.executor = executor
        self.workers = workers
        self.batch_size = batch_size
        self.base_seed = base_seed
        self.target = target
        self.records = []

    def _seed(self, grid_index, phase_code, batch_index):
        return (self.base_seed + phase_code * 100_000_000
                + grid_index * 10_000 + batch_index)

    def evaluate(self, grid_index, min_trials=MIN_TRIALS,
                 max_trials=MAX_TRIALS, phase='search'):
        """自适应估计：置信区间明确越过目标后可提前停止。"""
        phi = index_to_phi(grid_index)
        batches = []
        total = 0
        batch_index = 0
        t0 = time.perf_counter()
        phase_code = 0 if phase == 'search' else 1

        while total < max_trials:
            jobs = []
            for _ in range(self.workers):
                if total + sum(j[1] for j in jobs) >= max_trials:
                    break
                m = min(self.batch_size,
                        max_trials - total - sum(j[1] for j in jobs))
                jobs.append((phi, m,
                             self._seed(grid_index, phase_code, batch_index)))
                batch_index += 1
            new_results = list(self.executor.map(_batch_worker, jobs))
            batches.extend(new_results)
            total = sum(r['m'] for r in batches)
            result = merge_batches(phi, batches)
            classification = interval_classification(result, self.target)
            if total >= min_trials and classification != 'uncertain':
                break

        result['grid_index'] = grid_index
        result['phase'] = phase
        result['interval_classification'] = classification
        result['decision'] = point_decision(result, self.target)
        result['decisive'] = classification != 'uncertain'
        result['wall_elapsed_s'] = time.perf_counter() - t0
        self.records.append(result)
        return result

    def evaluate_fixed(self, grid_index, trials=CONFIRM_TRIALS,
                       phase='confirm'):
        """使用独立种子完成固定样本量的临界点确认。"""
        return self.evaluate(grid_index, min_trials=trials,
                             max_trials=trials, phase=phase)


def find_first_feasible(evaluate_index, low_index, high_index):
    """在已知低端不可行、高端可行的整数网格上二分搜索。"""
    low = int(low_index)
    high = int(high_index)
    while high - low > 1:
        mid = (low + high) // 2
        result = evaluate_index(mid)
        if result['decision'] == 'above':
            high = mid
        else:
            low = mid
    return low, high


def _write_results(records, summary):
    os.makedirs(RESULT_DIR, exist_ok=True)
    fields = [
        'phase', 'grid_index', 'phi', 'phi_percent', 'n_A', 'm', 'x',
        'p_hat', 'ci_lower', 'ci_upper', 'interval_classification',
        'decision', 'decisive', 'mean_fragments', 'mean_edges', 'mean_gjk',
        'worker_elapsed_s', 'wall_elapsed_s',
    ]
    with open(HISTORY_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row[key] for key in fields})
    with open(SUMMARY_JSON, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def _validate_config():
    if not (0.0 < TARGET < 1.0):
        raise ValueError('TARGET 必须位于 (0,1)')
    if GRID_STEP <= 0 or START_HIGH <= 0 or MAX_PHI < START_HIGH:
        raise ValueError('体积分数搜索范围或网格步长无效')
    if not (0 < MIN_TRIALS <= MAX_TRIALS):
        raise ValueError('应满足 0 < MIN_TRIALS <= MAX_TRIALS')
    if (CONFIRM_TRIALS <= 0 or CONFIRM_RADIUS <= 0
            or BATCH_SIZE <= 0 or MAX_WORKERS <= 0):
        raise ValueError('样本量、批大小和进程数必须为正整数')


def main():
    _validate_config()
    t0 = time.perf_counter()
    start_high_index = phi_to_index(START_HIGH)
    max_index = phi_to_index(MAX_PHI)
    workers = min(
        MAX_WORKERS,
        max(1, math.ceil(MAX_TRIALS / BATCH_SIZE),
            math.ceil(CONFIRM_TRIALS / BATCH_SIZE)),
    )
    print(f'Q3设置: target={TARGET:.2f}, step={GRID_STEP*100:.2f}%, '
          f'workers={workers}, search_trials={MIN_TRIALS}-{MAX_TRIALS}, '
          f'confirm_trials={CONFIRM_TRIALS}', flush=True)

    cache = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        estimator = ParallelEstimator(executor, workers=workers)

        def evaluate_search(index):
            if index not in cache:
                row = estimator.evaluate(index)
                cache[index] = row
                print(f'[搜索] phi={row["phi_percent"]:.2f}%, '
                      f'p_hat={row["p_hat"]:.4f}, '
                      f'CI=[{row["ci_lower"]:.4f},{row["ci_upper"]:.4f}], '
                      f'M={row["m"]}, decision={row["decision"]}',
                      flush=True)
            return cache[index]

        # φ=0 时无圆柱，必不导通，因此低端无需模拟。
        low_index = 0
        high_index = start_high_index
        high_result = evaluate_search(high_index)
        expansion = max(1, phi_to_index(0.0020))  # 每次向上扩0.20个百分点
        while high_result['decision'] != 'above' and high_index < max_index:
            low_index = high_index
            high_index = min(max_index, high_index + expansion)
            high_result = evaluate_search(high_index)
        if high_result['decision'] != 'above':
            raise RuntimeError('在设定的最大体积分数内未找到90%可行点')

        low_index, high_index = find_first_feasible(
            evaluate_search, low_index, high_index)

        # 使用独立随机种子确认临界点及其相邻网格点。
        confirm_indices = list(range(
            max(0, high_index - CONFIRM_RADIUS),
            min(max_index, high_index + CONFIRM_RADIUS) + 1,
        ))
        confirm_rows = evaluate_coupled_confirmation(
            executor, confirm_indices, CONFIRM_TRIALS, workers,
            BATCH_SIZE, BASE_SEED, max_index, TARGET)
        estimator.records.extend(confirm_rows)
        for row in confirm_rows:
            print(f'[确认] phi={row["phi_percent"]:.2f}%, '
                  f'p_hat={row["p_hat"]:.4f}, '
                  f'CI=[{row["ci_lower"]:.4f},{row["ci_upper"]:.4f}], '
                  f'M={row["m"]}', flush=True)

    confirm_rows.sort(key=lambda r: r['grid_index'])
    confirmed_index, bracket_valid = select_confirmed_bracket(
        confirm_rows, TARGET)
    # 只有共同样本在临界邻域形成有效夹逼时才修正搜索候选。
    recommended_index = (confirmed_index if bracket_valid else high_index)
    recommended_phi = index_to_phi(recommended_index)
    confirmed = next(
        (r for r in confirm_rows if r['grid_index'] == recommended_index), None)

    monotone = all(
        a['p_hat'] <= b['p_hat'] + 1e-12
        for a, b in zip(confirm_rows[:-1], confirm_rows[1:])
    )
    previous = next(
        (r for r in confirm_rows if r['grid_index'] == recommended_index - 1),
        None,
    )
    interval_certified = (
        confirmed is not None and confirmed['ci_lower'] >= TARGET
        and previous is not None and previous['ci_upper'] < TARGET
    )
    needs_more_trials = not interval_certified

    summary = {
        'target_probability': TARGET,
        'grid_step_fraction': GRID_STEP,
        'grid_step_percentage_point': GRID_STEP * 100.0,
        'search_candidate_phi': index_to_phi(high_index),
        'recommended_phi': recommended_phi,
        'recommended_phi_percent': recommended_phi * 100.0,
        'recommended_n_A': mc.n_cylinders(recommended_phi),
        'confirmation_monotone': bool(monotone),
        'confirmation_bracket_valid': bool(bracket_valid),
        'interval_certified': bool(interval_certified),
        'needs_more_trials': bool(needs_more_trials),
        'search_trials_range': [MIN_TRIALS, MAX_TRIALS],
        'confirmation_trials': CONFIRM_TRIALS,
        'base_seed': int(BASE_SEED),
        'total_wall_elapsed_s': float(time.perf_counter() - t0),
    }
    _write_results(estimator.records, summary)

    print('\n===== 问题3阶段性结论 =====')
    print(f'候选最低填充量: {recommended_phi*100:.2f}% '
          f'(N_A={summary["recommended_n_A"]})')
    print(f'临界邻域单调: {monotone}, 点估计夹逼成立: {bracket_valid}, '
          f'置信区间严格认证: {interval_certified}')
    if needs_more_trials:
        print('提示：当前适合作为模型测试结果；正式结论需扩大临界点样本量。')
    print(f'搜索历史: {HISTORY_CSV}')
    print(f'汇总结果: {SUMMARY_JSON}')


if __name__ == '__main__':
    main()
