# 本程序及代码是在AI工具辅助下完成的
"""问题4粗筛：在多个成本水平上扫描A/B整数配比。

本程序只负责定位可行边界，不把小样本结果当作最终最优解。后续应在边界附近
追加独立样本，并以置信区间检验导通概率是否达到0.9。
"""

import csv
from concurrent.futures import ProcessPoolExecutor
import json
import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import monte_carlo as mc  # noqa: E402


TRIALS = int(os.environ.get('SHUMO_Q4_SCAN_TRIALS', '4'))
BASE_SEED = int(os.environ.get('SHUMO_Q4_SCAN_SEED', '20260809'))
BUDGETS = [float(item) for item in
           os.environ.get('SHUMO_Q4_SCAN_BUDGETS', '7,8,9').split(',')]
A_STEP = int(os.environ.get('SHUMO_Q4_SCAN_A_STEP', '100'))
MAX_WORKERS = int(os.environ.get(
    'SHUMO_Q4_WORKERS', str(min(8, os.cpu_count() or 1))))
RESULT_DIR = os.path.join(HERE, 'results')
RESULT_JSON = os.path.join(RESULT_DIR, 'question4_cost_scan.json')
RESULT_CSV = os.path.join(RESULT_DIR, 'question4_cost_scan.csv')


def _simulate_job(job):
    index, budget, n_a, n_b = job
    row = mc.simulate_pair(
        n_a, n_b, m=TRIALS, seed=BASE_SEED + index * 100000)
    row['budget'] = budget
    return row


def main():
    jobs = []
    index = 0
    for budget in sorted(set(BUDGETS)):
        max_a = int(budget // mc.geo.COST_PER_A)
        a_counts = list(range(0, max_a + 1, A_STEP)) + [max_a]
        for n_a, n_b in mc.cost_frontier(budget, a_counts):
            jobs.append((index, budget, n_a, n_b))
            index += 1

    print(f'Q4多成本粗筛: M={TRIALS}, 候选={len(jobs)}组, '
          f'成本={sorted(set(BUDGETS))}', flush=True)
    with ProcessPoolExecutor(
            max_workers=min(MAX_WORKERS, len(jobs))) as executor:
        rows = list(executor.map(_simulate_job, jobs))
    rows.sort(key=lambda row: (row['budget'], row['n_A']))

    for row in rows:
        print(f'budget={row["budget"]:.2f}, A={row["n_A"]:4d}, '
              f'B={row["n_B"]:5d}, cost={row["cost"]:.4f}, '
              f'p={row["p_hat"]:.3f}, '
              f'CI=[{row["ci_lower"]:.3f},{row["ci_upper"]:.3f}]',
              flush=True)

    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(RESULT_JSON, 'w', encoding='utf-8') as handle:
        json.dump({'stage_only': True, 'rows': rows}, handle,
                  ensure_ascii=False, indent=2)
    with open(RESULT_CSV, 'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'粗筛结果: {RESULT_JSON}', flush=True)


if __name__ == '__main__':
    main()
