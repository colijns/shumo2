# 本程序及代码是在AI工具辅助下完成的
"""问题4阶段验证：在纯A参考成本上比较A/B替代方案。

该程序用于验证模型方向，不输出正式最优解。正式优化需在粗筛后对90%可行
边界追加样本，并证明所有更低成本组合不可行。
"""

import csv
from concurrent.futures import ProcessPoolExecutor
import json
import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import monte_carlo as mc  # noqa: E402


TRIALS = int(os.environ.get('SHUMO_Q4_TRIALS', '8'))
BASE_SEED = int(os.environ.get('SHUMO_Q4_BASE_SEED', '20260808'))
REFERENCE_A = int(os.environ.get('SHUMO_Q4_REFERENCE_A', '612'))
A_STEP = int(os.environ.get('SHUMO_Q4_A_STEP', '100'))
MAX_WORKERS = int(os.environ.get(
    'SHUMO_Q4_WORKERS', str(min(8, os.cpu_count() or 1))))
RESULT_DIR = os.path.join(HERE, 'results')
RESULT_JSON = os.path.join(RESULT_DIR, 'question4_stage_validation.json')
RESULT_CSV = os.path.join(RESULT_DIR, 'question4_stage_validation.csv')


def _simulate_job(job):
    index, n_a, n_b = job
    return mc.simulate_pair(
        n_a, n_b, m=TRIALS, seed=BASE_SEED + index * 100000)


def main():
    a_counts = list(range(0, REFERENCE_A + 1, A_STEP)) + [REFERENCE_A]
    pairs = mc.equal_cost_frontier(REFERENCE_A, a_counts)
    print(f'Q4阶段验证: M={TRIALS}, 参考纯A={REFERENCE_A}根, '
          f'等成本候选={len(pairs)}组', flush=True)
    jobs = [(index, n_a, n_b)
            for index, (n_a, n_b) in enumerate(pairs)]
    workers = min(MAX_WORKERS, len(jobs))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(_simulate_job, jobs))
    for row in rows:
        n_a, n_b = row['n_A'], row['n_B']
        print(f'A={n_a:4d}, B={n_b:5d}, cost={row["cost"]:.4f}, '
              f'p={row["p_hat"]:.3f}, '
              f'CI=[{row["ci_lower"]:.3f},{row["ci_upper"]:.3f}], '
              f't={row["elapsed_s"]:.2f}s', flush=True)

    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(RESULT_JSON, 'w', encoding='utf-8') as handle:
        json.dump({'stage_only': True, 'rows': rows}, handle,
                  ensure_ascii=False, indent=2)
    with open(RESULT_CSV, 'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'阶段结果: {RESULT_JSON}')


if __name__ == '__main__':
    main()
