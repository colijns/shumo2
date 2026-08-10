# 本程序及代码是在AI工具辅助下完成的
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31

import csv
from concurrent.futures import ProcessPoolExecutor
import json
import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import monte_carlo as mc


TRIALS = int(os.environ.get('SHUMO_Q4_REFINE_TRIALS', '20'))
BASE_SEED = int(os.environ.get('SHUMO_Q4_REFINE_SEED', '20261808'))
BUDGETS = [float(item) for item in
           os.environ.get('SHUMO_Q4_REFINE_BUDGETS', '8,8.5,9').split(',')]
A_COUNTS = [int(item) for item in
            os.environ.get('SHUMO_Q4_REFINE_A_COUNTS',
                           '400,450,500,550,600').split(',')]
EXPLICIT_PAIRS = os.environ.get('SHUMO_Q4_REFINE_PAIRS', '').strip()
MAX_WORKERS = int(os.environ.get(
    'SHUMO_Q4_WORKERS', str(min(8, os.cpu_count() or 1))))
RESULT_DIR = os.path.join(HERE, 'results')
RESULT_STEM = os.environ.get('SHUMO_Q4_REFINE_RESULT_STEM',
                             'question4_refined_scan')
RESULT_JSON = os.path.join(RESULT_DIR, RESULT_STEM + '.json')
RESULT_CSV = os.path.join(RESULT_DIR, RESULT_STEM + '.csv')


def _trial(seed):
    return mc.common_random_trial(PAIRS, seed)


def _candidate_pairs():
    if EXPLICIT_PAIRS:
        pairs = []
        for item in EXPLICIT_PAIRS.split(','):
            n_a, n_b = (int(value) for value in item.split(':'))
            pairs.append((mc.total_cost(n_a, n_b), (n_a, n_b)))
        return pairs
    tagged = []
    for budget in sorted(set(BUDGETS)):
        max_a = int(budget // mc.geo.COST_PER_A)
        counts = sorted(set([v for v in A_COUNTS if 0 <= v <= max_a]
                            + [max_a]))
        for pair in mc.cost_frontier(budget, counts):
            tagged.append((budget, pair))
    return tagged


TAGGED_PAIRS = _candidate_pairs()
PAIRS = [pair for _, pair in TAGGED_PAIRS]


def main():
    print(f'Q4共同随机数精筛: M={TRIALS}, 候选={len(PAIRS)}组', flush=True)
    seeds = [BASE_SEED + index for index in range(TRIALS)]
    with ProcessPoolExecutor(
            max_workers=min(MAX_WORKERS, TRIALS)) as executor:
        trials = list(executor.map(_trial, seeds))
    rows = mc.aggregate_common_trials(PAIRS, trials)
    for row, (budget, _) in zip(rows, TAGGED_PAIRS):
        row['budget'] = budget
        print(f'budget={budget:.2f}, A={row["n_A"]:4d}, '
              f'B={row["n_B"]:5d}, p={row["p_hat"]:.3f}, '
              f'CI=[{row["ci_lower"]:.3f},{row["ci_upper"]:.3f}]',
              flush=True)

    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(RESULT_JSON, 'w', encoding='utf-8') as handle:
        json.dump({'stage_only': True, 'common_random_numbers': True,
                   'rows': rows}, handle, ensure_ascii=False, indent=2)
    with open(RESULT_CSV, 'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'精筛结果: {RESULT_JSON}', flush=True)


if __name__ == '__main__':
    main()
