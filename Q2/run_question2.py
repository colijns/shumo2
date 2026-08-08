# 本程序及代码是在AI工具辅助下完成的
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31

"""问题2 入口：4 个体积分数全量蒙特卡洛，终端表 + results/question2_result.csv。

体积分数：0.50% / 0.60% / 0.70% / 1.00%，每点 M=2000，seed=BASE_SEED+i。
结果含 95% Wilson 置信区间，供论文表格与配图使用。
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import monte_carlo as mc  # noqa: E402

PHIS = [0.005, 0.006, 0.007, 0.01]
M = 2000
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'question2_result.csv')


def fmt_pct(v):
    return f'{v * 100:.2f}%'


def _run_one_phi(args):
    """进程池 worker：单个 φ 全量模拟（独立 seed，与单进程逐位一致）。"""
    phi, m, seed = args
    return mc.simulate_phi(phi, m=m, seed=seed)


def main():
    t_total0 = time.perf_counter()
    rows = []
    jobs = [(phi, M, mc.BASE_SEED + i) for i, phi in enumerate(PHIS)]
    with ProcessPoolExecutor(max_workers=len(PHIS)) as ex:
        for i, res in enumerate(ex.map(_run_one_phi, jobs)):
            print(f'[φ = {fmt_pct(res["phi"])}] N_A = {res["n_A"]}, '
                  f'导通 {res["x"]}/{res["m"]}, p̂ = {res["p_hat"]:.4f}, '
                  f'CI = [{res["ci_lower"]:.4f}, {res["ci_upper"]:.4f}], '
                  f'耗时 {res["elapsed_s"]:.1f}s', flush=True)
            rows.append(res)
    t_total = time.perf_counter() - t_total0

    # 终端表
    print('\n===== 问题2 结果：介质A体积分数 vs 导通概率 =====')
    print(f'{"φ":>8} {"N_A":>6} {"M":>6} {"x":>6} {"p̂":>10} '
          f'{"CI下界":>10} {"CI上界":>10} {"平均片段":>8} {"耗时(s)":>8}')
    for r in rows:
        print(f'{fmt_pct(r["phi"]):>8} {r["n_A"]:>6} {r["m"]:>6} {r["x"]:>6} '
              f'{r["p_hat"]:>10.4f} {r["ci_lower"]:>10.4f} {r["ci_upper"]:>10.4f} '
              f'{r["mean_fragments"]:>8.2f} {r["elapsed_s"]:>8.1f}')

    print(f'总耗时（4 φ 并行）: {t_total:.1f}s = {t_total / 60:.1f}min')

    # 写 CSV
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    header = ['phi', 'n_A', 'm', 'x', 'p_hat', 'ci_lower', 'ci_upper',
              'mean_fragments', 'mean_edges', 'mean_gjk', 'elapsed_s']
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        f.write(','.join(header) + '\n')
        for r in rows:
            f.write(','.join(f'{r[k]:.10g}' if isinstance(r[k], float) else str(r[k])
                             for k in header) + '\n')
    print(f'\n结果已写入 {OUT_CSV}')


if __name__ == '__main__':
    main()
