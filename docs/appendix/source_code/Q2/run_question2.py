# 本程序及代码是在AI工具辅助下完成的
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31

"""问题2 入口：分批并行蒙特卡洛，终端表 + results/question2_result.csv。

体积分数：0.50% / 0.60% / 0.70% / 1.00%，每点默认 M=2000。
全部“体积分数×试验批次”进入同一个进程池，避免外层与内层嵌套并行。
可用环境变量 SHUMO_Q2_TRIALS、SHUMO_Q2_BATCH_SIZE、SHUMO_Q2_WORKERS
调整试验数、批大小和进程数。
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
M = int(os.environ.get('SHUMO_Q2_TRIALS', '2000'))
BATCH_SIZE = int(os.environ.get('SHUMO_Q2_BATCH_SIZE', '50'))
MAX_WORKERS = int(os.environ.get(
    'SHUMO_Q2_WORKERS', str(min(8, os.cpu_count() or 1))))
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'question2_result.csv')


def fmt_pct(v):
    return f'{v * 100:.2f}%'


def _run_one_batch(args):
    """进程池 worker：完成一个 φ 的一小批独立试验。"""
    phi_index, batch_index, phi, m, seed = args
    return phi_index, batch_index, mc.simulate_phi(phi, m=m, seed=seed)


def _merge_batches(phi, batches):
    """合并同一 φ 的批次统计量并重新计算 Wilson 区间。"""
    total_m = sum(r['m'] for r in batches)
    total_x = sum(r['x'] for r in batches)
    p_hat, lo, hi = mc.wilson_ci(total_x, total_m)
    return {
        'phi': phi,
        'n_A': batches[0]['n_A'],
        'm': total_m,
        'x': total_x,
        'p_hat': p_hat,
        'ci_lower': lo,
        'ci_upper': hi,
        'mean_fragments': sum(r['mean_fragments'] * r['m'] for r in batches) / total_m,
        'mean_edges': sum(r['mean_edges'] * r['m'] for r in batches) / total_m,
        'mean_gjk': sum(r['mean_gjk'] * r['m'] for r in batches) / total_m,
        # 各 worker 计算时间之和，用于比较计算量；总墙钟时间在 main 中单独输出。
        'elapsed_s': sum(r['elapsed_s'] for r in batches),
    }


def main():
    if M <= 0 or BATCH_SIZE <= 0 or MAX_WORKERS <= 0:
        raise ValueError('M、BATCH_SIZE、MAX_WORKERS 必须为正整数')
    t_total0 = time.perf_counter()
    jobs = []
    for phi_index, phi in enumerate(PHIS):
        remaining = M
        batch_index = 0
        while remaining:
            batch_m = min(BATCH_SIZE, remaining)
            # 每个批次独立且确定的 seed，不受进程调度顺序影响。
            seed = mc.BASE_SEED + phi_index * 100_000 + batch_index
            jobs.append((phi_index, batch_index, phi, batch_m, seed))
            remaining -= batch_m
            batch_index += 1

    grouped = [[] for _ in PHIS]
    workers = min(MAX_WORKERS, len(jobs))
    print(f'并行设置：workers={workers}, batch_size={BATCH_SIZE}, jobs={len(jobs)}',
          flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for phi_index, batch_index, res in ex.map(_run_one_batch, jobs):
            grouped[phi_index].append((batch_index, res))

    rows = []
    for i, phi in enumerate(PHIS):
        batches = [r for _, r in sorted(grouped[i])]
        res = _merge_batches(phi, batches)
        print(f'[φ = {fmt_pct(res["phi"])}] N_A = {res["n_A"]}, '
              f'导通 {res["x"]}/{res["m"]}, p_hat = {res["p_hat"]:.4f}, '
              f'CI = [{res["ci_lower"]:.4f}, {res["ci_upper"]:.4f}]',
              flush=True)
        rows.append(res)
    t_total = time.perf_counter() - t_total0

    # 终端表
    print('\n===== 问题2 结果：介质A体积分数 vs 导通概率 =====')
    print(f'{"φ":>8} {"N_A":>6} {"M":>6} {"x":>6} {"p_hat":>10} '
          f'{"CI下界":>10} {"CI上界":>10} {"平均片段":>8} {"耗时(s)":>8}')
    for r in rows:
        print(f'{fmt_pct(r["phi"]):>8} {r["n_A"]:>6} {r["m"]:>6} {r["x"]:>6} '
              f'{r["p_hat"]:>10.4f} {r["ci_lower"]:>10.4f} {r["ci_upper"]:>10.4f} '
              f'{r["mean_fragments"]:>8.2f} {r["elapsed_s"]:>8.1f}')

    print(f'总墙钟耗时（分批并行）: {t_total:.1f}s = {t_total / 60:.1f}min')

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
