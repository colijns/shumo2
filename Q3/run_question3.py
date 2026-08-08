# 本程序及代码是在AI工具辅助下完成的
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31

"""问题3 入口：首次导通数量蒙特卡洛，90% 临界数量/体积分数估计。

口径（docs/问题3_新版.md）：
- N_max 根固定顺序完整介质 → 增量早停记录 N_c（嵌套共同样本）；
- P̂(N) = 经验 CDF，天然单调；N̂90 = min{N: P̂(N) ≥ 0.90}；
- N_safe = min{N: Wilson 95% 下界 ≥ 0.90}（保守可靠约束）；
- Bootstrap 重抽样得临界体积分数 95% 区间（宽度 > 0.01pp 则提示追加）；
- 敏感性口径（--same-source，假设一：同源片段自动电连续）对照；
- --verify 用未参与搜索的新种子独立复算候选点。

环境变量（与 Q2 同风格）：
SHUMO_Q3_TRIALS（默认 2000）、SHUMO_Q3_WORKERS（默认 min(8, cpu)）、
SHUMO_Q3_BASE_SEED（默认 42）、SHUMO_Q3_NMAX（默认 900）。
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Q2'))
import first_passage as fp  # noqa: E402

from monte_carlo import wilson_ci, Z_WILSON  # noqa: E402

N_MAX_DEFAULT = 900
M_DEFAULT = 2000
B_BOOTSTRAP = 2000          # Bootstrap 重抽样次数
BOOT_ALPHA = 0.05           # 95% 区间
P_TARGET = 0.90             # 题目导通概率要求
WILSON_MARGIN_PP = 0.01     # Bootstrap 区间宽度阈值（百分点）
VERIFY_SEED_OFFSET = 12345  # 独立复算种子偏移

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
CSV_MAIN = os.path.join(OUT_DIR, 'question3_result.csv')
CSV_CURVE = os.path.join(OUT_DIR, 'question3_curve.csv')
CSV_VERIFY = os.path.join(OUT_DIR, 'question3_verify.csv')
# 敏感性口径（假设一）结果独立落盘，避免覆盖主结果
CSV_MAIN_H1 = os.path.join(OUT_DIR, 'question3_result_hypothesis1.csv')
CSV_CURVE_H1 = os.path.join(OUT_DIR, 'question3_curve_hypothesis1.csv')


def fmt_pct(v):
    """百分比输出：保留百分号下两位小数。"""
    return f'{v * 100:.2f}%'


def empirical_cdf(contacts, n_max, m):
    """经验 CDF P̂(N) 与逐 N Wilson 区间。

    contacts: M 个 N_c（None = N_max 内未导通，视为 > n_max）。
    返回 (N 数组, p_hat 数组, lo 数组, hi 数组, x 数组)。
    """
    arr = np.array([v if v is not None else n_max + 1 for v in contacts])
    Ns = np.arange(1, n_max + 1, dtype=int)
    x = np.array([int(np.sum(arr <= n)) for n in Ns])
    p_hat = x / m
    lo = np.empty_like(p_hat)
    hi = np.empty_like(p_hat)
    for i, xi in enumerate(x):
        _, lo[i], hi[i] = wilson_ci(xi, m)
    return Ns, p_hat, lo, hi, x


def n_90_estimates(contacts, n_max, m):
    """点估计 N̂90 与保守临界 N_safe。返回 (n_hat, n_safe) 或 None。"""
    Ns, p_hat, lo, hi, _ = empirical_cdf(contacts, n_max, m)
    hit = np.nonzero(p_hat >= P_TARGET)[0]
    if len(hit) == 0:
        return None, None
    n_hat = int(Ns[hit[0]])
    safe = np.nonzero(lo >= P_TARGET)[0]
    n_safe = int(Ns[safe[0]]) if len(safe) else None
    return n_hat, n_safe


def bootstrap_ci(contacts, n_max, m, b=B_BOOTSTRAP):
    """Bootstrap 重抽样：90% 分位数（N̂90）分布 → 95% CI（换算体积分数）。

    返回 (ci_lo_pp, ci_hi_pp, width_pp)。宽度单位：百分点。
    """
    arr = np.array([v if v is not None else n_max + 1 for v in contacts])
    rng = np.random.default_rng(999)   # Bootstrap 自身固定种子（可复现）
    qs = np.empty(b)
    for k in range(b):
        sample = arr[rng.integers(0, m, size=m)]
        qs[k] = np.quantile(sample, P_TARGET, method='higher')
    lo = np.percentile(qs, 100 * BOOT_ALPHA / 2)
    hi = np.percentile(qs, 100 * (1 - BOOT_ALPHA / 2))
    phi = fp.geo.V_A / fp.geo.V_BOX * 100.0   # 单根圆柱对应的百分点数
    return lo * phi, hi * phi, (hi - lo) * phi


def crossing_ratio_check(seed, n_samples=2000):
    """全量跨壁比例验证：与理论基准（约 60.8%）对照。

    §7 精确判定：|c_k| + e_k > L/2 任一轴成立即跨壁。
    返回 {'n_cross', 'n_cyl', 'ratio'}。
    """
    rng = np.random.default_rng(seed)
    c, u, h = fp.geo.generate_cylinders(n_samples, rng)
    n_cross, n_cyl = fp.crossing_count(c, u, h)
    return {'n_cross': n_cross, 'n_cyl': n_cyl, 'ratio': n_cross / n_cyl}


def _run_trials(args):
    """进程池 worker：跑一批试验，返回 N_c 列表 + 跨壁统计。"""
    n_max, m, seed, same_source = args
    contacts, stats = fp.simulate_trials(n_max, m=m, seed=seed,
                                         same_source=same_source)
    return contacts, stats


def run_all(n_max, m, base_seed, workers, same_source):
    """并行跑 M 次试验，聚合样本与统计。"""
    if m <= 0 or workers <= 0:
        raise ValueError('M 与 WORKERS 必须为正整数')
    batch = max(1, min(50, m))
    jobs = []
    remaining = m
    batch_index = 0
    while remaining:
        bm = min(batch, remaining)
        seed = base_seed + batch_index * 100_000
        jobs.append((n_max, bm, seed, same_source))
        remaining -= bm
        batch_index += 1
    workers = min(workers, len(jobs))
    print(f'并行设置：workers={workers}, batch_size={batch}, jobs={len(jobs)}',
          flush=True)
    contacts_all, stats_all = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for contacts, stats in ex.map(_run_trials, jobs):
            contacts_all.extend(contacts)
            stats_all.append(stats)
    n_never = sum(s['n_never'] for s in stats_all)
    n_cross = sum(s['n_cross'] for s in stats_all)
    n_same = sum(s['n_same_contacts'] for s in stats_all)
    stats = {'n_never': n_never, 'n_cross': n_cross,
             'n_same_contacts': n_same}
    return contacts_all, stats


def main():
    ap = argparse.ArgumentParser(description='问题3 首次导通数量蒙特卡洛')
    ap.add_argument('--same-source', action='store_true',
                    help='敏感性口径：同源片段自动电连续（假设一）')
    ap.add_argument('--verify', action='store_true',
                    help='候选点独立复算（新种子）')
    ap.add_argument('--smoke', action='store_true',
                    help='小样本冒烟（M=30）')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    n_max = int(os.environ.get('SHUMO_Q3_NMAX', str(N_MAX_DEFAULT)))
    m = int(os.environ.get('SHUMO_Q3_TRIALS', str(M_DEFAULT)))
    base_seed = int(os.environ.get('SHUMO_Q3_BASE_SEED', '42'))
    workers = int(os.environ.get(
        'SHUMO_Q3_WORKERS', str(min(8, os.cpu_count() or 1))))
    if args.smoke:
        m = 30
    print(f'问题3：N_max={n_max}, M={m}, '
          f'口径={"假设一(同源电连续)" if args.same_source else "假设二(片段独立)"}',
          flush=True)
    t_total0 = time.perf_counter()

    contacts, stats = run_all(n_max, m, base_seed, workers, args.same_source)
    t_run = time.perf_counter() - t_total0

    Ns, p_hat, lo, hi, x = empirical_cdf(contacts, n_max, m)
    n_hat, n_safe = n_90_estimates(contacts, n_max, m)
    phi_per_cyl = fp.geo.V_A / fp.geo.V_BOX * 100.0   # 每根圆柱百分点
    if n_hat is None:
        print(f'警告：N_max={n_max} 内导通概率未达 90%（p̂={p_hat[-1]:.4f}），'
              '需提高 N_max')
        n_safe = None

    boot_lo, boot_hi, boot_w = bootstrap_ci(contacts, n_max, m)

    cross_ratio = crossing_ratio_check(base_seed)
    never_frac = stats['n_never'] / m

    # 主结果汇总
    res = {
        '口径': '假设二(片段独立)' if not args.same_source else '假设一(同源电连续)',
        'n_max': n_max,
        'm': m,
        'n_never': stats['n_never'],
        'never_frac': never_frac,
        'n_hat_90': n_hat,
        'phi_hat_90_pp': (n_hat * phi_per_cyl) if n_hat else None,
        'n_safe': n_safe,
        'phi_safe_pp': (n_safe * phi_per_cyl) if n_safe else None,
        'x_at_n_hat': int(x[n_hat - 1]) if n_hat else None,
        'p_at_n_hat': float(p_hat[n_hat - 1]) if n_hat else None,
        'ci_at_n_hat': (float(lo[n_hat - 1]), float(hi[n_hat - 1]))
                       if n_hat else None,
        'bootstrap_ci_pp': (boot_lo, boot_hi),
        'bootstrap_width_pp': boot_w,
        'cross_ratio': cross_ratio['ratio'],
        'n_cross': cross_ratio['n_cross'],
        'elapsed_s': time.perf_counter() - t_total0,
    }

    # 终端主表
    print('\n===== 问题3 结果 =====')
    print(f'口径              : {res["口径"]}')
    print(f'N_max / M         : {n_max} / {m}')
    print(f'未导通试验        : {stats["n_never"]} ({never_frac * 100:.2f}%)')
    print(f'N90_hat (点估计)  : {n_hat}  ->  phi_hat* = {fmt_pct(n_hat * phi_per_cyl / 100)}'
          f'（单根步长 {phi_per_cyl:.4f}pp）')
    if n_hat:
        print(f'  P_hat(N90_hat)   : {p_hat[n_hat - 1]:.4f}, '
              f'Wilson CI = [{lo[n_hat - 1]:.4f}, {hi[n_hat - 1]:.4f}]')
    if n_safe:
        print(f'Nsafe (下界>=90%) : {n_safe}  ->  phi_safe = '
              f'{fmt_pct(n_safe * phi_per_cyl / 100)}')
    else:
        print('Nsafe (下界>=90%) : 未达到（需提高 N_max 或增加试验）')
    print(f'Bootstrap 95% 区间: [{fmt_pct(boot_lo / 100)}, '
          f'{fmt_pct(boot_hi / 100)}]（宽 {boot_w:.4f}pp）')
    print(f'跨壁比例          : {cross_ratio["ratio"]:.4f} '
          f'({cross_ratio["n_cross"]}/{cross_ratio["n_cyl"]}，基准约 60.8%）')
    print(f'总耗时            : {time.perf_counter() - t_total0:.1f}s '
          f'(模拟 {t_run:.1f}s)')

    # 主结果 CSV（敏感性口径写入独立文件）
    csv_main = CSV_MAIN_H1 if args.same_source else CSV_MAIN
    with open(csv_main, 'w', encoding='utf-8') as f:
        import csv
        w = csv.DictWriter(f, fieldnames=list(res.keys()))
        w.writeheader()
        w.writerow(res)

    # 曲线 CSV（供绘图）
    csv_curve = CSV_CURVE_H1 if args.same_source else CSV_CURVE
    import csv as _csv
    with open(csv_curve, 'w', newline='', encoding='utf-8') as f:
        w = _csv.writer(f)
        w.writerow(['N', 'phi_pp', 'p_hat', 'ci_lower', 'ci_upper', 'x'])
        for i, n in enumerate(Ns):
            w.writerow([n, f'{n * phi_per_cyl:.6f}', f'{p_hat[i]:.6f}',
                        f'{lo[i]:.6f}', f'{hi[i]:.6f}', x[i]])

    # 独立复算（新种子，未参与搜索）
    if args.verify and not args.same_source:
        if n_hat is None or n_safe is None:
            print('警告：候选点缺失，跳过独立复算')
        else:
            verify_seed = base_seed + VERIFY_SEED_OFFSET
            v_contacts, v_stats = run_all(n_max, m, verify_seed, workers, False)
            v_Ns, v_p, v_lo, v_hi, v_x = empirical_cdf(v_contacts, n_max, m)
            rows = []
            for label, n in (('N90_hat', n_hat), ('N90_hat-1', n_hat - 1),
                             ('N90_hat+1', n_hat + 1), ('Nsafe', n_safe)):
                i = n - 1
                rows.append({'candidate': label, 'N': n,
                             'p_hat': v_p[i], 'ci_lower': v_lo[i],
                             'ci_upper': v_hi[i], 'x': v_x[i]})
            print('\n===== 独立复算（新种子）=====')
            for r in rows:
                ok = '可靠' if r['ci_lower'] >= P_TARGET else (
                    '不足' if r['ci_upper'] < P_TARGET else '区间跨线')
                print(f"  {r['candidate']:8s} N={r['N']}: p_hat={r['p_hat']:.4f} "
                      f"CI=[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}] {ok}")
            with open(CSV_VERIFY, 'w', newline='', encoding='utf-8') as f:
                w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            # 单调性检查：更高数量不下降
            mono_ok = np.all(np.diff(v_p) >= -1e-12)
            print(f'  单调性（更高 N 不降）: {mono_ok}')

    print(f'\n结果已写入 {csv_main} / {csv_curve}')


if __name__ == '__main__':
    main()
