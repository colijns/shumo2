# 本程序及代码是在AI工具辅助下完成的
# AI工具名称：DeepSeek-V4-Flash，版本 / 型号：DeepSeek-V4-Flash-0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026-07-31

"""Q4 接触阈值 δ 敏感性扫描。

两阶段设计（δ 网格默认 1.5~2.0，步长 0.1）：

- fast 阶段（近似几何，每点 2~3 min）：固定候选 = 正式独立复算 11 点 +
  纯A 605~625 窗口，M=4000，共同随机数种子 = 42+12345（与正式独立复算
  同种子，δ=1.8 结果应与 question4_verify.csv 逐点一致）。回答"临界
  数量随 δ 平移多少、含B 候选是否接近 90%"的趋势问题。
- strict 阶段（A 32 边多棱柱 + B 三级细分多面体实体夹逼，每点 37~50 min）：
  9 候选（8 含B + 619 纯A 对照），M=4000，与正式严格复核同内核
  （run_strict_mixed_audit），只跑端点（默认 1.5、2.0）认证结论是否翻转。

δ 由环境变量 SHUMO_DELTA 传入（Q1/core.py 读取），每个 δ 值用独立
subprocess 运行，避免 import 时固化的模块常量互相污染；strict 阶段复用
run_strict_mixed_audit 的按 trial 检查点机制（检查点文件按 δ 区分），
可断点续跑。

用法：
  python Q4/run_delta_sweep.py --stage fast            # 6 个 δ 全部
  python Q4/run_delta_sweep.py --stage fast --delta 1.5,1.9
  python Q4/run_delta_sweep.py --stage strict          # 默认 δ=1.5,2.0
  python Q4/run_delta_sweep.py --stage strict --delta 1.5,2.0,1.9
  python Q4/run_delta_sweep.py --point 1.7 --stage fast   # 子进程模式，勿手跑

输出：results/delta_sweep_fast_<δ>.json、delta_sweep_strict_<δ>.json/csv。
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULT_DIR = os.path.join(HERE, 'results')

DELTA_GRID = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
STRICT_DELTAS_DEFAULT = [1.5, 2.0]
M_FAST = 4000
M_STRICT = 4000
VERIFY_SEED = 42 + 12345          # 与正式独立复算同种子（question4_verify.csv）
BASE_SEED_STRICT = 20260808       # 与正式严格复核同种子
A_SEQUENCE_LENGTH = 750
CYL_SIDES = 32
BALL_SUBDIVISIONS = 3

# 正式严格复核 8 含B 候选全覆盖（含 (611,3)/(609,21)）
FAST_CANDIDATES = [
    (598, 62), (609, 5), (608, 14), (610, 4), (611, 3), (610, 12),
    (609, 21), (617, 1),
]
A_WINDOW = list(range(605, 626))  # 纯A 605~625，观察临界数量平移
# 严格复核 9 候选：8 含B 跨线/低价点 + 619 纯A 对照
STRICT_CANDIDATES = [
    (598, 62), (609, 5), (608, 14), (610, 4), (611, 3), (610, 12),
    (609, 21), (617, 1), (619, 0),
]


def env_delta(value):
    env = os.environ.copy()
    env['SHUMO_DELTA'] = str(value)
    return env


def run_fast_point(delta):
    """近似几何单点评估：11 正式候选 + 纯A 605~625，M=4000，verify 种子。"""
    os.environ['SHUMO_DELTA'] = str(delta)   # 必须先于 geometry 系列 import
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(ROOT, 'Q2'))
    sys.path.insert(0, os.path.join(ROOT, 'Q3'))
    import geometry_mix as gm                # noqa: E402
    from run_question4 import EvalStore, wilson_verdict, point_cost  # noqa: E402

    workers = int(os.environ.get('SHUMO_Q4_WORKERS',
                                 str(min(8, os.cpu_count() or 1))))
    pts = FAST_CANDIDATES + [(na, 0) for na in A_WINDOW]
    store = EvalStore(workers, na_max=625, nb_max=62, base_seed=VERIFY_SEED)
    try:
        x = store.eval(pts, M_FAST)
    finally:
        store.close()
    rows = []
    for (na, nb), xi in zip(pts, x):
        v, lo, hi = wilson_verdict(int(xi), M_FAST)
        rows.append({'N_A': na, 'N_B': nb, 'cost_yuan': point_cost((na, nb)),
                     'x': int(xi), 'm': M_FAST, 'p_hat': xi / M_FAST,
                     'ci_lower': lo, 'ci_upper': hi, 'verdict': v})
    summary = {
        'delta_nm': delta,
        'stage': 'fast_approximate',
        'seed': VERIFY_SEED,
        'trials': M_FAST,
        'candidates': rows,
        'elapsed_s': None,
    }
    path = os.path.join(RESULT_DIR, f'delta_sweep_fast_{delta}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    n_reli = sum(1 for r in rows if r['verdict'] == 'reliable')
    print(f'[fast δ={delta}] 完成：{len(rows)} 点，可靠 {n_reli}，'
          f'最低可靠成本 {min((r["cost_yuan"] for r in rows if r["verdict"] == "reliable"), default=None)}', flush=True)


def run_strict_point(delta):
    """严格实体单点复核：9 候选，M=4000，A 32 边 / B 三级细分。"""
    os.environ['SHUMO_DELTA'] = str(delta)   # 必须先于 geometry 系列 import
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(ROOT, 'Q2'))
    import run_strict_mixed_audit as rsa     # noqa: E402

    workers = int(os.environ.get('SHUMO_Q4_STRICT_WORKERS',
                                 str(min(8, os.cpu_count() or 1))))
    # 检查点/输出文件按 δ 区分，避免不同 δ 相互覆盖
    cp = os.path.join(RESULT_DIR, f'delta_sweep_strict_cp_{delta}.json')
    rsa.CHECKPOINT = cp
    rsa.CHECKPOINT_BACKUP = cp + '.bak'
    rsa.SUMMARY = os.path.join(RESULT_DIR, f'delta_sweep_strict_{delta}.json')
    rsa.CSV_RESULT = os.path.join(RESULT_DIR, f'delta_sweep_strict_{delta}.csv')

    started = time.perf_counter()
    rows = rsa.run_trials(STRICT_CANDIDATES, M_STRICT, BASE_SEED_STRICT,
                          CYL_SIDES, BALL_SUBDIVISIONS, workers,
                          A_SEQUENCE_LENGTH)
    summary = rsa.summarize(STRICT_CANDIDATES, rows, CYL_SIDES,
                            BALL_SUBDIVISIONS, BASE_SEED_STRICT,
                            A_SEQUENCE_LENGTH)
    summary['config']['delta_nm'] = delta
    summary['elapsed_s'] = time.perf_counter() - started
    rsa._write_result_files(summary)
    viol = summary['bracket_violations']
    print(f'[strict δ={delta}] 完成：9 候选，夹逼违反 {viol}，'
          f'耗时 {summary["elapsed_s"]:.1f}s', flush=True)
    for row in summary['candidate_results']:
        print(f"  ({row['N_A']:3d},{row['N_B']:3d}) "
              f"inner={row['inner_p_hat']:.4f} outer={row['outer_p_hat']:.4f} "
              f"{row['strict_verdict']} / {row['joint_strict_verdict']}", flush=True)


def sweep(stage, deltas):
    os.makedirs(RESULT_DIR, exist_ok=True)
    for d in deltas:
        out = os.path.join(RESULT_DIR, f'delta_sweep_{stage}_{d}.json')
        if os.path.exists(out):
            print(f'[{stage} δ={d}] 已存在，跳过：{out}')
            continue
        t0 = time.perf_counter()
        subprocess.run([sys.executable, os.path.abspath(__file__),
                        '--point', str(d), '--stage', stage],
                       env=env_delta(d), check=True)
        print(f'[{stage} δ={d}] 耗时 {time.perf_counter() - t0:.0f}s', flush=True)
    print(f'\n[{stage}] 汇总：')
    for d in deltas:
        path = os.path.join(RESULT_DIR, f'delta_sweep_{stage}_{d}.json')
        if not os.path.exists(path):
            print(f'  δ={d}: 未完成')
            continue
        with open(path, encoding='utf-8') as f:
            s = json.load(f)
        if stage == 'fast':
            rel = [r for r in s['candidates'] if r['verdict'] == 'reliable']
            pure_a = {r['N_A']: r for r in s['candidates'] if r['N_B'] == 0
                      and r['N_A'] >= 610}
            n90 = min((na for na, r in pure_a.items()
                       if r['p_hat'] >= 0.90), default=None)
            best = min((r['cost_yuan'] for r in rel), default=None)
            print(f'  δ={d}: 可靠 {len(rel)} 个，最低可靠成本 {best}，'
                  f'纯A 点估计 N90≈{n90}，纯A 611:'
                  f' {pure_a[611]["p_hat"]:.4f} 617:{pure_a[617]["p_hat"]:.4f} '
                  f'619:{pure_a[619]["p_hat"]:.4f}')
        else:
            jfeas = [r for r in s['candidate_results']
                     if r['joint_strict_verdict'] == 'joint_reliably_feasible']
            jins = [r for r in s['candidate_results']
                    if r['joint_strict_verdict'] == 'joint_reliably_insufficient']
            print(f'  δ={d}: 联合可靠 {len(jfeas)} 个，联合不足 {len(jins)} 个，'
                  f'待定 {len(s["candidate_results"]) - len(jfeas) - len(jins)} 个')


def main():
    ap = argparse.ArgumentParser(description='Q4 接触阈值 δ 敏感性扫描')
    ap.add_argument('--stage', choices=('fast', 'strict'), required=True)
    ap.add_argument('--delta', help='逗号分隔 δ 子集，如 1.5,1.9')
    ap.add_argument('--point', type=float, help='子进程模式：单点执行（勿手跑）')
    args = ap.parse_args()

    if args.point is not None:
        os.makedirs(RESULT_DIR, exist_ok=True)
        if args.stage == 'fast':
            run_fast_point(args.point)
        else:
            run_strict_point(args.point)
        return

    if args.delta:
        deltas = [float(x) for x in args.delta.split(',')]
    elif args.stage == 'strict':
        deltas = STRICT_DELTAS_DEFAULT
    else:
        deltas = DELTA_GRID
    sweep(args.stage, deltas)


if __name__ == '__main__':
    main()
