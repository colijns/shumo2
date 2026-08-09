# 本绘图程序在AI工具辅助下完成
# AI工具名称：DeepSeek-V4-Flash（Claude Code）
# 版本：v4-flash（2026-08 使用）
# 开发机构：杭州深度求索人工智能基础技术研究有限公司
# 使用日期：2026-08-09
# 人工修改记录：见 docs/appendix/interaction_logs/edit_trace.md

"""问题4 论文配图：成本-导通概率分布 + 90% 可行边界（png+pdf 双格式）。

- 图1 主图：全部搜索评估点的成本-导通概率散点图，p̂≥0.85 候选按 N_B 着色，
  叠加独立复算可靠/跨线候选与关键点标注（替换原 (N_A,N_B) 热图，原图在
  N_B 0~5416 全域插值产生大面积假色块，读者无法读出低成本区边界）；
- 图2 边界图：最低可行层 (N_A, N_B) 边界曲线 + Wilson 判定着色（可靠/不足/跨线）。

读 Q4/results/*.csv，输出 Q4/figures/（png+pdf，300dpi，双格式），并复制入
docs/appendix/figures/ 图池（competition-record 步骤 2+4）。
"""

import csv
import os
import shutil
import sys
import tempfile
from collections import defaultdict

os.environ.setdefault(
    'MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'shumo2-matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei',
                                   'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

_BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_BASE, 'results')
FIG_DIR = os.path.join(_BASE, 'figures')
APPENDIX_FIG_DIR = os.path.join(
    os.path.dirname(_BASE), 'docs', 'appendix', 'figures')
CSV_POINTS = os.path.join(OUT_DIR, 'question4_points.csv')
CSV_BOUNDARY = os.path.join(OUT_DIR, 'question4_boundary.csv')
CSV_RESULT = os.path.join(OUT_DIR, 'question4_result.csv')
CSV_VERIFY = os.path.join(OUT_DIR, 'question4_verify.csv')

P_TARGET = 0.90


def save_pair(fig, stem):
    """competition-record 步骤 2+4：png/pdf 双格式保存 + 复制入图池。"""
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(APPENDIX_FIG_DIR, exist_ok=True)
    base = os.path.join(FIG_DIR, stem)
    for ext in ('png', 'pdf'):
        fig.savefig(base + '.' + ext, dpi=300, bbox_inches='tight')
        shutil.copy2(base + '.' + ext,
                     os.path.join(APPENDIX_FIG_DIR, stem + '.' + ext))
    plt.close(fig)
    print(f'已保存并入池：{stem}.png / .pdf（dpi=300）')


def load_points(path):
    """读 points.csv → (na, nb, p_hat) 数组。"""
    na, nb, p = [], [], []
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            na.append(int(row['N_A']))
            nb.append(int(row['N_B']))
            p.append(float(row['p_hat']))
    return np.asarray(na), np.asarray(nb), np.asarray(p)


def load_result(path):
    with open(path, encoding='utf-8') as f:
        return next(csv.DictReader(f))


def load_verify(path):
    """读 verify.csv → (na, nb, cost, p, verdict) 行列表（独立种子复算）。"""
    rows = []
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append({'na': int(r['N_A']), 'nb': int(r['N_B']),
                         'cost': float(r['cost_元']), 'p': float(r['p_hat']),
                         'v': r['verdict']})
    return rows


def load_boundary(path):
    """读 boundary.csv → (na, nb, cost, p_hat, verdict) 行列表。

    verdict 优先取主种子判定列（新列名 main_seed_verdict），兼容旧文件
    verify_verdict 列名。"""
    rows = []
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append({'na': int(r['N_A']), 'nb': int(r['N_B']),
                         'cost': float(r['cost_元']),
                         'p': float(r['p_hat']),
                         'v': (r.get('main_seed_verdict', '')
                               or r.get('verify_verdict', ''))})
    return rows


def plot_cost_probability():
    """图1：成本-导通概率散点图（搜索点 + 高概率前景 + 独立复算关键候选标注）。

    横轴为总成本 C = c_A·N_A + c_B·N_B，纵轴为导通概率点估计。搜索评估点
    （约 1 万点集中于 8.92~9.08 元）以成本细箱分位带呈现：10~90 与 25~75
    分位填充带 + 中位数线，展示概率分布随成本上升的形态且零重叠；p̂≥0.85
    的 240 个候选按 N_A 聚为 54 个候选簇（x 带固定 seed 抖动、点大小=簇内
    组合数、颜色=簇内最小 N_B），叠加独立种子复算的关键候选（星=可靠 /
    叉=跨线）。
    """
    na, nb, p = load_points(CSV_POINTS)
    res = load_result(CSV_RESULT)
    c_a, c_b = float(res['c_A_元']), float(res['c_B_元'])
    cost = c_a * na + c_b * nb
    na_star, nb_star = int(res['N_A_star']), int(res['N_B_star'])
    c_star = float(res['C_star_元'])

    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    # 搜索评估点：成本细箱分位带（10~90/25~75 分位填充 + 中位数线），
    # 避免逐点绘制重叠；窗口 8.88~9.16 内约 1 万点、9.16 以上搜索未覆盖
    mask = (cost >= 8.88) & (cost <= 9.16)
    edges = np.linspace(8.88, 9.16, 30)
    xc, q10, q25, q50, q75, q90 = [], [], [], [], [], []
    for i in range(len(edges) - 1):
        m = mask & (cost >= edges[i]) & (cost < edges[i + 1])
        if m.sum() < 5:
            continue
        q = np.percentile(p[m], [10, 25, 50, 75, 90])
        xc.append(0.5 * (edges[i] + edges[i + 1]))
        q10.append(q[0]); q25.append(q[1]); q50.append(q[2])
        q75.append(q[3]); q90.append(q[4])
    ax.fill_between(xc, q10, q90, color='#C8C8C8', alpha=0.55,
                    label='搜索评估点 $\\hat P$ 10~90 分位带（$M=100$）')
    ax.fill_between(xc, q25, q75, color='#8E8E8E', alpha=0.55,
                    label='$\\hat P$ 25~75 分位带')
    ax.plot(xc, q50, color='#444444', lw=1.4, label='$\\hat P$ 中位数')
    # 前景：p̂ ≥ 0.85 的候选按 N_A 聚簇——同 N_A 的达标组合合并为一个代表点：
    # x=簇内最小成本（加固定 seed 抖动防重叠）、y=簇内最大 p̂、
    # 颜色=簇内最小 N_B、点大小=簇内组合数
    groups = defaultdict(list)
    for na_i, nb_i, p_i in zip(na, nb, p):
        if p_i >= 0.85:
            groups[na_i].append((nb_i, p_i, c_a * na_i + c_b * nb_i))
    gx, gy, gc, gs = [], [], [], []
    for na_k in sorted(groups):
        g = groups[na_k]
        gx.append(min(x[2] for x in g))
        gy.append(max(x[1] for x in g))
        gc.append(min(x[0] for x in g))
        gs.append(len(g))
    jit = np.random.default_rng(42).normal(0, 0.0015, size=len(gx))
    sc = ax.scatter(np.asarray(gx) + jit, gy, s=[11 + 5 * c for c in gs],
                    c=gc, cmap='viridis', vmin=0, vmax=62, linewidths=0.4,
                    edgecolors='#333333', label='$\\hat P\\geq0.85$ 候选簇'
                    '（同 $N_A$ 聚合，点大小=簇内组合数）')
    cb = fig.colorbar(sc, ax=ax, label='簇内最小介质B 数量 $N_B$（个）')
    # 独立种子复算关键候选（verify.csv）：星=可靠可行，叉=跨线
    drawn = set()
    for r in load_verify(CSV_VERIFY):
        if r['v'] == 'reliable':
            lab = '独立复算可靠（CI 下界 ≥ 0.90）'
            if lab in drawn:
                lab = None
            else:
                drawn.add(lab)
            ax.scatter([r['cost']], [r['p']], marker='*', s=160,
                       c='#55A868', edgecolors='#111111', linewidths=0.9,
                       label=lab)
        else:
            ax.scatter([r['cost']], [r['p']], marker='x', s=46,
                       c='#4C72B0', linewidths=1.4, zorder=3)
    # 0.90 目标线
    ax.axhline(P_TARGET, color='#C44E52', ls='--', lw=1.5,
               label='目标概率 $P=0.90$')
    # 关键候选文字标注（白底框压灰点，窗口内重摆）
    bbox_kw = dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.9)
    ax.annotate('搜索定位 $(598,62)$：\n8.981 元，$M=100$ 点估计 0.90，不可靠',
                xy=(c_a * 598 + c_b * 62, 0.90), xytext=(8.90, 0.97),
                fontsize=8.5, color='#333333', bbox=bbox_kw,
                arrowprops=dict(arrowstyle='->', color='#333333', lw=0.9))
    ax.annotate('$(608,14)$ 9.049 元：\n独立复算跨线',
                xy=(c_a * 608 + c_b * 14, 0.8925), xytext=(8.91, 0.77),
                fontsize=8.5, color='#4C72B0', bbox=bbox_kw,
                arrowprops=dict(arrowstyle='->', color='#4C72B0', lw=0.9))
    ax.annotate(f'独立复算可靠 $({na_star},{nb_star})$ {c_star:.4f} 元',
                xy=(c_star, 0.91075), xytext=(9.04, 0.99),
                fontsize=8.5, color='#2e7d32', bbox=bbox_kw,
                arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=0.9))
    ax.set_xlabel('总成本 $C=c_A N_A+c_B N_B$（元）')
    ax.set_ylabel('导通概率点估计 $\\hat P$')
    # 分位带图：x 8.88~9.24（9.16 以上搜索未覆盖），y 覆盖带与候选全貌
    ax.set_xlim(8.88, 9.24)
    ax.set_ylim(0.45, 1.02)
    ax.legend(loc='lower right', fontsize=8.5, framealpha=0.95)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_pair(fig, '问题4_混合导通热图')
    print(f'图1 成本-概率：{len(na)} 评估点，可靠标注 ({na_star},{nb_star})')


def plot_boundary():
    """图2：主种子验证带边界图——每 N_A 的 N_B 需求带 + 下包络 + 跨线/可靠点。

    原图为 3048 个点逐点散点（其中 3040 个 insufficient 全红，无法读出
    边界结构）；重设计为：x=N_A（0~617）、y=N_B（0~5500），每 N_A 的
    1~5 个验证点合成为 min~max 竖带（验证带，宽度约 56 个 B），叠加
    下包络线（该 N_A 最低验证点，仍不足 0.90），跨线 7 点蓝叉、可靠
    (617,0) 绿星，标注关键段。叙事：B 需求随 N_A 单调下降，608 段后
    进入小 B 区出现跨线，617 纯 A 可靠。
    """
    rows = load_boundary(CSV_BOUNDARY)
    res = load_result(CSV_RESULT)
    na_star, nb_star = int(res['N_A_star']), int(res['N_B_star'])
    c_star = float(res['C_star_元'])

    per = defaultdict(list)
    for r in rows:
        per[r['na']].append((r['nb'], r['v']))
    nas = sorted(per)
    lo = [min(x[0] for x in per[k]) for k in nas]
    hi = [max(x[0] for x in per[k]) for k in nas]

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    # 验证带：每 N_A 的 N_B 搜索范围（min~max）
    ax.fill_between(nas, lo, hi, color='#C8C8C8', alpha=0.55,
                    label='主种子验证带（该 $N_A$ 的 $N_B$ 需求区间，宽约 56 个）')
    # 下包络：各 N_A 最低验证点连线（仍不足 0.90）
    ax.plot(nas, lo, color='#C44E52', lw=1.8,
            label='最低验证点下包络（CI 上界 $<0.90$，仍不足）')
    # 跨线 7 点（区间含 0.90）与可靠 (617,0)
    cross = [(r['na'], r['nb']) for r in rows if r['v'] == 'crossing']
    if cross:
        ax.scatter([c[0] for c in cross], [c[1] for c in cross], marker='x',
                   s=60, c='#4C72B0', linewidths=1.6, zorder=4,
                   label='跨线（95% CI 含 0.90，$N_A\\geq608$ 小 B 段）')
    ax.scatter([na_star], [nb_star], marker='*', s=200, c='#55A868',
               edgecolors='#111111', linewidths=0.9, zorder=5,
               label=f'可靠 $(617,0)$，{c_star:.4f} 元')
    # 关键段标注
    ax.annotate('$N_A=0$：需 $5360\\sim5416$ 个 B\n（纯 B 端点，概率仍不足）',
                xy=(0, 5360), xytext=(20, 4700), fontsize=8.5,
                color='#333333', bbox=dict(boxstyle='round,pad=0.25',
                fc='white', ec='none', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=0.9))
    ax.annotate('$N_A=600$：仅需 $45\\sim101$ 个 B\n（每根 A 替代约 90 个 B）',
                xy=(600, 45), xytext=(430, 420), fontsize=8.5,
                color='#333333', bbox=dict(boxstyle='round,pad=0.25',
                fc='white', ec='none', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=0.9))
    ax.annotate('跨线段：$N_A=608\\sim611$，$N_B\\leq21$\n'
                '（验证带与 0.90 相交，不可认证）',
                xy=(609, 12), xytext=(500, 1800), fontsize=8.5,
                color='#4C72B0', bbox=dict(boxstyle='round,pad=0.25',
                fc='white', ec='none', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='#4C72B0', lw=0.9))
    ax.set_xlabel('介质A 数量 $N_A$（根）')
    ax.set_ylabel('介质B 数量 $N_B$（个）')
    ax.set_xlim(-15, 640)
    ax.set_ylim(-60, 5750)
    ax.legend(loc='upper right', fontsize=8.5, framealpha=0.95)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_pair(fig, '问题4_可行边界验证')
    print(f'图2 验证带：{len(rows)} 点，{len(per)} 个 N_A')


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    if not os.path.exists(CSV_POINTS):
        raise SystemExit(f'缺 {CSV_POINTS}，先跑 run_question4.py 生成结果')
    plot_cost_probability()
    if os.path.exists(CSV_BOUNDARY):
        plot_boundary()
    print(f'图片已写入 {FIG_DIR}（png+pdf）')


if __name__ == '__main__':
    main()
