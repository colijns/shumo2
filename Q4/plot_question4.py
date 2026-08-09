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

    横轴为总成本 C = c_A·N_A + c_B·N_B，纵轴为导通概率点估计；全部搜索评估点
    作灰色背景（大量低成本点概率极低，直观显示介质 B 无长程连通优势），
    p̂≥0.85 的候选按 N_B 着色，叠加独立种子复算的关键候选（星=可靠 / 叉=跨线）。
    """
    na, nb, p = load_points(CSV_POINTS)
    res = load_result(CSV_RESULT)
    c_a, c_b = float(res['c_A_元']), float(res['c_B_元'])
    cost = c_a * na + c_b * nb
    na_star, nb_star = int(res['N_A_star']), int(res['N_B_star'])
    c_star = float(res['C_star_元'])

    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    # 背景：全部搜索评估点（M=100 点估计），低成本段大团灰点 = 介质 B 无效区
    ax.scatter(cost, p, s=6, c='#A0A0A0', alpha=0.22, linewidths=0,
               label='搜索评估点（$M=100$ 点估计）')
    # 前景：p̂ ≥ 0.85 的候选，颜色按 N_B（0=纯A，高值=多B）
    m = p >= 0.85
    sc = ax.scatter(cost[m], p[m], s=26, c=nb[m], cmap='viridis', vmin=0,
                    vmax=62, linewidths=0.4, edgecolors='#333333',
                    label='$\\hat P\\geq0.85$ 候选（颜色=$N_B$）')
    cb = fig.colorbar(sc, ax=ax, label='介质B 数量 $N_B$（个）')
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
    # 关键候选文字标注
    ax.annotate('搜索定位 $(598,62)$：\n8.981 元，$M=100$ 点估计 0.90，不可靠',
                xy=(c_a * 598 + c_b * 62, 0.90), xytext=(6.6, 0.80),
                fontsize=8.5, color='#333333',
                arrowprops=dict(arrowstyle='->', color='#333333', lw=0.9))
    ax.annotate('$(608,14)$ 9.049 元：\n独立复算跨线',
                xy=(c_a * 608 + c_b * 14, 0.8925), xytext=(7.4, 0.60),
                fontsize=8.5, color='#4C72B0',
                arrowprops=dict(arrowstyle='->', color='#4C72B0', lw=0.9))
    ax.annotate(f'独立复算可靠 $({na_star},{nb_star})$ {c_star:.4f} 元',
                xy=(c_star, 0.91075), xytext=(c_star - 0.25, 0.965),
                fontsize=8.5, color='#2e7d32',
                arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=0.9))
    ax.set_xlabel('总成本 $C=c_A N_A+c_B N_B$（元）')
    ax.set_ylabel('导通概率点估计 $\\hat P$')
    ax.set_xlim(4.4, 9.3)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc='upper right', fontsize=8.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_pair(fig, '问题4_混合导通热图')
    print(f'图1 成本-概率：{len(na)} 评估点，可靠标注 ({na_star},{nb_star})')


def plot_boundary():
    """图2：最低可行层边界曲线，Wilson 判定着色。"""
    rows = load_boundary(CSV_BOUNDARY)
    res = load_result(CSV_RESULT)
    na_star, nb_star = int(res['N_A_star']), int(res['N_B_star'])

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    color = {'reliable': '#55A868', 'insufficient': '#C44E52',
             'crossing': '#4C72B0'}
    label = {'reliable': '可靠可行（CI 下界 ≥ 0.90）',
             'insufficient': '不足（CI 上界 < 0.90）',
             'crossing': '跨线（区间含 0.90）'}
    drawn = set()
    for r in rows:
        c = color.get(r['v'], '#888888')
        l = label.get(r['v'])
        if l in drawn:
            l = None
        elif l:
            drawn.add(l)
        ax.scatter([r['na']], [r['nb']], s=22, c=c, label=l,
                   alpha=0.85, linewidths=0)
    # 边界曲线连接（按 na 升序）
    rsort = sorted(rows, key=lambda r: r['na'])
    if len(rsort) > 1:
        ax.plot([r['na'] for r in rsort], [r['nb'] for r in rsort],
                color='#888888', lw=0.8, ls=':', zorder=0)
    ax.plot([na_star], [nb_star], marker='*', ms=15, mfc='#FFD700',
            mec='#111111', mew=1.2, ls='none',
            label=f"最优 $({na_star},{nb_star})$")
    ax.set_xlabel('介质A 数量 $N_A$（根）')
    ax.set_ylabel('介质B 数量 $N_B$（个）')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_pair(fig, '问题4_可行边界验证')
    print(f'图2 边界：{len(rows)} 点')


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
