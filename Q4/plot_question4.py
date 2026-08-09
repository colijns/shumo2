# 本绘图程序在AI工具辅助下完成
# AI工具名称：DeepSeek-V4-Flash（Claude Code）
# 版本：v4-flash（2026-08 使用）
# 开发机构：杭州深度求索人工智能基础技术研究有限公司
# 使用日期：2026-08-09
# 人工修改记录：见 docs/appendix/interaction_logs/edit_trace.md

"""问题4 论文配图：混合填充二维搜索热图 + 90% 可行边界（png+pdf 双格式）。

- 图1 主图：全部评估点 (N_A, N_B) → p_hat 热图（scipy griddata 线性插值），
  叠加 0.90 等值线、最优解标记、纯A/纯B 成本参考线；
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

try:
    from scipy.interpolate import griddata  # noqa: E402
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

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
GRID_N = 400          # 插值网格单轴密度
RHO = 0.012           # 热图配色：低 p 透明


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


def plot_heatmap():
    """图1：p_hat 二维热图 + 0.90 等值线 + 最优标记 + 纯方案参考线。"""
    na, nb, p = load_points(CSV_POINTS)
    res = load_result(CSV_RESULT)
    na_star, nb_star = int(res['N_A_star']), int(res['N_B_star'])
    cost_a = float(res['纯A成本_元'])
    cb_raw = res['纯B成本_元']
    cost_b = float(cb_raw) if cb_raw != '不可行' else None
    na_max, nb_max = na.max(), nb.max()

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    if HAS_SCIPY and len(na) > 3:
        gi = np.linspace(0, na_max, GRID_N)
        gj = np.linspace(0, nb_max, GRID_N)
        GI, GJ = np.meshgrid(gi, gj)
        P = griddata((na, nb), p, (GI, GJ), method='linear')
    else:
        GI, GJ = np.meshgrid(np.linspace(0, na_max, 4),
                             np.linspace(0, nb_max, 4))
        P = np.full_like(GI, np.nan)
    # 插值 NaN（热图边界外）置为最近非 NaN（线性插值外推不可行，用最近邻填充）
    if np.isnan(P).any():
        Pn = griddata((na, nb), p, (GI, GJ), method='nearest')
        P = np.where(np.isnan(P), Pn, P)
    im = ax.pcolormesh(GI, GJ, P, cmap='RdYlBu', vmin=0.0, vmax=1.0,
                       shading='auto')
    cs = ax.contour(GI, GJ, P, levels=[P_TARGET], colors='#111111',
                    linewidths=1.8)
    ax.clabel(cs, fmt='%.2f', fontsize=9)
    # 原始评估点（淡色，展示搜索带）
    ax.scatter(na, nb, s=3, c='#555555', alpha=RHO, linewidths=0)
    # 纯方案成本参考线
    if cost_a > 0:
        ax.axvline(float(res['N_A_hat90']), color='#C44E52', ls='--', lw=1.2,
                   label=f"纯A 成本线 $N_A$={res['N_A_hat90']} "
                        f"({cost_a:.3f} 元)")
    if cost_b is not None and cost_b > 0:
        ax.axhline(float(res['N_B_hat90']), color='#55A868', ls='--', lw=1.2,
                   label=f"纯B 成本线 $N_B$={res['N_B_hat90']} "
                        f"({float(cost_b):.3f} 元)")
    ax.plot([na_star], [nb_star], marker='*', ms=16, mfc='#FFD700',
            mec='#111111', mew=1.2, ls='none',
            label=f"最优 $({na_star},{nb_star})$ 成本 {float(res['C_star_元']):.4f} 元")
    ax.set_xlabel('介质A 数量 $N_A$（根）')
    ax.set_ylabel('介质B 数量 $N_B$（个）')
    ax.set_title('问题4：导通概率 $\\hat P(N_A,N_B)$ 热图与 90% 可行边界')
    ax.legend(loc='upper right', fontsize=8.5)
    cb = fig.colorbar(im, ax=ax, label='导通概率 $\\hat P$')
    cb.ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    save_pair(fig, '问题4_混合导通热图')
    print(f'图1 热图：{len(na)} 评估点，最优 ({na_star},{nb_star})')


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
    ax.set_title('问题4：验证带边界点与非支配判定')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_pair(fig, '问题4_可行边界验证')
    print(f'图2 边界：{len(rows)} 点')


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    if not os.path.exists(CSV_POINTS):
        raise SystemExit(f'缺 {CSV_POINTS}，先跑 run_question4.py 生成结果')
    plot_heatmap()
    if os.path.exists(CSV_BOUNDARY):
        plot_boundary()
    print(f'图片已写入 {FIG_DIR}（png+pdf）')


if __name__ == '__main__':
    main()
