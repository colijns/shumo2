# -*- coding: utf-8 -*-
"""
本绘图程序在AI工具辅助下完成。
AI工具名称：DeepSeek-V4-Flash（Claude Code）v4-flash，杭州深度求索人工智能基础技术研究有限公司，使用日期 2026-08-09。

功能：问题1 三组微构体的接触网络与导电路径 3D 全景 + 2D 投影组合配图
规格：docs/问题1报告_修订版.md；数据：results/question1_result.json + attachment/附件.xlsx
用法：python plot_question1.py
输出：Q1/figures/ 下每组 png（300 dpi）+ pdf（矢量）双格式；
      docs/appendix/figures/ 入池副本（competition-record 步骤 4）

口径：以统一口径修订版为准——主结果按片段实际位置判定（pbc=False），
      不导通组（组1）witness_path 为空，仅显示接触边，无路径高亮。
图例：黄色粗线 = 见证导电路径上的介质；红色粗线 = 路径接触边；
      灰色细线 = 其余介质；黑色细线 = 其余接触边；
      半透明黄面 = 左右带电面；
      2D 面板中黄色圆点（白描边）+ 编号标注 = 路径节点介质。
说明：介质 A 半径 30 nm 相对微构体边长 10000 nm 不可见，
      图以轴线表示介质（论文正文已注明比例缩放）。
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 注册 3D 投影，勿删
import numpy as np
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import load_cylinders

# 中文字体（headless 环境统一走 SimHei）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
XLSX = os.path.join(os.path.dirname(HERE), 'attachment', '附件.xlsx')
JSON_PATH = os.path.join(HERE, 'results', 'question1_result.json')
FIG_DIR = os.path.join(HERE, 'figures')
APPENDIX_FIG_DIR = os.path.join(os.path.dirname(HERE), 'docs', 'appendix', 'figures')

HALF = 5000.0
GROUP_NAMES = ['组1', '组2', '组3']

# 密度分级：(非路径介质 alpha, 非路径接触边 alpha)——组越大介质越多越淡化。
# 3D 面板：组3 全部介质低 alpha 表现填充密度，路径全强度，保证不糊。
DENSITY_3D = {1: (0.55, 0.45), 2: (0.45, 0.35), 3: (0.08, 0.30)}
# 2D 面板：组3 非网络介质压到极淡纹理（3D 已承担几何上下文），网络边居中。
DENSITY_2D = {1: (0.35, 0.40), 2: (0.30, 0.35), 3: (0.06, 0.40)}

# 路径高亮色
PATH_MED_COLOR = '#f9a825'
PATH_EDGE_COLOR = '#d32f2f'
ELECTRODE_COLOR = '#ffd54f'
TEXT_COLOR = '#333333'


def draw_cube_and_electrodes(ax):
    """立方体线框 + 左右带电面（半透明）。"""
    s = HALF
    for corners in (
        [(-s, -s, -s), (s, -s, -s)], [(-s, s, -s), (s, s, -s)],
        [(-s, -s, s), (s, -s, s)], [(-s, s, s), (s, s, s)],
        [(-s, -s, -s), (-s, s, -s)], [(s, -s, -s), (s, s, -s)],
        [(-s, -s, s), (-s, s, s)], [(s, -s, s), (s, s, s)],
        [(-s, -s, -s), (-s, -s, s)], [(s, -s, -s), (s, -s, s)],
        [(-s, s, -s), (-s, s, s)], [(s, s, -s), (s, s, s)],
    ):
        ax.plot(*zip(*corners), color='0.55', linewidth=0.8, zorder=1)

    # 左右带电面（X 方向两端）
    y, z = np.array([-s, s]), np.array([-s, s])
    Y, Z = np.meshgrid(y, z)
    for xf in (-s, s):
        ax.plot_surface(np.full_like(Y, xf), Y, Z,
                        color=ELECTRODE_COLOR, alpha=0.25, zorder=0)


def axis_x_intersection(p1, p2, x_target, tol=0.05):
    """介质轴线 p1->p2 与平面 x=x_target 的交点 3D 坐标；无交点返回 None。

    tol 允许小幅外推（默认 ±5%）：端面接触判定按距离阈值（介质半径），
    轴线端点与面之间可能有 30 nm 级间隙，严格 t∈[0,1] 会漏掉 S/T 端点。
    """
    x1, x2 = p1[0], p2[0]
    if abs(x2 - x1) < 1e-12:
        return None
    t = (x_target - x1) / (x2 - x1)
    if not (-tol <= t <= 1 + tol):
        return None
    return p1 + t * (p2 - p1)


def draw_2d(ax, c, u, h, p1, p2, group, path_nodes, path_edges, gidx,
            proj='xz'):
    """2D 投影面板：接触网络清晰视图（默认 x-z 平面，沿 y 看）。

    proj='xz' 取 (x, z) 轴，proj='xy' 取 (x, y) 轴（验证对比用）。
    层序：带电面带(0) → 立方体框(1) → 非路径介质(2) → 非路径边(3)
          → 路径边(5) → 路径介质(6) → 路径节点点与标签(7)。
    """
    ia, ib = (0, 2) if proj == 'xz' else (0, 1)
    med_alpha, edge_alpha = DENSITY_2D[gidx]
    wp = group['witness_path']

    ax.set_aspect('equal')
    ax.set_xlim(-HALF, HALF)
    ax.set_ylim(-HALF, HALF)
    ax.set_xticks([-HALF, 0, HALF])
    ax.set_yticks([-HALF, 0, HALF])
    ax.tick_params(labelsize=8)
    ax.set_xlabel('x (nm)')
    ax.set_ylabel(('z (nm)' if proj == 'xz' else 'y (nm)'))

    # 带电面：淡黄带 + 边线 + 文字标注（文本不穿数据色）
    for xf, label in ((-HALF, '左带电面 x=-5000'), (HALF, '右带电面 x=+5000')):
        ax.axvspan(xf - 120, xf + 120, color=ELECTRODE_COLOR, alpha=0.15,
                   zorder=0)
        ax.axvline(xf, color=ELECTRODE_COLOR, linewidth=2, alpha=0.6, zorder=1)
        ax.text(xf, HALF * 0.95, label, ha='center', va='top', fontsize=8.5,
                color=TEXT_COLOR, zorder=8)
    # 立方体轮廓
    ax.add_patch(Rectangle((-HALF, -HALF), 2 * HALF, 2 * HALF,
                           fill=False, edgecolor='0.55', linewidth=0.8,
                           zorder=1))

    # 非路径介质轴线投影
    for i in range(len(c)):
        if i in path_nodes:
            continue
        ax.plot([p1[i, ia], p2[i, ia]], [p1[i, ib], p2[i, ib]],
                color='0.65', linewidth=0.6, alpha=med_alpha, zorder=2)

    # 接触边：路径边红粗置顶，其余黑细
    for e in group['contact_edges']:
        i = int(e['i'][1:]) - 1
        j = int(e['j'][1:]) - 1
        if (min(i, j), max(i, j)) in path_edges:
            ax.plot([c[i, ia], c[j, ia]], [c[i, ib], c[j, ib]],
                    color=PATH_EDGE_COLOR, linewidth=2.8,
                    solid_capstyle='round', zorder=5)
        else:
            ax.plot([c[i, ia], c[j, ia]], [c[i, ib], c[j, ib]],
                    color='0.25', linewidth=0.8, alpha=edge_alpha, zorder=3)

    # 路径介质 + 节点标注（只标路径节点，选择性直接标注）
    if wp:
        for i in path_nodes:
            ax.plot([p1[i, ia], p2[i, ia]], [p1[i, ib], p2[i, ib]],
                    color=PATH_MED_COLOR, linewidth=3.0, zorder=6)
            ax.plot(c[i, ia], c[i, ib], 'o', ms=7, color=PATH_MED_COLOR,
                    markeredgecolor='white', markeredgewidth=1.0, zorder=7)
            ax.annotate(f"A{i + 1}", (c[i, ia], c[i, ib]), xytext=(7, 7),
                        textcoords='offset points', fontsize=8.5,
                        color=TEXT_COLOR, zorder=8)

        # S/T 端点：路径首/末介质轴线与 x=±5000 平面的交点
        first, last = int(wp[1][1:]) - 1, int(wp[-2][1:]) - 1
        for idx, x_target, tag in ((first, -HALF, 'S'), (last, HALF, 'T')):
            hit = axis_x_intersection(p1[idx], p2[idx], x_target)
            if hit is not None:
                ax.plot(hit[ia], hit[ib], 'o', ms=5, color=PATH_EDGE_COLOR,
                        zorder=8)
                ax.annotate(tag, (hit[ia], hit[ib]), xytext=(4, 4),
                            textcoords='offset points', fontsize=8,
                            color=TEXT_COLOR, zorder=8)


def main():
    with open(JSON_PATH, encoding='utf-8') as f:
        payload = json.load(f)

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(APPENDIX_FIG_DIR, exist_ok=True)

    for group, idx in zip(payload['groups'], range(3)):
        c, u, h, _ = load_cylinders(XLSX, idx)
        n = len(c)
        p1 = c - h[:, None] * u
        p2 = c + h[:, None] * u
        gidx = idx + 1
        med_alpha, edge_alpha = DENSITY_3D[gidx]

        # 统一口径主结果：不导通组 witness_path 为 null（如组1），
        # 此时无路径节点与路径边，直接跳过路径高亮。
        wp = group['witness_path']
        path_nodes = set()
        path_edges = set()
        if wp:
            path_nodes = set(int(v[1:]) - 1 for v in wp[1:-1])
            for a, b in zip(wp[1:-1], wp[2:-1]):
                path_edges.add((min(int(a[1:]) - 1, int(b[1:]) - 1),
                                max(int(a[1:]) - 1, int(b[1:]) - 1)))

        # 有接触边的介质（网络参与体）：3D 面板提亮分层
        network_nodes = set()
        for e in group['contact_edges']:
            network_nodes.add(int(e['i'][1:]) - 1)
            network_nodes.add(int(e['j'][1:]) - 1)

        fig = plt.figure(figsize=(14.5, 6.8))
        ax3d = fig.add_subplot(1, 2, 1, projection='3d')
        ax2d = fig.add_subplot(1, 2, 2)
        # 图内不设总标题（组级信息放论文图注），仅保留 (a)(b) 面板标注
        fig.subplots_adjust(wspace=0.22, bottom=0.14, top=0.92)

        draw_cube_and_electrodes(ax3d)

        # ---- 3D 面板：接触边（黑细）与路径边（红粗）----
        for e in group['contact_edges']:
            i = int(e['i'][1:]) - 1
            j = int(e['j'][1:]) - 1
            if (min(i, j), max(i, j)) in path_edges:
                ax3d.plot(*zip(c[i], c[j]), color=PATH_EDGE_COLOR,
                          linewidth=2.6, zorder=5)
            else:
                ax3d.plot(*zip(c[i], c[j]), color='0.25',
                          linewidth=0.7, alpha=edge_alpha, zorder=3)

        # ---- 3D 面板：介质轴线（路径黄粗，网络内提亮，其余按密度淡化）----
        for i in range(n):
            if i in path_nodes:
                ax3d.plot(*zip(p1[i], p2[i]), color=PATH_MED_COLOR,
                          linewidth=3.0, zorder=6)
            elif i in network_nodes:
                ax3d.plot(*zip(p1[i], p2[i]), color='0.55',
                          linewidth=0.9, alpha=min(med_alpha * 3.0, 0.35),
                          zorder=2)
            else:
                ax3d.plot(*zip(p1[i], p2[i]), color='0.65',
                          linewidth=0.6, alpha=med_alpha, zorder=2)

        ax3d.set_xlim(-HALF, HALF)
        ax3d.set_ylim(-HALF, HALF)
        ax3d.set_zlim(-HALF, HALF)
        ax3d.set_box_aspect((1, 1, 1))
        ax3d.set_xlabel('x (nm)')
        ax3d.set_ylabel('y (nm)')
        ax3d.set_zlabel('z (nm)')
        ax3d.tick_params(labelsize=8)
        # 组3 介质密集，略抬视角减少前后遮挡
        ax3d.view_init(elev=18 if gidx != 3 else 20,
                       azim=-60 if gidx != 3 else -55)
        ax3d.set_title('(a) 3D 立体全景', fontsize=11)

        # ---- 2D 面板 ----
        draw_2d(ax2d, c, u, h, p1, p2, group, path_nodes, path_edges, gidx,
                proj='xz')
        ax2d.set_title('(b) 接触网络 2D 投影（x–z 平面）', fontsize=11)

        state = '导通' if group['conductive'] else '不导通'

        # 图例（底部通栏）：无见证路径时（不导通组）不显示路径相关项
        legend_items = []
        if wp:
            legend_items += [
                Line2D([0], [0], color=PATH_MED_COLOR, linewidth=3.0,
                       label='见证路径介质'),
                Line2D([0], [0], color=PATH_EDGE_COLOR, linewidth=2.6,
                       label='路径接触边'),
            ]
        legend_items += [
            Line2D([0], [0], color='0.65', linewidth=1.0, label='其余介质（轴线）'),
            Line2D([0], [0], color='0.25', linewidth=0.7, label='其余接触边'),
            Line2D([0], [0], color=ELECTRODE_COLOR, linewidth=4, alpha=0.5,
                   label='左右带电面'),
        ]
        fig.legend(handles=legend_items, loc='lower center', fontsize=9,
                   ncol=len(legend_items))

        stem = f'问题1_{group["name"]}_接触网络与导电路径'
        png = os.path.join(FIG_DIR, stem + '.png')
        pdf = os.path.join(FIG_DIR, stem + '.pdf')
        fig.savefig(png, dpi=300, bbox_inches='tight')
        fig.savefig(pdf, bbox_inches='tight')
        plt.close(fig)
        # competition-record 步骤 4：入池（矢量 + png）
        for src in (png, pdf):
            shutil.copy2(src, os.path.join(APPENDIX_FIG_DIR, os.path.basename(src)))
        print(f'{stem}: {n} 介质, {len(group["contact_edges"])} 接触边, '
              f'路径 {wp}, {state} -> {png}')

    print(f'figures pool: {APPENDIX_FIG_DIR}')


if __name__ == '__main__':
    main()
