# -*- coding: utf-8 -*-
"""
本绘图程序在AI工具辅助下完成。
AI工具名称：DeepSeek-V4-Flash（Claude Code）v4-flash，杭州深度求索人工智能基础技术研究有限公司，使用日期 2026-08-07。

功能：问题1 三组微构体的接触网络与导电路径 3D 配图
规格：docs/问题1报告_修订版.md；数据：results/question1_result.json + attachment/附件.xlsx
用法：python plot_question1.py
输出：Q1/figures/ 下每组 png（300 dpi）+ pdf（矢量）双格式；
      docs/appendix/figures/ 入池副本（competition-record 步骤 4）

口径：以统一口径修订版为准——主结果按片段实际位置判定（pbc=False），
      不导通组（组1）witness_path 为空，仅显示接触边，无路径高亮。
图例：黄色粗线 = 见证导电路径上的介质；红色粗线 = 路径接触边；
      灰色细线 = 其余介质；黑色细线 = 其余接触边；
      半透明黄面 = 左右带电面。
说明：介质 A 半径 30 nm 相对微构体边长 10000 nm 不可见，
      图以轴线表示介质（论文正文已注明比例缩放）。
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
                        color='#ffd54f', alpha=0.25, zorder=0)


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

        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection='3d')
        draw_cube_and_electrodes(ax)

        # 接触边（黑细）与路径边（红粗）；JSON 边为 dict {'i','j','k','distance'}
        for e in group['contact_edges']:
            i = int(e['i'][1:]) - 1
            j = int(e['j'][1:]) - 1
            if (min(i, j), max(i, j)) in path_edges:
                ax.plot(*zip(c[i], c[j]), color='#d32f2f',
                        linewidth=2.6, zorder=5)
            else:
                ax.plot(*zip(c[i], c[j]), color='0.25',
                        linewidth=0.7, alpha=0.55, zorder=3)

        # 介质轴线：路径上黄色粗线，其余灰色细线
        for i in range(n):
            if i in path_nodes:
                ax.plot(*zip(p1[i], p2[i]), color='#f9a825',
                        linewidth=3.2, zorder=6)
            else:
                ax.plot(*zip(p1[i], p2[i]), color='0.65',
                        linewidth=1.0, alpha=0.8, zorder=2)

        ax.set_xlim(-HALF, HALF)
        ax.set_ylim(-HALF, HALF)
        ax.set_zlim(-HALF, HALF)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        ax.set_zlabel('z (nm)')
        ax.view_init(elev=18, azim=-60)

        state = '导通' if group['conductive'] else '不导通'
        path_txt = '（' + '—'.join(group['witness_path']) + '）' \
            if group['witness_path'] else ''
        ax.set_title(f"{group['name']} 微构体接触网络与导电路径：{state} {path_txt}")

        # 图例（手动构造）：无见证路径时（不导通组）不显示路径相关项
        from matplotlib.lines import Line2D
        legend_items = []
        if wp:
            legend_items += [
                Line2D([0], [0], color='#f9a825', linewidth=3.2, label='见证路径介质'),
                Line2D([0], [0], color='#d32f2f', linewidth=2.6, label='路径接触边'),
            ]
        legend_items += [
            Line2D([0], [0], color='0.65', linewidth=1.0, label='其余介质（轴线）'),
            Line2D([0], [0], color='0.25', linewidth=0.7, label='其余接触边'),
            Line2D([0], [0], color='#ffd54f', linewidth=4, alpha=0.5, label='左右带电面'),
        ]
        ax.legend(handles=legend_items, loc='upper left', fontsize=9, ncol=2)

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
              f'{state} -> {png}')

    print(f'figures pool: {APPENDIX_FIG_DIR}')


if __name__ == '__main__':
    main()
