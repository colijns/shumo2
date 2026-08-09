# 本绘图程序在AI工具辅助下完成
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31

"""问题2 附图：接触网络导通机理示意（图9）。

手工构造示意（非数值结果）：x–y 平面俯视图，展示
  - 左右带电面与导通链（片段端面间距 ≤ 1.8 nm 逐段连通，S→T 贯通路径）；
  - 跨壁圆柱经周期平移在相对侧产生同源片段（不自动电连接）；
  - 接触阈值局部放大插图（1.8 nm 阈值与 r=30 nm 端面圆）。
介质以轴线表示，30 nm 半径相对 10000 nm 微构体不可见（论文正文已注明比例缩放）。
输出：Q2/figures/问题2_接触网络_机理示意.png + .pdf（300dpi，双格式），
      并复制入 docs/appendix/figures/ 图池（competition-record 步骤 2+4）。
"""

import os
import shutil
import tempfile

os.environ.setdefault(
    'MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'shumo2-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle

_Q2 = os.path.dirname(os.path.abspath(__file__))
# 与模板 plot_style 同款中文样式（内联，避免依赖 templates 路径）
sys_fonts = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['font.sans-serif'] = sys_fonts
plt.rcParams['axes.unicode_minus'] = False

FIG_DIR = os.path.join(_Q2, 'figures')
APPENDIX_FIG_DIR = os.path.join(os.path.dirname(_Q2), 'docs', 'appendix', 'figures')

HALF = 5000.0
PATH_MED_COLOR = '#f9a825'      # 导通链片段（黄）
PATH_EDGE_COLOR = '#d32f2f'     # 链接触边 / S/T 端点（红）
ELECTRODE_COLOR = '#ffd54f'     # 带电面（淡黄）
TEXT_COLOR = '#333333'

# 导通链关节坐标（示意性，非实测；片段=相邻关节连线，关节即接触点）
JOINTS = [
    (-5000, -400), (-3700, 500), (-2450, -600), (-1200, 300), (100, -150),
    (1350, 700), (2650, 0), (3950, -500), (5000, 200),
]

# 背景介质轴线（示意）：3 根随机朝向 + 一对"留缝不接触"片段
BG_LINES = [
    ((500, 3600), (4500, 3950)),
    ((-4300, -3400), (-1800, -3050)),
    ((-3300, 4100), (-500, 4400)),
]
# 留缝对：端头间距约 255 nm >> 1.8 nm，不建边
GAP_A = ((-2100, -2900), (-1650, -2960))
GAP_B = ((-1400, -3010), (-950, -3070))

# 跨壁平移演示：原圆柱 (1000,-2600)->(5800,-2150) 跨 x=+5000 边界，
# 越界段 (5000..5800) 沿 -x 平移 10000 得同源片段 (-5000..-4200, y=-2150)。
CROSS_ORIG = ((1000, -2600), (5800, -2150))
CROSS_TRANSLATED = ((-5000, -2150), (-4200, -2150))


def draw_electrodes_and_box(ax):
    """左右带电面淡黄带 + 边线 + 文字，微构体盒轮廓。"""
    for xf, label in ((-HALF, '左带电面 x=-5000'), (HALF, '右带电面 x=+5000')):
        ax.axvspan(xf - 120, xf + 120, color=ELECTRODE_COLOR, alpha=0.15)
        ax.axvline(xf, color=ELECTRODE_COLOR, linewidth=2, alpha=0.6)
        ax.text(xf, HALF * 0.95, label, ha='center', va='top', fontsize=8.5,
                color=TEXT_COLOR)
    ax.add_patch(Rectangle((-HALF, -HALF), 2 * HALF, 2 * HALF, fill=False,
                           edgecolor='0.55', linewidth=0.8))


def draw_conductive_chain(ax):
    """导通链：黄粗片段 + 红粗接触边 + 节点编号 + S/T 端点。"""
    pts = list(zip(*JOINTS))
    for (x1, y1), (x2, y2) in zip(JOINTS[:-1], JOINTS[1:]):
        ax.plot([x1, x2], [y1, y2], color=PATH_MED_COLOR, linewidth=3.0,
                solid_capstyle='round', zorder=6)
    # 接触边：连相邻片段中心（节点），红粗
    centers = [((JOINTS[i][0] + JOINTS[i + 1][0]) / 2,
                (JOINTS[i][1] + JOINTS[i + 1][1]) / 2)
               for i in range(len(JOINTS) - 1)]
    for i in range(len(centers) - 1):
        ax.plot([centers[i][0], centers[i + 1][0]],
                [centers[i][1], centers[i + 1][1]], color=PATH_EDGE_COLOR,
                linewidth=2.8, solid_capstyle='round', zorder=5)
    for i, (cx, cy) in enumerate(centers):
        ax.plot(cx, cy, 'o', ms=7, color=PATH_MED_COLOR,
                markeredgecolor='white', markeredgewidth=1.0, zorder=7)
        ax.annotate(f'A{i + 1}', (cx, cy), xytext=(7, 7),
                    textcoords='offset points', fontsize=8.5, color=TEXT_COLOR,
                    zorder=8)
    # S/T 端点：链首末关节落在电极面 x=±5000 上
    for (sx, sy), tag in ((JOINTS[0], 'S'), (JOINTS[-1], 'T')):
        ax.plot(sx, sy, 'o', ms=5, color=PATH_EDGE_COLOR, zorder=8)
        ax.annotate(tag, (sx, sy), xytext=(4, 4), textcoords='offset points',
                    fontsize=8, color=TEXT_COLOR, zorder=8)


def draw_crossing_translation(ax):
    """跨壁平移演示：越界段与原圆柱同源，平移后同源片段在相对侧。"""
    (x1, y1), (x2, y2) = CROSS_ORIG
    # 原圆柱盒内段实线、越界段浅色
    ax.plot([x1, min(x2, HALF)], [y1, y1 + (y2 - y1) * (HALF - x1) / (x2 - x1)],
            color='0.5', linewidth=1.2, zorder=2)
    ax.plot([HALF, x2], [y1 + (y2 - y1) * (HALF - x1) / (x2 - x1), y2],
            color='0.75', linewidth=1.0, zorder=2)
    # 平移后同源片段（虚线浅灰）
    ax.plot([CROSS_TRANSLATED[0][0], CROSS_TRANSLATED[1][0]],
            [CROSS_TRANSLATED[0][1], CROSS_TRANSLATED[1][1]],
            color='0.65', linestyle='--', linewidth=1.4, zorder=2)
    # 平移映射弧线箭头：越界段中点 -> 同源片段中点
    mid_orig = ((HALF + x2) / 2, (y1 + y2) / 2)
    mid_new = ((CROSS_TRANSLATED[0][0] + CROSS_TRANSLATED[1][0]) / 2,
               (CROSS_TRANSLATED[0][1] + CROSS_TRANSLATED[1][1]) / 2)
    ax.annotate('', xy=mid_new, xytext=mid_orig,
                arrowprops=dict(arrowstyle='-|>', color='0.4', linewidth=1.2,
                                connectionstyle='arc3,rad=0.35'))
    ax.text(-2500, -3450, '周期平移：越界部分平移一个边长，\n同源片段出现在相对侧',
            ha='center', va='top', fontsize=8.5, color=TEXT_COLOR,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(-4700, -1700, '同源片段不自动电连接\n（须实际距离 ≤ 1.8 nm）',
            ha='left', va='bottom', fontsize=8, color=TEXT_COLOR,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def draw_background(ax):
    """背景随机朝向介质 + 留缝不接触片段对（偶发缺口）。"""
    for (x1, y1), (x2, y2) in BG_LINES:
        ax.plot([x1, x2], [y1, y2], color='0.65', linewidth=0.8, alpha=0.4,
                zorder=2)
    for (x1, y1), (x2, y2) in (GAP_A, GAP_B):
        ax.plot([x1, x2], [y1, y2], color='0.65', linewidth=0.8, alpha=0.4,
                zorder=2)
    # 留缝处标注（端面间距约 255 nm >> 1.8 nm）
    ax.annotate('间距 > 1.8 nm → 不接触', (-1525, -2985),
                xytext=(-2600, -2600), fontsize=8, color=TEXT_COLOR,
                arrowprops=dict(arrowstyle='-', color='0.5', linewidth=0.9))


def draw_threshold_inset(fig, ax):
    """接触阈值局部放大插图：1.8 nm 阈值 + r=30 nm 端面圆（示意，非严格比例）。"""
    # 主图：链上中间关节画虚线放大圈 + 引出箭头
    jx, jy = JOINTS[4]
    ax.add_patch(Circle((jx, jy), 230, fill=False, color=PATH_EDGE_COLOR,
                        linestyle='--', linewidth=1.1, zorder=9))
    ax.annotate('', xy=(0.76, 0.32), xycoords='figure fraction',
                xytext=(jx, jy), textcoords='data',
                arrowprops=dict(arrowstyle='-|>', color='0.35', linewidth=1.0,
                                connectionstyle='arc3,rad=-0.25'))
    # 插图：两片段端头 + 表面间距双头箭头 + 端面圆
    axins = ax.inset_axes([0.72, 0.06, 0.26, 0.30])
    axins.set_xlim(0, 10)
    axins.set_ylim(0, 6)
    axins.axis('off')
    axins.set_title('接触阈值局部放大（示意）', fontsize=8)
    axins.add_patch(Rectangle((3.2, 1), 1.2, 4, facecolor=PATH_MED_COLOR,
                              edgecolor='none', alpha=0.9))
    axins.add_patch(Rectangle((5.6, 1), 1.2, 4, facecolor=PATH_MED_COLOR,
                              edgecolor='none', alpha=0.9))
    axins.annotate('', xy=(5.6, 3), xytext=(4.4, 3),
                   arrowprops=dict(arrowstyle='<|-|>', color=PATH_EDGE_COLOR,
                                   linewidth=1.4))
    axins.text(5, 3.5, '表面间距 ≤ 1.8 nm\n→ 判为接触', ha='center',
               fontsize=7.5, color=TEXT_COLOR)
    axins.add_patch(Circle((8, 1.3), 0.9, fill=False, edgecolor='0.4',
                           linewidth=1.0))
    axins.text(8, 0.4, 'r = 30 nm', ha='center', fontsize=7.5, color=TEXT_COLOR)


def main():
    fig, ax = plt.subplots(figsize=(8.6, 7))
    ax.set_aspect('equal')
    ax.set_xlim(-HALF, HALF)
    ax.set_ylim(-HALF, HALF)
    ax.set_xticks([-HALF, 0, HALF])
    ax.set_yticks([-HALF, 0, HALF])
    ax.tick_params(labelsize=8)
    ax.set_xlabel('x (nm)')
    ax.set_ylabel('y (nm)')

    draw_electrodes_and_box(ax)
    draw_background(ax)
    draw_crossing_translation(ax)
    draw_conductive_chain(ax)
    draw_threshold_inset(fig, ax)

    # 图例（底部通栏）
    legend_items = [
        Line2D([0], [0], color='0.65', linewidth=0.8, label='介质A轴线'),
        Line2D([0], [0], color=PATH_MED_COLOR, linewidth=3.0,
               label='导通链片段'),
        Line2D([0], [0], color=PATH_EDGE_COLOR, linewidth=2.8, label='接触边（≤ 1.8 nm）'),
        Line2D([0], [0], color='0.65', linestyle='--', linewidth=1.4,
               label='跨壁平移后的同源片段'),
        Line2D([0], [0], color=ELECTRODE_COLOR, linewidth=4, alpha=0.5,
               label='左右带电面'),
        Line2D([0], [0], marker='o', color='none', markeredgecolor=PATH_EDGE_COLOR,
               markerfacecolor=PATH_EDGE_COLOR, markersize=6,
               label='导通链与电极接触端点 S/T'),
    ]
    fig.legend(handles=legend_items, loc='lower center', fontsize=9, ncol=3)

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(APPENDIX_FIG_DIR, exist_ok=True)
    stem = '问题2_接触网络_机理示意'
    base = os.path.join(FIG_DIR, stem)
    for ext in ('png', 'pdf'):
        fig.savefig(base + '.' + ext, dpi=300, bbox_inches='tight')
        shutil.copy2(base + '.' + ext,
                     os.path.join(APPENDIX_FIG_DIR, stem + '.' + ext))
    plt.close(fig)
    print(f'已保存并入池：{stem}.png / .pdf（dpi=300）')
    print(f'图池：{APPENDIX_FIG_DIR}')


if __name__ == '__main__':
    main()
