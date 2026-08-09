# 本绘图程序在AI工具辅助下完成
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31

"""问题2 附图：实体几何统计量随体积分数变化（图7）。

数据来源：Q2/results/solid_boundary_comparison.json（正式版，M=2000/体积分数，
seed=20260808，K=64，内接/外切共用同一批随机构型）。四线：内接/外切平均
片段数与平均跨壁片段数，实线实心为内接、虚线空心为外切（内外接差异小于
0.05，曲线近乎重合）。数值标注仅标内接跨壁片段数（渗流机理主角）。
输出：Q2/figures/问题2_几何统计量_vs_体积分数.png + .pdf（300dpi，双格式），
并复制入 docs/appendix/figures/ 图池（competition-record 步骤 2+4）。
"""

import json
import os
import shutil
import tempfile

os.environ.setdefault(
    'MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'shumo2-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_Q2 = os.path.dirname(os.path.abspath(__file__))
# 与模板 plot_style 同款中文样式（内联，避免依赖 templates 路径）
sys_fonts = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['font.sans-serif'] = sys_fonts
plt.rcParams['axes.unicode_minus'] = False

RESULT_JSON = os.path.join(_Q2, 'results', 'solid_boundary_comparison.json')
FIG_DIR = os.path.join(_Q2, 'figures')
APPENDIX_FIG_DIR = os.path.join(os.path.dirname(_Q2), 'docs', 'appendix', 'figures')


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


def load_results():
    with open(RESULT_JSON, encoding='utf-8') as f:
        payload = json.load(f)
    return sorted(payload['rows'], key=lambda r: r['phi'])


def plot_statistics(rows):
    """图7：内/外接平均片段数、平均跨壁片段数 vs 体积分数。

    单轴四条线：片段数（蓝）与跨壁片段数（绿）近似过原点线性；
    实线实心为内接、虚线空心为外切，两者差异小于 0.05 故近乎重合。
    上轴标注各考察点介质数量 N_A，仅内接跨壁片段数数值标注。
    """
    phis = np.array([r['phi'] for r in rows]) * 100
    n_a = [int(r['n_A']) for r in rows]
    inner_frag = np.array([r['mean_inner_fragments'] for r in rows])
    outer_frag = np.array([r['mean_outer_fragments'] for r in rows])
    inner_cross = np.array([r['mean_inner_crossing'] for r in rows])
    outer_cross = np.array([r['mean_outer_crossing'] for r in rows])

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(phis, inner_frag, 'o-', color='#1f77b4', linewidth=1.8,
            markersize=6, label='内接平均片段数')
    ax.plot(phis, outer_frag, 'o--', color='#1f77b4', linewidth=1.8,
            markersize=6, markerfacecolor='none', label='外切平均片段数')
    ax.plot(phis, inner_cross, 's-', color='#2ca02c', linewidth=1.8,
            markersize=6, label='内接平均跨壁片段数')
    ax.plot(phis, outer_cross, 's--', color='#2ca02c', linewidth=1.8,
            markersize=6, markerfacecolor='none', label='外切平均跨壁片段数')

    # 内接跨壁片段数数值标注（渗流机理主角：215.4→258.1→301.0→429.6）
    for x, y in zip(phis, inner_cross):
        ax.annotate(f'{y:.1f}', (x, y), textcoords='offset points',
                    xytext=(0, -14), fontsize=9, ha='center', color='#2ca02c')

    ax.set_xlabel('体积分数 φ（%）')
    ax.set_ylabel('每样本平均数量（根）')
    ax.set_xticks([0.5, 0.6, 0.7, 1.0])
    ax.set_xlim(0.45, 1.05)
    ax.set_ylim(0, 1350)  # 含原点，强调片段/跨壁近似过原点线性
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', fontsize=10)

    # 上轴：介质数量 N_A（与下轴四考察点刻度对齐）
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(ax.get_xticks())
    ax_top.set_xticklabels([str(n) for n in n_a])
    ax_top.set_xlabel('介质数量 N$_A$（根）')

    save_pair(fig, '问题2_几何统计量_vs_体积分数')


def main():
    rows = load_results()
    plot_statistics(rows)
    print(f'图池：{APPENDIX_FIG_DIR}')


if __name__ == '__main__':
    main()
