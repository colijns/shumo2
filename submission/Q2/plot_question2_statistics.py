# 本程序及代码是在AI工具辅助下完成的
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31

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
sys_fonts = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['font.sans-serif'] = sys_fonts
plt.rcParams['axes.unicode_minus'] = False

RESULT_JSON = os.path.join(_Q2, 'results', 'solid_boundary_comparison.json')
FIG_DIR = os.path.join(_Q2, 'figures')
APPENDIX_FIG_DIR = os.path.join(os.path.dirname(_Q2), 'docs', 'appendix', 'figures')


def save_pair(fig, stem):
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
    phis = np.array([r['phi'] for r in rows]) * 100
    n_a = [int(r['n_A']) for r in rows]
    inner_frag = np.array([r['mean_inner_fragments'] for r in rows])
    inner_cross = np.array([r['mean_inner_crossing'] for r in rows])

    fig, ax = plt.subplots(figsize=(7, 4.8))


    ax.plot(phis, inner_frag, 'o-', color='#1f77b4', linewidth=1.8,
            markersize=6, label='平均片段数（内接，外切重合）')
    ax.plot(phis, inner_cross, 's-', color='#2ca02c', linewidth=1.8,
            markersize=6, label='平均跨壁片段数（内接，外切重合）')


    for x, y in zip(phis, inner_cross):
        ax.annotate(f'{y:.1f}', (x, y), textcoords='offset points',
                    xytext=(0, -14), fontsize=9, ha='center', color='#2ca02c')

    ax.set_xlabel('体积分数 φ（%）')
    ax.set_ylabel('每样本平均数量（根）')
    ax.set_xticks([0.5, 0.6, 0.7, 1.0])
    ax.set_xlim(0.45, 1.05)
    ax.set_ylim(0, 1350)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', fontsize=10)


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
