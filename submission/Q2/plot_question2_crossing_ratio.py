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

COMPARISON_JSON = os.path.join(_Q2, 'results', 'solid_boundary_comparison.json')
SENSITIVITY_JSON = os.path.join(_Q2, 'results', 'solid_boundary_sensitivity.json')
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


def load_data():
    with open(COMPARISON_JSON, encoding='utf-8') as f:
        comp = json.load(f)
    with open(SENSITIVITY_JSON, encoding='utf-8') as f:
        sens = json.load(f)
    rows = sorted(comp['rows'], key=lambda r: r['phi'])

    total = sum(r['n_A'] * r['m'] for r in rows)
    rates = {
        'axis': sum(r['mean_axis_crossing'] * r['m'] for r in rows) / total,
        'inner': sum(r['mean_inner_crossing'] * r['m'] for r in rows) / total,
        'outer': sum(r['mean_outer_crossing'] * r['m'] for r in rows) / total,
    }
    return rows, rates, total, sens


def main():
    rows, rates, total, sens = load_data()
    names = ['轴线截断', '内接实体', '外切实体']
    values = [rates['axis'] * 100, rates['inner'] * 100, rates['outer'] * 100]
    colors = ['#7f7f7f', '#1f77b4', '#ff7f0e']
    axis_theory = sens['axis_crossing_rate'] * 100
    solid_theory = sens['solid_crossing_rate'] * 100

    fig, ax = plt.subplots(figsize=(7, 4.4))
    bars = ax.bar(names, values, color=colors, alpha=0.85,
                  width=0.55, label=f'实测（共 {total:,} 根完整圆柱）')
    for bar, v in zip(bars, values):
        ax.annotate(f'{v:.2f}%', (bar.get_x() + bar.get_width() / 2, v),
                    textcoords='offset points', xytext=(0, 6),
                    fontsize=10, ha='center')


    ax.axhline(axis_theory, color='#404040', linestyle='--',
               linewidth=1.6, label='轴线理论率 60.15%')
    ax.axhline(solid_theory, color='#d62728', linestyle='--',
               linewidth=1.6, label='实体理论率 60.88%')

    ax.set_ylabel('跨壁圆柱比例（%）')
    ax.set_ylim(58, 62)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=10)

    save_pair(fig, '问题2_跨壁比例_诊断')


if __name__ == '__main__':
    main()
