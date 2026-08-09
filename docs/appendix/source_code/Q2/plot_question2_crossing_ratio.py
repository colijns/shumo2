# 本绘图程序在AI工具辅助下完成
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31

"""问题2 附图：跨壁圆柱比例诊断（图8）。

数据来源：Q2/results/solid_boundary_comparison.json（正式运行实测，共
3,960,000 根完整圆柱）与 solid_boundary_sensitivity.json（百万样本单圆柱
投影理论率）。柱高按 Σ(mean_crossing × m) / Σ(n_A × m) 实算，不硬编码；
虚线为理论参考：轴线 60.15%、实体 60.88%。实体口径比轴线多识别约 0.73 个
百分点的跨壁圆柱（主要来自"轴线仍在盒内但半径或端面越界"情形）。
输出：Q2/figures/问题2_跨壁比例_诊断.png + .pdf（300dpi，双格式），
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

COMPARISON_JSON = os.path.join(_Q2, 'results', 'solid_boundary_comparison.json')
SENSITIVITY_JSON = os.path.join(_Q2, 'results', 'solid_boundary_sensitivity.json')
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


def load_data():
    with open(COMPARISON_JSON, encoding='utf-8') as f:
        comp = json.load(f)
    with open(SENSITIVITY_JSON, encoding='utf-8') as f:
        sens = json.load(f)
    rows = sorted(comp['rows'], key=lambda r: r['phi'])
    # 跨壁比例 = Σ(mean_crossing × m) / Σ(n_A × m)，不硬编码
    total = sum(r['n_A'] * r['m'] for r in rows)
    rates = {
        'axis': sum(r['mean_axis_crossing'] * r['m'] for r in rows) / total,
        'inner': sum(r['mean_inner_crossing'] * r['m'] for r in rows) / total,
        'outer': sum(r['mean_outer_crossing'] * r['m'] for r in rows) / total,
    }
    return rows, rates, total, sens


def main():
    rows, rates, total, sens = load_data()
    names = ['轴线截断\n（旧口径）', '内接实体', '外切实体']
    values = [rates['axis'] * 100, rates['inner'] * 100, rates['outer'] * 100]
    colors = ['#7f7f7f', '#1f77b4', '#ff7f0e']
    axis_theory = sens['axis_crossing_rate'] * 100  # 60.15
    solid_theory = sens['solid_crossing_rate'] * 100  # 60.88

    fig, ax = plt.subplots(figsize=(7, 4.4))
    bars = ax.bar(names, values, color=colors, alpha=0.85,
                  width=0.55, label=f'实测（共 {total:,} 根完整圆柱）')
    for bar, v in zip(bars, values):
        ax.annotate(f'{v:.2f}%', (bar.get_x() + bar.get_width() / 2, v),
                    textcoords='offset points', xytext=(0, 6),
                    fontsize=10, ha='center')

    # 理论参考虚线（百万样本单圆柱投影）
    ax.axhline(axis_theory, color='#404040', linestyle='--',
               linewidth=1.6, label='轴线理论率 60.15%')
    ax.axhline(solid_theory, color='#d62728', linestyle='--',
               linewidth=1.6, label='实体理论率 60.88%')
    ax.text(1.5, (axis_theory + solid_theory) / 2 - 0.075,
            '实体 - 轴线 = 0.73 个百分点\n（半径/端面越界，轴线仍在内）',
            fontsize=9, color='#333333', ha='center', va='center')

    ax.set_ylabel('跨壁圆柱比例')
    ax.set_ylim(58, 62)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=10)

    save_pair(fig, '问题2_跨壁比例_诊断')


if __name__ == '__main__':
    main()
