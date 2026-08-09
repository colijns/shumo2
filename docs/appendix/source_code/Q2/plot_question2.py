# 本绘图程序在AI工具辅助下完成
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31

"""问题2 配图：介质A体积分数 vs 导通概率（完整实体内外接夹逼，图4）。

数据来源：Q2/results/solid_boundary_comparison.json（正式版，M=2000/体积分数，
seed=20260808，K=64，内外接共用同一批随机构型）。内接正多棱柱为导通概率
下界（蓝圆点），外切正多棱柱为上界（橙方点），两者之间灰色阴影带为夹逼区间，
误差棒为 95% Wilson 区间。旧轴线口径 question2_result.csv 已弃用（见报告 6 节）。
输出：Q2/figures/问题2_导通概率_vs_体积分数.png + .pdf（300dpi，双格式），
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


def main():
    rows = load_results()
    phis = np.array([r['phi'] for r in rows])
    n_a = [int(r['n_A']) for r in rows]
    inner_p = np.array([r['inner']['p_hat'] for r in rows])
    inner_lo = np.array([r['inner']['ci_lower'] for r in rows])
    inner_hi = np.array([r['inner']['ci_upper'] for r in rows])
    outer_p = np.array([r['outer']['p_hat'] for r in rows])
    outer_lo = np.array([r['outer']['ci_lower'] for r in rows])
    outer_hi = np.array([r['outer']['ci_upper'] for r in rows])

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.fill_between(phis * 100, inner_p, outer_p, color='gray', alpha=0.25,
                    label='内外接夹逼区间')
    ax.errorbar(phis * 100, inner_p, yerr=[inner_p - inner_lo, inner_hi - inner_p],
                fmt='o-', color='#1f77b4', capsize=4, linewidth=1.8,
                markersize=6, label='内接正多棱柱（导通概率下界）')
    ax.errorbar(phis * 100, outer_p, yerr=[outer_p - outer_lo, outer_hi - outer_p],
                fmt='s-', color='#ff7f0e', capsize=4, linewidth=1.8,
                markersize=6, label='外切正多棱柱（导通概率上界）')
    ax.set_xlabel('体积分数 φ（%）')
    ax.set_ylabel('导通概率')
    ax.set_xlim(0.45, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, linestyle='--', alpha=0.5)
    for x, y, n in zip(phis * 100, outer_p, n_a):
        ax.annotate(f'N$_A$={n}', (x, y), textcoords='offset points',
                    xytext=(0, 10), fontsize=9, ha='center')
    ax.legend(loc='upper left')

    save_pair(fig, '问题2_导通概率_vs_体积分数')


if __name__ == '__main__':
    main()
