# 本绘图程序在AI工具辅助下完成
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31

"""问题2 配图：介质A体积分数 vs 导通概率（95% Wilson 误差棒）。

数据来源：Q2/results/question2_result.csv（蒙特卡洛 M=2000，seed=42+i）。
输出：Q2/figures/问题2_导通概率_vs_体积分数.png + .pdf（300dpi，双格式）。
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

_Q2 = os.path.dirname(os.path.abspath(__file__))
# 与模板 plot_style 同款中文样式（内联，避免依赖 templates 路径）
sys_fonts = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['font.sans-serif'] = sys_fonts
plt.rcParams['axes.unicode_minus'] = False

RESULT_CSV = os.path.join(_Q2, 'results', 'question2_result.csv')
FIG_DIR = os.path.join(_Q2, 'figures')


def load_results():
    rows = []
    with open(RESULT_CSV, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def main():
    rows = load_results()
    rows.sort(key=lambda r: r['phi'])
    phis = np.array([r['phi'] for r in rows])
    p_hat = np.array([r['p_hat'] for r in rows])
    lo = np.array([r['ci_lower'] for r in rows])
    hi = np.array([r['ci_upper'] for r in rows])
    n_a = [int(r['n_A']) for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.errorbar(phis * 100, p_hat, yerr=[p_hat - lo, hi - p_hat],
                fmt='o-', color='#1f77b4', capsize=4, linewidth=1.8,
                markersize=6, label='导通概率（95% Wilson CI 误差棒）')
    ax.set_xlabel('体积分数 φ（%）')
    ax.set_ylabel('导通概率')
    ax.set_title('问题2：介质A体积分数与导通概率（蒙特卡洛 M=2000）')
    ax.set_xlim(0.45, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, linestyle='--', alpha=0.5)
    for x, y, n in zip(phis * 100, p_hat, n_a):
        ax.annotate(f'N$_A$={n}', (x, y), textcoords='offset points',
                    xytext=(0, 10), fontsize=9, ha='center')
    ax.legend(loc='upper left')

    os.makedirs(FIG_DIR, exist_ok=True)
    base = os.path.join(FIG_DIR, '问题2_导通概率_vs_体积分数')
    for ext in ('png', 'pdf'):
        fig.savefig(base + '.' + ext, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'图片已保存：{base}.png / {base}.pdf（dpi=300）')


if __name__ == '__main__':
    main()
