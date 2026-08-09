# 本绘图程序在AI工具辅助下完成
# AI工具名称：DeepSeek-V4-Flash，版本 / 型号：DeepSeek-V4-Flash-0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026-07-31

"""问题3配图：边界口径敏感性对照（图4）。

两种边界口径的临界值对照：(a) 临界介质数量（根）、(b) 临界体积分数（pp，
对数轴）。主口径（同源片段独立判定，M=10000 正式运行）：外切点估计 613、
内接点估计 616、内接保守 619，约 0.87%；反面口径（同源片段自动电连续，
假设一验证运行 M=2000）：8 根、约 0.0113 pp——对数坐标下相差两个数量级。
数据来源：Q3/results/question3_result_hypothesis1.csv（无 BOM，utf-8）+
question3_solid_curve.csv 与 question3_solid_summary.json。
输出：Q3/figures/问题3_边界口径敏感性_临界对照.png + .pdf（300dpi，
双格式），并复制入 docs/appendix/figures/ 图池（competition-record 步骤 2+4）。
"""

import csv
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

_Q3 = os.path.dirname(os.path.abspath(__file__))
# 与模板 plot_style 同款中文样式（内联，避免依赖 templates 路径）
sys_fonts = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['font.sans-serif'] = sys_fonts
plt.rcParams['axes.unicode_minus'] = False

CURVE_CSV = os.path.join(_Q3, 'results', 'question3_solid_curve.csv')
SUMMARY_JSON = os.path.join(_Q3, 'results', 'question3_solid_summary.json')
HYP1_CSV = os.path.join(_Q3, 'results', 'question3_result_hypothesis1.csv')
FIG_DIR = os.path.join(_Q3, 'figures')
APPENDIX_FIG_DIR = os.path.join(os.path.dirname(_Q3), 'docs', 'appendix', 'figures')

N_POINT_OUTER = 613
N_POINT_INNER = 616
N_SAFE = 619


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


def load_phi_at(n_list):
    """从曲线取指定 n 的体积分数（pp）。"""
    with open(CURVE_CSV, encoding='utf-8-sig') as f:
        rows = {int(r['n']): float(r['phi_percent']) for r in csv.DictReader(f)}
    return [rows[n] for n in n_list]


def load_hyp1():
    with open(HYP1_CSV, encoding='utf-8') as f:
        row = next(csv.DictReader(f))
    return {
        'n_hat': int(row['n_hat_90']),
        'phi_pp': float(row['phi_hat_90_pp']),
        'm': int(row['m']),
        'n_max': int(row['n_max']),
    }


def main():
    hyp1 = load_hyp1()
    assert hyp1['n_hat'] == 8, f'假设一 n_hat={hyp1["n_hat"]}'
    assert abs(hyp1['phi_pp'] - 0.0113) < 1e-4, f'假设一 phi={hyp1["phi_pp"]}'
    phi_613, phi_616, phi_619 = load_phi_at(
        [N_POINT_OUTER, N_POINT_INNER, N_SAFE])
    assert abs(phi_613 - 0.866608) < 1e-5, f'phi(613)={phi_613}'
    assert abs(phi_616 - 0.870849) < 1e-5, f'phi(616)={phi_616}'
    assert abs(phi_619 - 0.875091) < 1e-5, f'phi(619)={phi_619}'
    print(f'断言通过：phi(613/616/619)='
          f'{phi_613:.6f}/{phi_616:.6f}/{phi_619:.6f} pp，'
          f'假设一 {hyp1["n_hat"]} 根 / {hyp1["phi_pp"]:.6f} pp '
          f'（M={hyp1["m"]}）')

    labels = ['外切点估计', '内接点估计', '内接保守', '假设一\n（自动电连续）']
    counts = [N_POINT_OUTER, N_POINT_INNER, N_SAFE, hyp1['n_hat']]
    phis = [phi_613, phi_616, phi_619, hyp1['phi_pp']]
    colors = ['#1f77b4', '#2ca02c', '#006400', '#ff7f0e']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    bars1 = ax1.bar(range(4), counts, color=colors, width=0.6)
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel('临界介质数量 N（根）')
    ax1.set_ylim(0, 700)
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)

    bars2 = ax2.bar(range(4), phis, color=colors, width=0.6)
    ax2.set_yscale('log')
    ax2.set_yticks([0.01, 0.1, 1.0])
    ax2.set_yticklabels(['0.01', '0.1', '1'])
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel('临界体积分数 $\\phi_A$（pp，对数轴）')
    ax2.set_ylim(0.005, 2.0)
    ax2.axhline(0.0113, color='#ff7f0e', linewidth=1.2, linestyle=':')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)

    fig.tight_layout()
    save_pair(fig, '问题3_边界口径敏感性_临界对照')


if __name__ == '__main__':
    main()
