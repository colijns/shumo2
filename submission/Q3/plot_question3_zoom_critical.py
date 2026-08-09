# 本绘图程序在AI工具辅助下完成
# AI工具名称：DeepSeek-V4-Flash，版本 / 型号：DeepSeek-V4-Flash-0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026-07-31

"""问题3配图：临界区放大图（图2）。

N=600~640 区间放大视图：外切（蓝实线，几何下界）与内接（绿虚线，几何上界）
导通概率曲线及 95% Wilson 区间带，90% 要求线，[613, 616] 经验夹逼底纹，
三条临界竖线（613 外切点估计 / 616 内接点估计与外切保守 / 619 内接保守）。
上轴体积分数刻度由 CSV 行直接对应（不插值）。
数据来源：Q3/results/question3_solid_curve.csv（M=10000，K=32，
seed=20260808+i）+ question3_solid_summary.json。
输出：Q3/figures/问题3_临界区放大图.png + .pdf（300dpi，双格式），
并复制入 docs/appendix/figures/ 图池（competition-record 步骤 2+4）。
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
FIG_DIR = os.path.join(_Q3, 'figures')
APPENDIX_FIG_DIR = os.path.join(os.path.dirname(_Q3), 'docs', 'appendix', 'figures')

N_LO, N_HI = 600, 640
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


def load_curve():
    rows = {}
    with open(CURVE_CSV, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            rows[int(r['n'])] = {k: float(v) for k, v in r.items()}
    return rows


def load_summary():
    with open(SUMMARY_JSON, encoding='utf-8') as f:
        return json.load(f)


def main():
    curve = load_curve()
    summary = load_summary()
    trials = int(summary['config']['trials'])
    assert trials == 10000, f'预期 M=10000，汇总为 M={trials}'

    n_vals = np.arange(N_LO, N_HI + 1)
    phi = np.array([curve[n]['phi_percent'] for n in n_vals])
    outer_p = np.array([curve[n]['outer_p_hat'] for n in n_vals])
    outer_lo = np.array([curve[n]['outer_ci_lower'] for n in n_vals])
    outer_hi = np.array([curve[n]['outer_ci_upper'] for n in n_vals])
    inner_p = np.array([curve[n]['inner_p_hat'] for n in n_vals])
    inner_lo = np.array([curve[n]['inner_ci_lower'] for n in n_vals])
    inner_hi = np.array([curve[n]['inner_ci_upper'] for n in n_vals])

    # 数值断言（与报告 §4.2 表一致）
    assert abs(curve[N_POINT_OUTER]['outer_p_hat'] - 0.9009) < 1e-4, \
        f"p_hat(613)={curve[N_POINT_OUTER]['outer_p_hat']}"
    assert abs(curve[N_POINT_INNER]['inner_p_hat'] - 0.9005) < 1e-4, \
        f"p_hat(616)={curve[N_POINT_INNER]['inner_p_hat']}"
    assert abs(curve[N_SAFE]['inner_ci_lower'] - 0.9000) < 1e-4, \
        f"P_L(619)={curve[N_SAFE]['inner_ci_lower']}"
    print('数值断言通过：p_hat(613)=0.9009, p_hat(616)=0.9005, P_L(619)=0.9000')

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.fill_between(n_vals, outer_lo, outer_hi, color='#1f77b4', alpha=0.15)
    ax.fill_between(n_vals, inner_lo, inner_hi, color='#2ca02c', alpha=0.15)
    ax.plot(n_vals, outer_p, color='#1f77b4', linewidth=2,
            label='外切多棱柱 $\\widehat P$（几何下界）')
    ax.plot(n_vals, inner_p, color='#2ca02c', linewidth=1.8, linestyle='--',
            label='内接多棱柱 $\\widehat P$（几何上界）')
    ax.axhline(0.90, color='red', linestyle=':', linewidth=1.6,
               label='90% 要求线')
    ax.axvspan(N_POINT_OUTER, N_POINT_INNER, color='gray', alpha=0.15,
               label='经验夹逼区间 [613, 616]')

    ax.axvline(N_POINT_OUTER, color='#1f77b4', linewidth=1.4)
    ax.axvline(N_POINT_INNER, color='#2ca02c', linewidth=1.4)
    ax.axvline(N_SAFE, color='#2ca02c', linewidth=1.4, linestyle='--')

    ax.set_xlabel('完整介质数量 N（根）')
    ax.set_ylabel('导通概率')
    ax.set_xlim(N_LO, N_HI)
    ax.set_ylim(0.86, 0.93)
    ax.set_xticks(np.arange(600, 641, 5))
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=9)

    # 上轴体积分数：与 N 行一一对应（取每 5 根一个刻度）
    ax_top = ax.twiny()
    tick_ns = np.arange(N_LO, N_HI + 1, 10)
    tick_phis = [curve[n]['phi_percent'] for n in tick_ns]
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(tick_ns)
    ax_top.set_xticklabels([f'{p:.3f}' for p in tick_phis], fontsize=9)
    ax_top.set_xlabel('体积分数 $\\phi_A$（pp）')

    save_pair(fig, '问题3_临界区放大图')


if __name__ == '__main__':
    main()
