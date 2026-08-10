# 本程序及代码是在AI工具辅助下完成的
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31

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
