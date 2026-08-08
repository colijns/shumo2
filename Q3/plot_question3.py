# 本程序及代码是在AI工具辅助下完成的
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31

"""问题3 论文配图：首次导通数量 → 导通概率曲线（png+pdf 双格式）。

- 图1 主图：P̂(N) 经验 CDF + Wilson 区间带 + 0.90 参考线 +
  N90_hat / Nsafe 标注；x 轴双刻度（圆柱数量 N + 体积分数 φ%）。
- 图2 敏感性对照：假设一（同源电连续）vs 假设二（片段独立）。

读 Q3/results/*.csv，输出 Q3/figures/（png+pdf）。
"""

import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei',
                                   'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Q2'))

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
CURVE_H2 = os.path.join(OUT_DIR, 'question3_curve.csv')
CURVE_H1 = os.path.join(OUT_DIR, 'question3_curve_hypothesis1.csv')
RESULT_H2 = os.path.join(OUT_DIR, 'question3_result.csv')
RESULT_H1 = os.path.join(OUT_DIR, 'question3_result_hypothesis1.csv')

P_TARGET = 0.90


def load_curve(path):
    """读曲线 csv → (N, phi_pp, p_hat, lo, hi)。"""
    N, phi, p, lo, hi = [], [], [], [], []
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            N.append(int(row['N']))
            phi.append(float(row['phi_pp']))
            p.append(float(row['p_hat']))
            lo.append(float(row['ci_lower']))
            hi.append(float(row['ci_upper']))
    return N, phi, p, lo, hi


def load_result(path):
    with open(path, encoding='utf-8') as f:
        return next(csv.DictReader(f))


def plot_main():
    """图1：导通概率 vs N（双 x 轴），标注两个临界点。"""
    N, phi, p, lo, hi = load_curve(CURVE_H2)
    res = load_result(RESULT_H2)
    n_hat = int(res['n_hat_90'])
    n_safe = int(res['n_safe']) if res['n_safe'] else None

    fig, ax = plt.subplots(figsize=(8.4, 5.6), dpi=150)
    ax.fill_between(N, lo, hi, color='#4C72B0', alpha=0.15,
                    label='95% Wilson 区间')
    ax.plot(N, p, color='#4C72B0', lw=2.0, label='经验 CDF $\\hat{P}(N)$')
    ax.axhline(P_TARGET, color='#C44E52', ls='--', lw=1.4,
               label='90% 导通要求')
    ax.axvline(n_hat, color='#55A868', ls=':', lw=1.6,
               label=f'$\\hat{{N}}_{{90}}$ = {n_hat}')
    if n_safe:
        ax.axvline(n_safe, color='#8172B3', ls=':', lw=1.6,
                   label=f'$N_\\mathrm{{safe}}$ = {n_safe}')
    ax.set_xlabel('完整介质数量 $N$（根）')
    ax.set_ylabel('导通概率 $P(Y=1\\mid N)$')
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(0, max(N))
    ax.legend(loc='lower right', fontsize=9)

    # 上轴：体积分数刻度
    ax2 = ax.twiny()
    ax2.set_xlim(0, max(N))
    ticks = [0, 200, 400, 600, 800, 900]
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f'{t * 1.4137e-5 * 100:.2f}' for t in ticks])
    ax2.set_xlabel('体积分数 $\\varphi_A$（%，单根步长 $\\Delta\\varphi_A'
                   '\\approx 0.00141$ pp）')
    ax.grid(alpha=0.25)

    title = ('问题3：导通概率随介质数量变化（假设二：片段独立接触）')
    if n_hat:
        phi_hat = float(res['phi_hat_90_pp'])
        title += f'\n$\\hat\\varphi_A^*$ = {phi_hat:.2f}%'
    if n_safe:
        title += f'，$\\varphi_{{A,\\mathrm{{safe}}}}$ = {float(res["phi_safe_pp"]):.2f}%'
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG_DIR, f'问题3_导通概率_vs_介质数量.{ext}'))
    plt.close(fig)
    print('图1 已输出 png+pdf')


def plot_sensitivity():
    """图2：两种边界口径对照（假设一退化 vs 假设二非退化）。"""
    N1, _, p1, _, _ = load_curve(CURVE_H1)
    N2, _, p2, _, _ = load_curve(CURVE_H2)

    fig, ax = plt.subplots(figsize=(8.4, 5.6), dpi=150)
    ax.plot(N1, p1, color='#C44E52', lw=2.0, ls='-',
            label='假设一：同源片段自动电连续（退化）')
    ax.plot(N2, p2, color='#4C72B0', lw=2.0, ls='-',
            label='假设二：平移片段按实际位置判定')
    ax.axhline(P_TARGET, color='#333333', ls='--', lw=1.2,
               label='90% 导通要求')
    ax.set_xlabel('完整介质数量 $N$（根）')
    ax.set_ylabel('导通概率 $P(Y=1\\mid N)$')
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(0, max(N2))
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_title('问题3：边界口径敏感性对照', fontsize=12)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG_DIR, f'问题3_边界口径敏感性.{ext}'))
    plt.close(fig)
    print('图2 已输出 png+pdf')


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    plot_main()
    if os.path.exists(CURVE_H1):
        plot_sensitivity()
    else:
        print('敏感性曲线不存在，跳过图2')


if __name__ == '__main__':
    main()
