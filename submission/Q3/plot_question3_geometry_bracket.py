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

_Q3 = os.path.dirname(os.path.abspath(__file__))
sys_fonts = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['font.sans-serif'] = sys_fonts
plt.rcParams['axes.unicode_minus'] = False

SUMMARY_JSON = os.path.join(_Q3, 'results', 'question3_solid_summary.json')
FIG_DIR = os.path.join(_Q3, 'figures')
APPENDIX_FIG_DIR = os.path.join(os.path.dirname(_Q3), 'docs', 'appendix', 'figures')

R_A = 30.0
DELTA = 1.8


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


def load_n_sides():
    with open(SUMMARY_JSON, encoding='utf-8') as f:
        payload = json.load(f)
    return int(payload['config']['n_sides'])


def polygon_vertices(radius, k, phase):
    angles = 2.0 * np.pi * np.arange(k) / k + phase
    return radius * np.column_stack((np.cos(angles), np.sin(angles)))


def main():
    k = load_n_sides()
    assert k == 32, f'预期 K=32，汇总文件为 K={k}'


    inner_v = polygon_vertices(R_A, k, np.pi / k)

    outer_v = polygon_vertices(R_A / np.cos(np.pi / k), k, np.pi / k)
    circle = R_A * np.column_stack((
        np.cos(np.linspace(0, 2 * np.pi, 400)),
        np.sin(np.linspace(0, 2 * np.pi, 400))))

    eps_plus = R_A * (1.0 / np.cos(np.pi / k) - 1.0)
    eps_minus = R_A * (1.0 - np.cos(np.pi / k))
    eps = max(eps_plus, eps_minus)
    print(f'K={k}  eps_plus={eps_plus:.4f}  eps_minus={eps_minus:.4f}  '
          f'eps={eps:.4f} nm  阈值之比={eps / DELTA:.3f}')


    k_demo = 8
    inner_v8 = polygon_vertices(R_A, k_demo, np.pi / k_demo)
    outer_v8 = polygon_vertices(R_A / np.cos(np.pi / k_demo),
                                k_demo, np.pi / k_demo)
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(12.0, 5.6),
                                  gridspec_kw={'width_ratios': [1.35, 1]})
    ax.set_aspect('equal')
    ax.plot(circle[:, 0], circle[:, 1], color='#1f77b4', linewidth=2,
            label='真实圆柱横截面（半径 30 nm）')
    inner8_closed = np.vstack((inner_v8, inner_v8[0]))
    outer8_closed = np.vstack((outer_v8, outer_v8[0]))
    ax.plot(inner8_closed[:, 0], inner8_closed[:, 1], color='#2ca02c',
            linewidth=1.8, linestyle='--',
            label='内接正多边形（几何下界）')
    ax.plot(outer8_closed[:, 0], outer8_closed[:, 1], color='#ff7f0e',
            linewidth=1.8, linestyle='--',
            label='外切正多边形（几何上界）')



    mid_angle = np.pi / k_demo
    eps8_plus = R_A * (1.0 / np.cos(mid_angle) - 1.0)
    eps8_minus = R_A * (1.0 - np.cos(mid_angle))
    ax.annotate('', xy=(R_A + eps8_plus) * np.array(
                    [np.cos(mid_angle), np.sin(mid_angle)]),
                xytext=R_A * np.array([np.cos(mid_angle), np.sin(mid_angle)]),
                arrowprops=dict(arrowstyle='<->', color='#d62728',
                                linewidth=2.0))
    inner_mid8 = R_A * np.cos(mid_angle) * np.array([1.0, 0.0])
    ax.annotate('', xy=(R_A, 0), xytext=inner_mid8,
                arrowprops=dict(arrowstyle='<->', color='#2ca02c',
                                linewidth=2.0))
    ax.set_xlabel('横截面坐标 x（nm）')
    ax.set_ylabel('横截面坐标 y（nm）')
    ax.set_xlim(-33, 33)
    ax.set_ylim(-33, 33)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=9)


    eps32 = max(eps_plus, eps_minus)
    axb.barh([0], [eps32], color='#d62728', height=0.45,
             edgecolor='#8c1d18', linewidth=0.6)
    axb.barh([1], [DELTA], color='#7f7f7f', height=0.45)
    axb.axvline(eps32, color='#d62728', linewidth=1.0, linestyle=':')
    axb.axvline(DELTA, color='#7f7f7f', linewidth=1.0, linestyle=':')
    axb.set_yticks([0, 1])
    axb.set_yticklabels(['$\\varepsilon(32)$', '阈值 $\\delta$'], fontsize=11)
    axb.set_xlabel('径向距离（nm）')
    axb.set_xlim(0, 2.0)
    axb.set_ylim(-0.7, 1.7)
    axb.grid(True, axis='x', linestyle='--', alpha=0.5)

    save_pair(fig, '问题3_实体夹逼几何示意')


if __name__ == '__main__':
    main()
