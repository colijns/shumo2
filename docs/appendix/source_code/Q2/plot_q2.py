# -*- coding: utf-8 -*-
# ============================================================
# 本绘图程序在AI工具辅助下完成
# AI工具名称：DeepSeek-V4-Flash（Claude Code）
# 版本：v4-flash（2026-08 使用）
# 开发机构：杭州深度求索人工智能基础技术研究有限公司
# 使用日期：2026-08-18
# 人工修改记录：见 docs/appendix/interaction_logs/edit_trace.md
# ============================================================
"""
问题2 成果图（图6、图7、图8）。

图6：问题1 与问题2 各无人机工作时长对比（2×2，分组柱状图）——
     直观展示负载均衡改善：问题2 各机工作时长更接近。
图7：双目标权衡图（2×2）——横轴总体完成时间 T_max，纵轴负载差 δ，
     箭头表示问题1 解 → 问题2 解的移动方向（δ 大幅下降，T_max 基本不变）。
图8：跨算例对比（单图两子图）——T_max 与 δ 在问题1→问题2 的变化。

数据来源：
- Q1 解档案 outputs/workbooks/q1_solution_Case*.json（uavs[].work_s、Tmax_s/Tmin_s）
- Q2 严格解档案 outputs/workbooks/q2/strict/Case*.json（metrics.route_work_s 等）
输出：Q2/figs/fig6_*.png/.pdf、fig7_*.png/.pdf、fig8_*.png/.pdf

用法（仓库根目录，math 环境）：
    python Q2/plot_q2.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无头环境保存图片
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# 中文字体：须在 style.use 之后再设置，否则被 seaborn 风格覆盖为 Arial
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES = ["Case1", "Case2", "Case3", "Case4"]
Q2_DIR = REPO_ROOT / "Q2"
FIG_DIR = Q2_DIR / "figs"
ARCHIVE = REPO_ROOT / "outputs" / "workbooks"

C_Q1 = "#4C72B0"  # 问题1 蓝
C_Q2 = "#DD8452"  # 问题2 橙


def save_fig(fig, path, dpi=300):
    """保存 png/pdf 双格式。"""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"图片已保存到：{path}（dpi={dpi}）")


def load_q1(case: str) -> dict:
    return json.load(open(ARCHIVE / f"q1_solution_{case}.json", encoding="utf-8"))


def load_q2(case: str) -> dict:
    return json.load(open(ARCHIVE / "q2" / "strict" / f"{case}.json", encoding="utf-8"))


def fig6_workload_compare(q1: dict[str, dict], q2: dict[str, dict]) -> plt.Figure:
    """图6：问题1 vs 问题2 各无人机工作时长对比。"""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, case in zip(axes.flat, CASES):
        a1, a2 = q1[case], q2[case]
        w1 = [u["work_s"] / 3600.0 for u in a1["uavs"]]
        w2 = [s / 3600.0 for s in a2["metrics"]["route_work_s"]]
        n = max(len(w1), len(w2))
        x = np.arange(1, n + 1)
        width = 0.38
        ax.bar(x - width / 2, w1 + [np.nan] * (n - len(w1)), width, color=C_Q1,
               edgecolor="white", label="问题1", zorder=3)
        ax.bar(x + width / 2, w2 + [np.nan] * (n - len(w2)), width, color=C_Q2,
               edgecolor="white", label="问题2", zorder=3)
        ax.axhline(a1["Tmax_h"], color=C_Q1, linestyle="--", linewidth=1.0,
                   alpha=0.7)
        ax.axhline(a2["metrics"]["Tmax_s"] / 3600.0, color=C_Q2, linestyle="--",
                   linewidth=1.0, alpha=0.7)
        d1 = a1["Tmax_s"] - a1["Tmin_s"]
        d2 = a2["metrics"]["delta_s"]
        ax.set_title(f"{case}：δ 由 {d1 / 60:.1f} min 降至 {d2 / 60:.1f} min", fontsize=12)
        ax.set_xlabel("无人机编号", fontsize=10)
        ax.set_ylabel("工作时长（h）", fontsize=10)
        ax.set_xticks(x)
        ax.legend(fontsize=9, loc="lower right")
    fig.suptitle("问题1 与问题2 各无人机工作时长对比（负载均衡改善）",
                 fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def fig7_tradeoff(q1: dict[str, dict], q2: dict[str, dict]) -> plt.Figure:
    """图7：双目标权衡图（Q1 解 → Q2 解移动方向）。"""
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5))
    for ax, case in zip(axes.flat, CASES):
        a1, a2 = q1[case], q2[case]
        t1, d1 = a1["Tmax_h"], (a1["Tmax_s"] - a1["Tmin_s"]) / 60.0
        t2, d2 = a2["metrics"]["Tmax_s"] / 3600.0, a2["metrics"]["delta_s"] / 60.0
        ax.scatter([t1], [d1], s=160, marker="*", color=C_Q1, zorder=5,
                   label=f"问题1（T_max={t1:.2f} h）")
        ax.scatter([t2], [d2], s=120, marker="D", color=C_Q2, zorder=5,
                   label=f"问题2（δ={d2:.0f} min）")
        ax.annotate("", xy=(t2, d2), xytext=(t1, d1),
                    arrowprops=dict(arrowstyle="->", color="#555555",
                                    lw=1.6, connectionstyle="arc3,rad=0.18"))
        ax.set_title(f"{case}", fontsize=12)
        ax.set_xlabel("总体完成时间 T_max（h）", fontsize=10)
        ax.set_ylabel("负载差 δ（min）", fontsize=10)
        ax.legend(fontsize=8.5, loc="center right")
    fig.suptitle("问题1 → 问题2 双目标权衡：以极小 T_max 代价换取负载大幅均衡",
                 fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def fig8_cross_case(q1: dict[str, dict], q2: dict[str, dict]) -> plt.Figure:
    """图8：跨算例 T_max 与 δ 在问题1→问题2 的变化。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    x = np.arange(len(CASES))
    width = 0.35
    t1 = [q1[c]["Tmax_h"] for c in CASES]
    t2 = [q2[c]["metrics"]["Tmax_s"] / 3600.0 for c in CASES]
    d1 = [(q1[c]["Tmax_s"] - q1[c]["Tmin_s"]) / 60.0 for c in CASES]
    d2 = [q2[c]["metrics"]["delta_s"] / 60.0 for c in CASES]

    b1 = ax1.bar(x - width / 2, t1, width, color=C_Q1, edgecolor="white",
                 label="问题1", zorder=3)
    b2 = ax1.bar(x + width / 2, t2, width, color=C_Q2, edgecolor="white",
                 label="问题2", zorder=3)
    for bars in (b1, b2):
        for b in bars:
            ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05,
                     f"{b.get_height():.2f}", ha="center", fontsize=9)
    ax1.set_ylabel("总体完成时间 T_max（h）", fontsize=11)
    ax1.set_title("跨算例对比：T_max 与负载差 δ（问题1 → 问题2）", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)

    b3 = ax2.bar(x - width / 2, d1, width, color=C_Q1, edgecolor="white",
                 label="问题1", zorder=3)
    b4 = ax2.bar(x + width / 2, d2, width, color=C_Q2, edgecolor="white",
                 label="问题2", zorder=3)
    for bars in (b3, b4):
        for b in bars:
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                     f"{b.get_height():.0f}", ha="center", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(CASES)
    ax2.set_ylabel("负载差 δ（min）", fontsize=11)
    ax2.set_xlabel("测试算例", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    q1 = {c: load_q1(c) for c in CASES}
    q2 = {c: load_q2(c) for c in CASES}

    for fn, name in [(fig6_workload_compare, "fig6_问题1vs2工作时长对比"),
                     (fig7_tradeoff, "fig7_问题2双目标权衡"),
                     (fig8_cross_case, "fig8_跨算例Tmax与δ对比")]:
        fig = fn(q1, q2)
        for ext in ("png", "pdf"):
            save_fig(fig, FIG_DIR / f"{name}.{ext}", dpi=300)
        plt.close(fig)
        print(f"完成：{name}")


if __name__ == "__main__":
    main()
