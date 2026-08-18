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
问题2 成果图（图6、图7）。

图6：Case1 问题1 → 问题2 单机工作时长调整细节（单图，点线连接图）——
     问题2 的单机路径调整仅发生在 Case1（UAV1 −4.8 min、UAV2 +14.3 min），
     单图放大呈现：蓝圆点=问题1、橙菱形=问题2、灰线=同机调整量 Δ（min）。
图7：Case1 双目标权衡图（单图）——上：T_max-δ 权衡平面（0 起点，
     大点 + 粗箭头示移动方向）；
     下：T_max 与 δ 各一行（行内独立刻度）的变化前后值对比——问题1 蓝圆点
     → 问题2 橙菱形，δ 行箭头示下降（27.8→13.5 min），行内标注相对变化率
     （T_max +0.0% / δ −51.6%），量化"以零总耗时代价换取负载大幅均衡"。

数据来源：
- Q1 解档案 outputs/workbooks/q1_solution_Case*.json（uavs[].work_s、Tmax_s/Tmin_s）
- Q2 严格解档案 outputs/workbooks/q2/strict/Case*.json（metrics.route_work_s 等）
输出：Q2/figs/fig6_*.png/.pdf、fig7_*.png/.pdf

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
    """图6：Case1 问题1 → 问题2 单机工作时长调整细节（单图点线连接图）。

    问题1→问题2 的单机路径调整仅出现在 Case1（UAV1 −4.8 min、UAV2
    +14.3 min），单图放大呈现：每机蓝圆点=问题1、橙菱形=问题2、灰线=
    同机调整、线旁标注 Δ（min，正值=变长）。
    """
    case = "Case1"
    a1, a2 = q1[case], q2[case]
    w1 = [u["work_s"] / 3600.0 for u in a1["uavs"]]
    w2 = [s / 3600.0 for s in a2["metrics"]["route_work_s"]]
    n = len(w1)
    x = np.arange(1, n + 1)
    dx = 0.13
    fig, ax = plt.subplots(figsize=(8.5, 5))
    # 同机连线（先画，置于底层）
    for xi, v1, v2 in zip(x, w1, w2):
        ax.plot([xi - dx, xi + dx], [v1, v2], color="#999999",
                linewidth=1.2, zorder=1, alpha=0.85)
    ax.scatter(x - dx, w1, s=90, marker="o", color=C_Q1, zorder=3,
               edgecolors="white", linewidths=0.8, label="问题1")
    ax.scatter(x + dx, w2, s=90, marker="D", color=C_Q2, zorder=3,
               edgecolors="white", linewidths=0.8, label="问题2")
    # 调整量标注（min，带符号）
    for xi, v1, v2 in zip(x, w1, w2):
        d = (v2 - v1) * 60.0
        top = max(v1, v2)
        ax.annotate(f"Δ {d:+.1f} min", xy=(xi, top), xytext=(xi, top + 0.10),
                    ha="center", fontsize=10, color="#333333")
    ymax = max(w1 + w2 + [a1["Tmax_h"], a2["metrics"]["Tmax_s"] / 3600.0])
    ax.axhline(a1["Tmax_h"], color=C_Q1, linestyle="--", linewidth=1.0,
               alpha=0.7, label=f"T_max(问题1)={a1['Tmax_h']:.2f} h")
    ax.axhline(a2["metrics"]["Tmax_s"] / 3600.0, color=C_Q2, linestyle="--",
               linewidth=1.0, alpha=0.7,
               label=f"T_max(问题2)={a2['metrics']['Tmax_s'] / 3600.0:.2f} h")
    ax.set_xlabel("无人机编号", fontsize=11)
    ax.set_ylabel("工作时长（h）", fontsize=11)
    ax.set_xticks(x)
    ax.set_ylim(0, ymax * 1.15)
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.015, 0.98, case, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left", zorder=6,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2))
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    return fig


def fig7_tradeoff(q1: dict[str, dict], q2: dict[str, dict]) -> plt.Figure:
    """图7：Case1 双目标权衡图（上：权衡平面；下行两行：变化前后值对比）。

    上子图：T_max-δ 权衡平面（坐标从 0 起，保持尺度诚实）——问题1 蓝星、
    问题2 橙菱形，粗箭头示移动方向；
    下行两行：T_max 与 δ 各一行（行内独立刻度、指标作 x 轴单位）——
    问题1 蓝圆点 → 问题2 橙菱形，δ 行灰箭头示下降（27.8→13.5 min），
    行内标注相对变化率（T_max +0.0% / δ −51.6%），量化"以零总耗时代价
    换取负载大幅均衡"。
    """
    case = "Case1"
    a1, a2 = q1[case], q2[case]
    t1, d1 = a1["Tmax_h"], (a1["Tmax_s"] - a1["Tmin_s"]) / 60.0
    t2, d2 = a2["metrics"]["Tmax_s"] / 3600.0, a2["metrics"]["delta_s"] / 60.0
    pct_t = (t2 - t1) / t1 * 100.0
    pct_d = (d2 - d1) / d1 * 100.0

    fig = plt.figure(figsize=(8.5, 8.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 0.6, 0.6], hspace=0.55)
    ax1 = fig.add_subplot(gs[0])

    # ---- 上子图：双目标权衡平面 ----
    ax1.scatter([t1], [d1], s=220, marker="*", color=C_Q1, zorder=5,
                label=f"问题1（T_max={t1:.2f} h，δ={d1:.0f} min）")
    ax1.scatter([t2], [d2], s=180, marker="D", color=C_Q2, zorder=5,
                label=f"问题2（T_max={t2:.2f} h，δ={d2:.0f} min）")
    ax1.annotate("", xy=(t2, d2), xytext=(t1, d1),
                 arrowprops=dict(arrowstyle="->,head_length=0.6,head_width=0.4",
                                 color="#555555", lw=3.2,
                                 connectionstyle="arc3,rad=0.15"))
    ax1.set_xlabel("总体完成时间 T_max（h）", fontsize=11)
    ax1.set_ylabel("负载差 δ（min）", fontsize=11)
    ax1.set_xlim(0, t2 * 1.15)
    ax1.set_ylim(0, d1 * 1.35)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc="lower right", labelspacing=1.5)
    ax1.text(0.015, 0.98, case, transform=ax1.transAxes, fontsize=12,
             fontweight="bold", va="top", ha="left", zorder=6,
             bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2))

    # ---- 下行两行：两指标变化前后值对比（替代变化率双柱）----
    axb = fig.add_subplot(gs[1])
    axc = fig.add_subplot(gs[2])
    for ax, (name, v1, v2, pct) in zip(
            (axb, axc),
            [("T_max（h）", t1, t2, pct_t), ("δ（min）", d1, d2, pct_d)]):
        same = abs(v2 - v1) < 1e-9
        ax.scatter([v1], [0.0], s=90, marker="o", color=C_Q1, zorder=3,
                   edgecolors="white", linewidths=0.8)
        ax.scatter([v2], [0.0], s=85, marker="D", color=C_Q2, zorder=3,
                   edgecolors="white", linewidths=0.8)
        if not same:
            ax.annotate("", xy=(v2, 0.0), xytext=(v1, 0.0),
                        arrowprops=dict(arrowstyle="->,head_length=0.4,head_width=0.3",
                                        color="#555555", lw=2.4,
                                        shrinkA=8, shrinkB=8))
        ax.text((v1 + v2) / 2.0, 0.24, f"{pct:+.1f}%", ha="center",
                fontsize=11, fontweight="bold", color="#333333")
        if same:
            ax.text(v1, 0.06, f"{v1:.2f} h", ha="center", fontsize=9,
                    color="#555555")
        else:
            ax.text(v1, 0.06, f"{v1:.1f}", ha="center", fontsize=9,
                    color="#4C72B0")
            ax.text(v2, 0.06, f"{v2:.1f}", ha="center", fontsize=9,
                    color="#DD8452")
        ax.set_yticks([])
        ax.set_ylim(-0.45, 0.45)
        ax.set_xlim(0, max(v1, v2) * 1.28)
        ax.set_xlabel(name, fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    # GridSpec 手动布局下 tight_layout 不可用，用 subplots_adjust 固定边距
    fig.subplots_adjust(left=0.10, right=0.96, top=0.96, bottom=0.05, hspace=0.55)
    return fig


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    q1 = {c: load_q1(c) for c in CASES}
    q2 = {c: load_q2(c) for c in CASES}

    for fn, name in [(fig6_workload_compare, "fig6_问题1vs2工作时长对比"),
                     (fig7_tradeoff, "fig7_问题2双目标权衡")]:
        fig = fn(q1, q2)
        for ext in ("png", "pdf"):
            save_fig(fig, FIG_DIR / f"{name}.{ext}", dpi=300)
        plt.close(fig)
        print(f"完成：{name}")


if __name__ == "__main__":
    main()
