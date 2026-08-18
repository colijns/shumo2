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

图6：Case1 单机工作时长调整细节（单图，纵轴无人机按问题1 工时升序）——
     任务重分配机制展示：最短路线 UAV2 工时抬升（+17.3 min）、UAV4 换出
     部分任务（−3.0 min）、UAV1/UAV3 完全不动（灰色弱化），均衡调整仅
     发生在 UAV2 与 UAV4 之间；蓝圆点=问题1、橙菱形=问题2、灰线=同机
     调整量 Δ（min）；点旁标注 4 位小数时长（h，与表3 口径一致）；
     两条 T_max 虚线完全重合，直观传递"最长工时未劣化"。
图7：Case1 双目标权衡图（单图）——上：T_max-δ 权衡平面局部放大
     （x=δ min 0~30、y=T_max 基线±0.1 h，0 起点尺度下两点挤在角落的
     趋势在此可见），蓝星→橙菱形粗箭头 +「帕累托改进」标注，刻度与
     报告表3 对齐；
     下：相对变化率条形图（T_max +0.0% / δ −62.1%，以问题1 为基准），
     量化"以零总耗时代价换取负载大幅均衡"。
图8：跨算例 δ 均衡效果汇总（单图，分组条形图）——横轴四算例，每算例
     蓝柱=问题1 δ、橙柱=问题2 δ（min），柱顶标注相对降幅（Case1
     −62.1%、Case2 −51.8%、Case3 −41.3%、Case4 −71.4%，以问题1 为
     基准），一眼呈现"四算例全部实现均衡改善"的整体结论，与报告
     4.2 节对比表格图文呼应（Case4 降幅最大、Case3 最小）。

数据来源：
- Q1 解档案 outputs/workbooks/q1_solution_Case*.json（uavs[].work_s、Tmax_s/Tmin_s）
- Q2 最终解档案 enhanced_run_20260817_e2000/q2/strict/Case*.json
  （metrics.route_work_s 等；报告表3 依据——注意 outputs/workbooks/q2/strict/
   为过时档案，Case1 δ=807 s 等旧值与正式结果不一致，不可用于绘图）
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
import matplotlib.ticker as mticker  # noqa: E402

# 中文字体：须在 style.use 之后再设置，否则被 seaborn 风格覆盖为 Arial
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES = ["Case1", "Case2", "Case3", "Case4"]
Q2_DIR = REPO_ROOT / "Q2"
FIG_DIR = Q2_DIR / "figs"
ARCHIVE = REPO_ROOT / "outputs" / "workbooks"
# Q2 最终解档案：报告表3 依据（增强邻域正式预算运行 2026-08-17 落盘于
# enhanced_run_20260817_e2000/）。outputs/workbooks/q2/strict/ 为过时档案
# （Case1 δ=807 s 等旧值），与正式结果不一致，不可用于绘图。
ARCHIVE_Q2 = REPO_ROOT / "enhanced_run_20260817_e2000" / "q2" / "strict"

C_Q1 = "#4C72B0"  # 问题1 蓝
C_Q2 = "#DD8452"  # 问题2 橙
C_GRAY = "#AAAAAA"  # 未调整无人机（灰色弱化）
C_TXT = "#333333"


def save_fig(fig, path, dpi=300):
    """保存 png/pdf 双格式。"""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"图片已保存到：{path}（dpi={dpi}）")


def load_q1(case: str) -> dict:
    return json.load(open(ARCHIVE / f"q1_solution_{case}.json", encoding="utf-8"))


def load_q2(case: str) -> dict:
    return json.load(open(ARCHIVE_Q2 / f"{case}.json", encoding="utf-8"))


def fig6_workload_compare(q1: dict[str, dict], q2: dict[str, dict]) -> plt.Figure:
    """图6：Case1 单机工作时长调整细节（纵轴无人机，按问题1 工时升序）。

    任务重分配机制展示：最短路线 UAV2 工时抬升（29587→30623 s，
    +17.3 min）、UAV4 换出部分任务（30977→30800 s，−3.0 min）、
    UAV1/UAV3 完全不动（灰色弱化），均衡调整仅发生在 UAV2 与 UAV4
    之间；蓝圆点=问题1、橙菱形=问题2、灰线=同机调整量 Δ（min）；
    点旁标注 4 位小数时长（h）；两条 T_max 虚线完全重合，直观传递
    "最长工时未劣化"。纵轴局部放大（8.0~8.9 h）以呈现调整细节。
    """
    case = "Case1"
    a1, a2 = q1[case], q2[case]
    w1 = [u["work_s"] / 3600.0 for u in a1["uavs"]]
    w2 = [s / 3600.0 for s in a2["metrics"]["route_work_s"]]
    ids = np.arange(1, len(w1) + 1)
    # 按问题1 工时从低到高排序（最短路线置顶，"最短抬升、最长不动"趋势可见）
    order = np.argsort(w1)
    ids, w1, w2 = ids[order], [w1[i] for i in order], [w2[i] for i in order]
    n = len(w1)
    y = np.arange(n)[::-1]  # 顶部=工时最短
    delta_min = [(b - a) * 60.0 for a, b in zip(w1, w2)]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    # 同机连线（仅调整机画线；零调整机两点重合，画线无信息）
    for yi, v1, v2, d in zip(y, w1, w2, delta_min):
        if abs(d) > 1e-9:
            ax.plot([v1, v2], [yi, yi], color="#999999", linewidth=1.4,
                    zorder=1, alpha=0.9)
    # 零调整机（UAV1/UAV3）：灰色弱化，明确"均衡调整仅发生在 UAV2/UAV4"
    for yi, v1, v2, d in zip(y, w1, w2, delta_min):
        if abs(d) < 1e-9:
            ax.scatter([v1], [yi], s=80, marker="o", color=C_GRAY, zorder=3,
                       edgecolors="white", linewidths=0.8)
            ax.text(v1 + 0.012, yi, f"{v1:.4f}", fontsize=8, color="#999999",
                    va="center", ha="left")
    # 调整机（UAV2/UAV4）：蓝圆=问题1、橙菱=问题2，点旁标 4 位小数时长
    for yi, v1, v2, d in zip(y, w1, w2, delta_min):
        if abs(d) < 1e-9:
            continue
        ax.scatter([v1], [yi], s=90, marker="o", color=C_Q1, zorder=3,
                   edgecolors="white", linewidths=0.8)
        ax.scatter([v2], [yi], s=90, marker="D", color=C_Q2, zorder=3,
                   edgecolors="white", linewidths=0.8)
        # 蓝标注在连线上方、橙标注在连线下方，避免两点间距小时文字重叠
        ax.text(v1 - 0.012, yi + 0.16, f"{v1:.4f}", fontsize=8.5, color=C_Q1,
                ha="right", va="center")
        ax.text(v2 + 0.012, yi - 0.16, f"{v2:.4f}", fontsize=8.5, color=C_Q2,
                ha="left", va="center")
        # Δ 标注（连线中点上方）
        ax.text((v1 + v2) / 2.0, yi + 0.38, f"Δ {d:+.1f} min", ha="center",
                fontsize=8.5, color=C_TXT, fontweight="bold")
    # T_max 参考线：问题1 蓝虚线、问题2 橙虚线，完全重合（最长工时未劣化）
    tmax1 = a1["Tmax_h"]
    tmax2 = a2["metrics"]["Tmax_s"] / 3600.0
    ax.axvline(tmax1, color=C_Q1, linestyle="--", linewidth=1.2, alpha=0.85,
               zorder=2)
    ax.axvline(tmax2, color=C_Q2, linestyle="--", linewidth=1.2, alpha=0.85,
               zorder=2)
    # 纵轴局部放大：数据差异（h 级 0.01）在 0 起点轴上不可见
    ax.set_xlim(8.0, 8.9)
    ax.set_xticks(np.arange(8.0, 8.91, 0.1))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_xlabel("工作时长（h）", fontsize=10.5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"UAV{i}" for i in ids])
    ax.set_ylabel("无人机（按问题1 工时升序）", fontsize=10.5)
    ax.set_ylim(-0.6, 3.75)
    ax.grid(axis="x", alpha=0.3)
    ax.text(0.012, 0.97, case, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left", zorder=6,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2))
    # 顶部共用图例（手动构造 handle，避免循环内 label 重复）
    q1_h = plt.Line2D([], [], marker="o", linestyle="none", color=C_Q1,
                      markersize=7, markeredgecolor="white", markeredgewidth=0.8)
    q2_h = plt.Line2D([], [], marker="D", linestyle="none", color=C_Q2,
                      markersize=6.5, markeredgecolor="white", markeredgewidth=0.8)
    gray_h = plt.Line2D([], [], marker="o", linestyle="none", color=C_GRAY,
                        markersize=7, markeredgecolor="white", markeredgewidth=0.8)
    t1_h = plt.Line2D([], [], color=C_Q1, linestyle="--", linewidth=1.2)
    t2_h = plt.Line2D([], [], color=C_Q2, linestyle="--", linewidth=1.2)
    fig.legend([q1_h, q2_h, gray_h, t1_h, t2_h],
               ["问题1", "问题2", "未调整（Δ=0）",
                f"T_max(问题1)={tmax1:.2f} h", f"T_max(问题2)={tmax2:.2f} h"],
               loc="upper center", ncol=5, fontsize=8.5, frameon=True,
               bbox_to_anchor=(0.5, 1.02), framealpha=0.9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def fig7_tradeoff(q1: dict[str, dict], q2: dict[str, dict]) -> plt.Figure:
    """图7：Case1 双目标权衡图（上：T_max-δ 权衡平面局部放大；下：变化率条形）。

    上子图：x=负载差 δ（min，0~30）、y=T_max（h，基线±0.1 h 局部放大）——
    问题1 蓝星 → 问题2 橙菱形，粗箭头示移动方向，箭头旁标注「帕累托改进」
    （T_max 不升、δ 大幅下降），刻度与报告表3 对齐；
    下子图：相对变化率横向条形图——T_max +0.0%（条高为 0，零代价）与
    δ −62.1%（负向条），以问题1 为基准，量化"以零总耗时代价换取负载
    大幅均衡"。
    """
    case = "Case1"
    a1, a2 = q1[case], q2[case]
    t1, d1 = a1["Tmax_h"], (a1["Tmax_s"] - a1["Tmin_s"]) / 60.0
    t2, d2 = a2["metrics"]["Tmax_s"] / 3600.0, a2["metrics"]["delta_s"] / 60.0
    pct_t = (t2 - t1) / t1 * 100.0
    pct_d = (d2 - d1) / d1 * 100.0

    fig = plt.figure(figsize=(8.5, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.6, 1.0], hspace=0.5)
    ax1 = fig.add_subplot(gs[0])

    # ---- 上子图：T_max-δ 权衡平面（局部放大，避免两点挤在 0 起点角落）----
    ax1.scatter([d1], [t1], s=220, marker="*", color=C_Q1, zorder=5,
                label=f"问题1（T_max={t1:.2f} h，δ={d1:.1f} min）")
    ax1.scatter([d2], [t2], s=180, marker="D", color=C_Q2, zorder=5,
                label=f"问题2（T_max={t2:.2f} h，δ={d2:.1f} min）")
    # 粗箭头：问题1 → 问题2 移动方向（δ 收窄、T_max 不变）
    ax1.annotate("", xy=(d2, t2), xytext=(d1, t1),
                 arrowprops=dict(arrowstyle="->,head_length=0.7,head_width=0.5",
                                 color="#555555", lw=3.0,
                                 shrinkA=16, shrinkB=16))
    ax1.text((d1 + d2) / 2.0, t1 + 0.055, "帕累托改进", ha="center",
             va="bottom", fontsize=11, fontweight="bold", color="#C0392B",
             bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))
    ax1.set_xlabel("负载差 δ（min）", fontsize=11)
    ax1.set_ylabel("总体完成时间 T_max（h）", fontsize=11)
    ax1.set_xlim(0, 30)
    ax1.set_xticks(range(0, 31, 5))
    ax1.set_ylim(t1 - 0.1, t1 + 0.1)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc="lower left", labelspacing=1.5)
    ax1.text(0.015, 0.98, case, transform=ax1.transAxes, fontsize=12,
             fontweight="bold", va="top", ha="left", zorder=6,
             bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2))

    # ---- 下子图：相对变化率横向条形图（以问题1 为基准）----
    axb = fig.add_subplot(gs[1])
    yidx = np.arange(2)[::-1]  # 上=T_max、下=δ
    vals = [pct_t, pct_d]
    names = ["T_max（h）", "δ（min）"]
    colors = [C_Q1, C_Q2]
    axb.barh(yidx, vals, height=0.5, color=colors, edgecolor="white", zorder=3)
    axb.axvline(0, color="#333333", linewidth=1.5, zorder=4)
    for yi, v, nm in zip(yidx, vals, names):
        if abs(v) < 1e-9:
            # T_max 零变化：条高 0 不可见，用文本明确"零代价"
            axb.text(1.5, yi, f"{nm}: +0.0%（零代价）", va="center",
                     fontsize=10, color=C_TXT)
        else:
            axb.text(v - 2.0, yi, f"{v:+.1f}%", ha="right", va="center",
                     fontsize=11, fontweight="bold", color=colors[1])
    axb.set_yticks(yidx)
    axb.set_yticklabels(names)
    axb.set_xlim(-75, 20)
    axb.set_xticks([-75, -50, -25, 0, 25])
    axb.set_xlabel("相对变化率（%）（负值=下降，以问题1 为基准）", fontsize=10)
    axb.set_ylim(-0.6, 1.6)
    axb.grid(axis="x", alpha=0.3)
    # GridSpec 手动布局下 tight_layout 不可用，用 subplots_adjust 固定边距
    fig.subplots_adjust(left=0.12, right=0.96, top=0.97, bottom=0.05, hspace=0.5)
    return fig


def fig8_cross_case_delta(q1: dict[str, dict], q2: dict[str, dict]) -> plt.Figure:
    """图8：跨算例 δ 均衡效果汇总（分组条形图）。

    横轴四算例，每算例两根柱：蓝=问题1 δ、橙=问题2 δ（min）；
    橙柱上方标注 δ 相对降幅（以问题1 为基准，红色加粗）。一眼呈现
    "四算例全部实现均衡改善"的整体结论，与报告 4.2 节对比表格图文
    呼应（Case4 降幅最大 −71.4%、Case3 最小 −41.3%）。
    """
    d1 = [(q1[c]["Tmax_s"] - q1[c]["Tmin_s"]) / 60.0 for c in CASES]
    d2 = [q2[c]["metrics"]["delta_s"] / 60.0 for c in CASES]
    pct = [(b - a) / a * 100.0 for a, b in zip(d1, d2)]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(CASES))
    width = 0.35
    ax.bar(x - width / 2, d1, width, color=C_Q1, edgecolor="white",
           label="问题1", zorder=3)
    ax.bar(x + width / 2, d2, width, color=C_Q2, edgecolor="white",
           label="问题2", zorder=3)
    # 柱顶标注 δ 相对降幅（以问题1 为基准）
    for xi, a, b, p in zip(x, d1, d2, pct):
        top = max(a, b)
        ax.text(xi, top + 0.6, f"{p:+.1f}%", ha="center", fontsize=11,
                fontweight="bold", color="#C0392B", zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(CASES)
    ax.set_xlabel("测试算例", fontsize=11)
    ax.set_ylabel("负载差 δ（min）", fontsize=11)
    ax.set_ylim(0, max(d1) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    return fig


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    q1 = {c: load_q1(c) for c in CASES}
    q2 = {c: load_q2(c) for c in CASES}

    for fn, name in [(fig6_workload_compare, "fig6_问题1vs2工作时长对比"),
                     (fig7_tradeoff, "fig7_问题2双目标权衡"),
                     (fig8_cross_case_delta, "fig8_跨算例δ均衡效果对比")]:
        fig = fn(q1, q2)
        for ext in ("png", "pdf"):
            save_fig(fig, FIG_DIR / f"{name}.{ext}", dpi=300)
        plt.close(fig)
        print(f"完成：{name}")


if __name__ == "__main__":
    main()
