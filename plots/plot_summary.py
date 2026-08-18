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
总结对比图（图12）：T_max / δ 两核心指标在问题1→问题2→问题3 的变化。

- T_max：总体完成时间（问题3 用 S_max）。9 h 处黑色虚线标注「原 9 小时时限」，
  直观呈现 Case2/3/4 引入禁飞区后超出原时限；问题3 柱上方以红色标注相对
  问题2 的增幅（+0.0% / +55.2% / +15.7% / +19.0%），量化禁飞区时间代价。
- δ：负载差（T_max − T_min，问题3 用 Δ），单位 min（差值均 < 0.5 h，
  分钟制对比直观）；问题2/3 柱上方以绿色标注相对问题1 的降幅；
  Case2 问题3 δ=0 以金色「★ 完全均衡」突出优化成果。
2×1 子图（删除原 N 子图，版面全部留给两个核心指标），横轴为四个测试算例，
三问并列柱状对比。

数据来源：
- Q1 解档案 outputs/workbooks/q1_solution_Case*.json（N / Tmax_s / Tmin_s）
- Q2 严格解档案 enhanced_run_20260817_e2000/q2/strict/Case*.json
  （fleet_size / Tmax_s / delta_s——报告表3 依据；勿用
  outputs/workbooks/q2/strict/，该目录为过时档案）
- Q3 严格解档案 outputs/workbooks/q3/strict/Case*.json（fleet_size / S_max_s / delta_s）
输出：plots/figs/fig12_三问题结果汇总.png/.pdf

用法（仓库根目录，math 环境）：
    python plots/plot_summary.py
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
FIG_DIR = REPO_ROOT / "plots" / "figs"
ARCHIVE = REPO_ROOT / "outputs" / "workbooks"
# Q2 最终档案（报告表3 依据）；outputs/workbooks/q2/strict/ 为过时档案不可用
Q2_ARCHIVE = REPO_ROOT / "enhanced_run_20260817_e2000" / "q2" / "strict"

Q_COLORS = {"问题1": "#4C72B0", "问题2": "#DD8452", "问题3": "#55A868"}


def save_fig(fig, path, dpi=300):
    """保存 png/pdf 双格式。"""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"图片已保存到：{path}（dpi={dpi}）")


def collect() -> dict[str, dict[str, float]]:
    """{case: {问题1: {N,Tmax,delta}, 问题2: {...}, 问题3: {...}}}（时间单位 h）。"""
    out = {}
    for c in CASES:
        q1 = json.load(open(ARCHIVE / f"q1_solution_{c}.json", encoding="utf-8"))
        q2 = json.load(open(Q2_ARCHIVE / f"{c}.json", encoding="utf-8"))
        q3 = json.load(open(ARCHIVE / "q3" / "strict" / f"{c}.json", encoding="utf-8"))
        out[c] = {
            "问题1": {"N": q1["N"], "Tmax": q1["Tmax_s"] / 3600.0,
                      "delta": (q1["Tmax_s"] - q1["Tmin_s"]) / 3600.0},
            "问题2": {"N": q2["fleet_size"], "Tmax": q2["metrics"]["Tmax_s"] / 3600.0,
                      "delta": q2["metrics"]["delta_s"] / 3600.0},
            "问题3": {"N": q3["fleet_size"], "Tmax": q3["metrics"]["S_max_s"] / 3600.0,
                      "delta": q3["metrics"]["delta_s"] / 3600.0},
        }
    return out


def fig12_summary(data: dict[str, dict[str, dict[str, float]]]) -> plt.Figure:
    """图12：T_max / δ 两指标三问题对比（2×1 子图，四组三柱、组间留白）。

    (a) 最长完成时间 (h)（问题1/2 为工作时长 T_max、问题3 为含等待的完成
    时刻 S_max）：9 h 黑色虚线「原 9 小时时限（问题1 约束）」参考线；问题3
    柱正上方红色增幅标注 + 上标 ¹（基准=问题2）。
    (b) 工作时长极差 δ (min)：问题2/3 柱右上方绿色降幅标注 + 上标 ²
    （基准=问题1）；Case2 问题3 δ=0 以金色「★ 完全均衡（δ=0）²」突出。
    基准说明以脚注形式置于整图底部（¹ 增幅 vs 问题2、² 降幅 vs 问题1）。
    整图仅一组图例（右上角）；无 xlabel；网格 #EEEEEE 极浅灰。
    """
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
    x = np.arange(len(CASES)) * 1.25  # 组内紧贴、组间留白强化分组
    width = 0.24
    x_lo, x_hi = -0.9, 5.7  # 为右侧参考线标注留空间
    handles = []

    # ---- (a) 上子图：最长完成时间 ----
    ax = axes[0]
    ax.text(0.008, 0.965, "(a)", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left", zorder=7)
    for j, (qname, color) in enumerate(Q_COLORS.items()):
        vals = [data[c][qname]["Tmax"] for c in CASES]
        bars = ax.bar(x + (j - 1) * width, vals, width, color=color,
                      edgecolor="white", label=qname, zorder=3)
        handles.append(bars)  # BarContainer 整体入图例句柄（patch 默认 label 为 _nolegend_）
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9, zorder=6)
    q3_t = [data[c]["问题3"]["Tmax"] for c in CASES]
    q2_t = [data[c]["问题2"]["Tmax"] for c in CASES]
    ymax_t = max(q3_t) * 1.18
    ax.set_ylim(0, ymax_t)
    # 原 9 小时时限参考线（问题1 约束，问题3 不继承——图面证据）
    ax.axhline(9, color="black", ls="--", lw=1.2, zorder=1)
    ax.text(x[-1] + width + 0.02, 9, "原 9 小时时限（问题1 约束）", ha="left",
            va="center", fontsize=9, color="black", zorder=6,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1))
    # 问题3 相对问题2 的增幅（红色，正上方统一偏移纵向对齐；¹ 脚注说明基准）
    for i in range(len(CASES)):
        pct = (q3_t[i] - q2_t[i]) / q2_t[i] * 100
        xc = x[i] + width  # 问题3 柱中心
        ax.text(xc, q3_t[i] + 0.075 * ymax_t, f"{pct:+.1f}%$^{1}$",
                ha="center", va="bottom", fontsize=9, color="#C0392B", zorder=6)
    ax.set_ylabel("最长完成时间 (h)", fontsize=11)
    ax.set_xlim(x_lo, x_hi)
    ax.grid(axis="y", color="#EEEEEE", linewidth=0.8)

    # ---- (b) 下子图：工作时长极差 δ (min) ----
    ax = axes[1]
    ax.text(0.008, 0.965, "(b)", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left", zorder=7)
    for j, (qname, color) in enumerate(Q_COLORS.items()):
        vals = [data[c][qname]["delta"] * 60 for c in CASES]  # h → min
        bars = ax.bar(x + (j - 1) * width, vals, width, color=color,
                      edgecolor="white", zorder=3)
        for b, v in zip(bars, vals):
            if v < 0.05:  # δ=0 零柱不标数值（星标旁注）
                continue
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=9, zorder=6)
    d1 = [data[c]["问题1"]["delta"] * 60 for c in CASES]
    d2 = [data[c]["问题2"]["delta"] * 60 for c in CASES]
    d3 = [data[c]["问题3"]["delta"] * 60 for c in CASES]
    ymax_d = max(d1) * 1.18
    ax.set_ylim(0, ymax_d)
    # 问题2/3 相对问题1 的降幅（绿色，柱右上方，² 脚注说明基准）；δ=0 星标
    for i in range(len(CASES)):
        for j, dv in ((0, d2[i]), (1, d3[i])):
            xpos = x[i] + j * width  # 问题2/问题3 柱中心
            if dv < 0.05:  # δ=0：星标放柱上方空白（含 δ 值，杜绝"漏画"误解）
                ax.text(xpos, 0.09 * ymax_d, "★ 完全均衡（δ=0）$^{2}$",
                        ha="center", va="bottom", fontsize=9.5,
                        color="#D4A017", fontweight="bold", zorder=6)
            else:
                pct = (d1[i] - dv) / d1[i] * 100
                xr = xpos + width / 2 + 0.01  # 柱右缘外侧
                ax.text(xr, dv + 0.01 * ymax_d, f"-{pct:.0f}%$^{2}$", ha="left",
                        va="bottom", fontsize=9, color="#2E7D32", zorder=6)
    ax.set_ylabel("工作时长极差 δ (min)", fontsize=11)
    ax.set_xlim(x_lo, x_hi)
    ax.grid(axis="y", color="#EEEEEE", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(CASES)
    # 脚注：标注基准说明（¹ 增幅 vs 问题2、² 降幅 vs 问题1）
    fig.text(0.5, 0.008,
             "$^{1}$ 增幅以问题 2 方案为基准；$^{2}$ 降幅以问题 1 方案为基准"
             "（Case2 问题3 完全均衡，降幅为 -100%）",
             ha="center", va="bottom", fontsize=8, color="#666666")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.legend(handles, [h.get_label() for h in handles], loc="upper right",
               fontsize=9, frameon=True)
    return fig


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = collect()
    fig = fig12_summary(data)
    for ext in ("png", "pdf"):
        save_fig(fig, FIG_DIR / f"fig12_三问题结果汇总.{ext}", dpi=300)
    plt.close(fig)
    print("完成：fig12_三问题结果汇总")


if __name__ == "__main__":
    main()
