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
总结对比图（图12）：N / T_max / δ 三指标在问题1→问题2→问题3 的变化。

- N：各算例无人机数量（三问保持一致，体现方案约束下数量稳定）；
- T_max：总体完成时间（问题3 用 S_max）；
- δ：负载差（T_max − T_min，问题3 用 Δ）。
每个指标一个子图（3×1），横轴为四个测试算例，三问并列柱状对比。

数据来源：
- Q1 解档案 outputs/workbooks/q1_solution_Case*.json（N / Tmax_s / Tmin_s）
- Q2 严格解档案 outputs/workbooks/q2/strict/Case*.json（fleet_size / Tmax_s / delta_s）
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
        q2 = json.load(open(ARCHIVE / "q2" / "strict" / f"{c}.json", encoding="utf-8"))
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
    """图12：N / T_max / δ 三指标三问题对比（3×1 子图）。"""
    labels = ["N（架）", "T_max（h）", "δ（h）"]
    keys = ["N", "Tmax", "delta"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    x = np.arange(len(CASES))
    width = 0.25
    for ax, key, label in zip(axes, keys, labels):
        for j, (qname, color) in enumerate(Q_COLORS.items()):
            vals = [data[c][qname][key] for c in CASES]
            bars = ax.bar(x + (j - 1) * width, vals, width, color=color,
                          edgecolor="white", label=qname, zorder=3)
            for b, v in zip(bars, vals):
                fmt = f"{v:.0f}" if key == "N" else f"{v:.2f}"
                ax.text(b.get_x() + b.get_width() / 2, v * 1.02, fmt,
                        ha="center", fontsize=8.5)
        ax.set_ylabel(label, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        if key == "N":
            ax.set_title("三问题结果汇总：无人机数量 N 保持一致，完成时间与负载差逐步优化",
                         fontsize=13)
        else:
            ax.legend(fontsize=9, loc="upper right")
    ax.set_xticks(x)
    ax.set_xticklabels(CASES)
    ax.set_xlabel("测试算例", fontsize=11)
    fig.tight_layout()
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
