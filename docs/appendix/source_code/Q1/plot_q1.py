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
问题1 成果图（图3、图4、图5）。

图3：任务分配与路径规划图（2×2，每算例一子图）——各无人机自基地 (0,0)
     出发、巡检、返回的完整航迹，按无人机编号分色，巡检点按等级着色。
图4：各无人机工作时长柱状图（2×2）——标注 T_max / T_min 参考线。
图5：求解收敛曲线（2×2）——OR-Tools 两阶段求解过程中目标值随搜索时间下降。

数据来源：
- 解档案 outputs/workbooks/q1_solution_Case*.json（uavs[].point_seq / work_s / work_h）
- 附件1 附件1.xlsx（坐标与巡检等级，1 单位 = 100 m）
- 收敛日志 Q1/logs/convergence_Case*.csv（elapsed_s, objective）
输出：Q1/figs/fig3_*.png/.pdf、fig4_*.png/.pdf、fig5_*.png/.pdf

用法（仓库根目录，math 环境）：
    python Q1/plot_q1.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无头环境保存图片
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# 中文字体：须在 style.use 之后再设置，否则被 seaborn 风格覆盖为 Arial
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES = ["Case1", "Case2", "Case3", "Case4"]
Q1_DIR = REPO_ROOT / "Q1"
FIG_DIR = Q1_DIR / "figs"
ARCHIVE = REPO_ROOT / "outputs" / "workbooks"
ATTACH_XLSX = REPO_ROOT / "attachment" / "附件1.xlsx"
BASE = (0.0, 0.0)  # 飞行基地坐标（单位：100 m）

LEVEL_STYLE = {
    "I":   ("#C0392B", 3, "I级（3次）"),
    "II":  ("#F39C12", 2, "II级（2次）"),
    "III": ("#2980B9", 1, "III级（1次）"),
}
# 无人机航迹配色（tab20 循环，N 最大 5 架）
UAV_CMAP = plt.get_cmap("tab20")


def save_fig(fig, path, dpi=300):
    """保存 png/pdf 双格式。"""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"图片已保存到：{path}（dpi={dpi}）")


def load_points(case: str) -> dict[int, dict]:
    """附件1 某算例 → {pid: {x, y, level}}。"""
    df = pd.read_excel(ATTACH_XLSX, sheet_name=case)
    pts = {}
    for row in df.itertuples(index=False):
        pts[int(row.Point_ID)] = {
            "x": float(row.X_Coordinate),
            "y": float(row.Y_Coordinate),
            "level": str(row.Inspection_Level).strip().upper(),
        }
    return pts


def load_archives() -> dict[str, dict]:
    return {c: json.load(open(ARCHIVE / f"q1_solution_{c}.json", encoding="utf-8"))
            for c in CASES}


def plot_route_points(ax, pts: dict[int, dict]):
    """在子图上绘制巡检点（按等级着色）。"""
    for lvl, (color, _, _) in LEVEL_STYLE.items():
        xs = [p["x"] for p in pts.values() if p["level"] == lvl]
        ys = [p["y"] for p in pts.values() if p["level"] == lvl]
        ax.scatter(xs, ys, s=22, c=color, alpha=0.8, edgecolors="white",
                   linewidths=0.3, zorder=3)
    ax.plot(*BASE, marker="*", markersize=13, color="gold",
            markeredgecolor="black", markeredgewidth=0.7, zorder=5)


def fig3_routes(archives: dict[str, dict], pts_all: dict[str, dict]) -> plt.Figure:
    """图3：任务分配与路径规划图（2×2）。"""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
    for ax, case in zip(axes.flat, CASES):
        arc = archives[case]
        pts = pts_all[case]
        plot_route_points(ax, pts)
        for i, u in enumerate(arc["uavs"]):
            seq = [BASE] + [(pts[pid]["x"], pts[pid]["y"]) for pid in u["point_seq"]] + [BASE]
            xs, ys = zip(*seq)
            ax.plot(xs, ys, "-", color=UAV_CMAP(2 * i % 20), linewidth=1.4,
                    alpha=0.85, zorder=2, label=f"无人机{i + 1}")
        ax.set_title(f"{case}：N={arc['N']} 架，T_max={arc['Tmax_h']:.2f} h",
                     fontsize=12)
        ax.set_xlabel("X 坐标（单位：100 m）", fontsize=10)
        ax.set_ylabel("Y 坐标（单位：100 m）", fontsize=10)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.suptitle("问题1 多无人机任务分配与路径规划方案（各无人机自基地出发并返回）",
                 fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def fig4_workload(archives: dict[str, dict]) -> plt.Figure:
    """图4：各无人机工作时长柱状图（2×2）。"""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, case in zip(axes.flat, CASES):
        arc = archives[case]
        uavs = arc["uavs"]
        ids = [u["uav_id"] for u in uavs]
        work_h = [u["work_h"] for u in uavs]
        bars = ax.bar(ids, work_h, width=0.55, color=[UAV_CMAP(2 * i % 20) for i in range(len(ids))],
                      edgecolor="white", zorder=3)
        for b, w in zip(bars, work_h):
            ax.text(b.get_x() + b.get_width() / 2, w + 0.05, f"{w:.2f}",
                    ha="center", fontsize=9)
        ax.axhline(arc["Tmax_h"], color="#C0392B", linestyle="--", linewidth=1.2,
                   label=f"T_max={arc['Tmax_h']:.2f} h")
        ax.axhline(arc["Tmin_h"], color="#2980B9", linestyle=":", linewidth=1.2,
                   label=f"T_min={arc['Tmin_h']:.2f} h")
        ax.set_title(f"{case}：N={arc['N']} 架", fontsize=12)
        ax.set_xlabel("无人机编号", fontsize=10)
        ax.set_ylabel("工作时长（h）", fontsize=10)
        ax.set_ylim(0, arc["Tmax_h"] * 1.18)
        ax.legend(fontsize=9, loc="upper right")
    fig.suptitle("问题1 各无人机工作时长分布（飞行 + 巡检）", fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def fig5_convergence() -> plt.Figure:
    """图5：求解收敛曲线（2×2）。"""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, case in zip(axes.flat, CASES):
        csv_path = Q1_DIR / "logs" / f"convergence_{case}.csv"
        df = pd.read_csv(csv_path)
        # 按秒取整避免曲线过密；步进线反映目标值离散下降过程
        ax.plot(df["elapsed_s"] / 60.0, df["objective"] / 3600.0,
                drawstyle="steps-post", color="#4C72B0", linewidth=1.3)
        final = df["objective"].iloc[-1] / 3600.0
        ax.axhline(final, color="#C0392B", linestyle="--", linewidth=1.0,
                   label=f"最终目标 {final:.2f} h")
        ax.set_title(f"{case}", fontsize=12)
        ax.set_xlabel("求解时间（min）", fontsize=10)
        ax.set_ylabel("目标值（h）", fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
    fig.suptitle("问题1 求解过程目标值收敛曲线（OR-Tools 两阶段搜索）",
                 fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    archives = load_archives()
    pts_all = {c: load_points(c) for c in CASES}

    for fn, name in [
        (fig3_routes, "fig3_问题1路径规划方案"),
        (fig4_workload, "fig4_问题1各机工作时长"),
        (fig5_convergence, "fig5_问题1求解收敛曲线"),
    ]:
        args = (archives, pts_all) if name.startswith("fig3") else (archives,) if name.startswith("fig4") else ()
        fig = fn(*args)
        for ext in ("png", "pdf"):
            save_fig(fig, FIG_DIR / f"{name}.{ext}", dpi=300)
        plt.close(fig)
        print(f"完成：{name}")


if __name__ == "__main__":
    main()
