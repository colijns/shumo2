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
问题3 成果图（图9、图10、图11）。

图9：禁飞区与绕行路径图（典型算例 Case2 单图大图）——四个算例中仅
     Case2 存在真实绕行（detour 5 段 + visibility 6 段 + wait_direct
     1 段，其余算例禁飞区未造成实际绕行），故图9 选取 Case2 放大绘制：
     圆形禁飞区（半透明红，标注编号与时间窗）叠加各无人机航迹，路径按
     类型着色。关键：detour/visibility 段绘制真实绕行路径（切点直线 +
     沿禁飞区圆边弧线，迭代绕行保证路径不穿过任何禁飞区），而非起终点
     直线——其直线弦恰好穿过禁飞区（几何验证 d=3.1~43.3 < r），直线
     画法会造成"红线穿禁飞区"的物理矛盾；direct/wait_direct 段保持
     直线（飞行时刻与禁飞窗不冲突，等待后直飞段已等禁区解除）。
图10：无人机任务甘特图（2×2）——每算例一子图：横轴为自 8:00 起的时间，
      纵轴为无人机编号，飞行段/巡检服务段分色，禁飞时段以红色背景标出。
图11：绕行代价对比（单图两子图）——Q1 与 Q3 总飞行距离对比（禁飞区
      带来的距离代价），以及 Q3 各路径类型段数分布。

数据来源：
- Q3 解档案 outputs/workbooks/q3/strict/Case*.json（schedules/segments/metrics）
- Q1 解档案 outputs/workbooks/q1_solution_Case*.json（uavs[].dist_km）
- 附件1（坐标与巡检等级）、附件2（禁飞区圆心/半径/时间窗）
输出：Q3/figs/fig9_*.png/.pdf、fig10_*.png/.pdf、fig11_*.png/.pdf

用法（仓库根目录，math 环境）：
    python Q3/plot_q3.py
"""

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无头环境保存图片
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Circle

# 中文字体：须在 style.use 之后再设置，否则被 seaborn 风格覆盖为 Arial
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES = ["Case1", "Case2", "Case3", "Case4"]
FIG_DIR = REPO_ROOT / "Q3" / "figs"
ARCHIVE = REPO_ROOT / "outputs" / "workbooks"
ATTACH_XLSX = REPO_ROOT / "attachment" / "附件1.xlsx"
ATTACH_ZONES = REPO_ROOT / "attachment" / "附件2.xlsx"
START_HOUR = 8.0  # 每天 8:00 开始执行任务
BASE = (0.0, 0.0)

LEVEL_VISITS = {"I": 3, "II": 2, "III": 1}  # 与 Q1 solve_q1.py 一致的展开规则
LEVEL_COLOR = {"I": "#C0392B", "II": "#F39C12", "III": "#2980B9"}
# 路径类型配色
PATH_STYLE = {
    "direct":      ("#4C72B0", "直飞"),
    "detour":      ("#C44E52", "绕行"),
    "wait_direct": ("#55A868", "等待后直飞"),
    "visibility":  ("#8172B3", "可见性调整"),
}


def save_fig(fig, path, dpi=300):
    """保存 png/pdf 双格式。"""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"图片已保存到：{path}（dpi={dpi}）")


# ---------------- 数据装载 ----------------
def load_case(case: str) -> dict:
    """附件1 → {pid: {x, y, level}}，并按 Q1 规则展开任务：{task_id: pid}。"""
    df = pd.read_excel(ATTACH_XLSX, sheet_name=case)
    points = {}
    for row in df.itertuples(index=False):
        pid = int(row.Point_ID)
        points[pid] = {"x": float(row.X_Coordinate), "y": float(row.Y_Coordinate),
                       "level": str(row.Inspection_Level).strip().upper()}
    task_of = {}
    tid = 1
    for pid in points:
        for _ in range(LEVEL_VISITS[points[pid]["level"]]):
            task_of[tid] = pid
            tid += 1
    return {"points": points, "task_of": task_of}


def load_zones(case: str) -> list[dict]:
    """附件2 → 禁飞区列表（时间窗转自 8:00 起的秒数）。"""
    df = pd.read_excel(ATTACH_ZONES, sheet_name=case)
    zones = []
    for row in df.itertuples(index=False):
        def to_s(t: str) -> int:
            h, m = str(t).strip().split(":")
            return (int(h) - START_HOUR) * 3600 + int(m) * 60
        zones.append({
            "id": str(row.Zone_ID),
            "x": float(row.Center_X), "y": float(row.Center_Y),
            "r": float(row.Radius),
            "t0": to_s(row.Start_Time), "t1": to_s(row.End_Time),
            "win": f"{row.Start_Time}–{row.End_Time}",
        })
    return zones


def load_q3(case: str) -> dict:
    return json.load(open(ARCHIVE / "q3" / "strict" / f"{case}.json", encoding="utf-8"))


def load_q1(case: str) -> dict:
    return json.load(open(ARCHIVE / f"q1_solution_{case}.json", encoding="utf-8"))


def seg_point(case_data: dict, node_id: int):
    """节点号 → (x, y)：0 为基地，其余为任务号。"""
    if node_id == 0:
        return BASE
    pid = case_data["task_of"][node_id]
    p = case_data["points"][pid]
    return (p["x"], p["y"])


# ---------------- 禁飞区绕行路径重建（复用求解器同款几何） ----------------
sys.path.insert(0, str(REPO_ROOT))  # domain 依赖顶层包 Q2
sys.path.insert(0, str(REPO_ROOT / "Q3"))
from geometry import detour_path, visibility_path, TAU  # noqa: E402
from domain import EPS_ARC_KM, SAFETY_MARGIN_KM  # noqa: E402

KM_UNIT = 0.1  # 附件坐标 1 单位 = 100 m = 0.1 km（求解器几何在 km 空间）


def _to_km(p) -> tuple[float, float]:
    return (p[0] * KM_UNIT, p[1] * KM_UNIT)


def _to_att(p) -> tuple[float, float]:
    return (p[0] / KM_UNIT, p[1] / KM_UNIT)


def _arc_sample(center, r, a0, a1, n=36) -> list[tuple[float, float]]:
    """圆边 a0→a1 的短弧采样（不含端点）。

    方向与 safe_path._arc_bounds 同款重算：forward 取短弧（≤ π），
    不直接依赖 VisibilityPath.arc_legs 的 direction——该字段对反向边
    （b→a 的边）未按路径方向翻转，直接使用会走出长弧。
    """
    forward = (a1 - a0) % TAU
    sweep = forward if forward <= TAU - forward else forward - TAU
    return [_to_att((center[0] + r * math.cos(a0 + sweep * j / n),
                     center[1] + r * math.sin(a0 + sweep * j / n)))
            for j in range(1, n)]


def _detour_polyline(p0, p1, zone) -> list[tuple[float, float]]:
    """单圆绕行路径（与求解器 safe_path 的 detour 分支同构）：
    p0 → 切点 → 短弧 → 切点 → p1，弧构造半径 = safe_radius + EPS。"""
    c = _to_km((zone["x"], zone["y"]))
    r = zone["r"] * KM_UNIT + SAFETY_MARGIN_KM + EPS_ARC_KM
    res = detour_path(_to_km(p0), _to_km(p1), c, r)
    if res is None:  # 端点无切点（求解器不会产生此情形），直线兜底
        return [p0, p1]
    points, _, _ = res
    a0 = math.atan2(points[1][1] - c[1], points[1][0] - c[0])
    a1 = math.atan2(points[2][1] - c[1], points[2][0] - c[0])
    return [p0] + _arc_sample(c, r, a0, a1) + [p1]


def _visibility_polyline(p0, p1, zones_active) -> list[tuple[float, float]]:
    """多圆可见性图绕行路径（与求解器 visibility 分支同构）。"""
    if not zones_active:
        return [p0, p1]
    disks = [(_to_km((z["x"], z["y"])), z["r"] * KM_UNIT + SAFETY_MARGIN_KM)
             for z in zones_active]
    vp = visibility_path(_to_km(p0), _to_km(p1), disks, margin=EPS_ARC_KM)
    if vp is None:
        return [p0, p1]
    pts_list = []
    for i in range(len(vp.points) - 1):
        pts_list.append(_to_att(vp.points[i]))
        leg = next((lg for lg in vp.arc_legs if lg.start == i), None)
        if leg is not None:
            a0 = math.atan2(vp.points[i][1] - leg.center[1],
                            vp.points[i][0] - leg.center[0])
            a1 = math.atan2(vp.points[i + 1][1] - leg.center[1],
                            vp.points[i + 1][0] - leg.center[0])
            pts_list += _arc_sample(leg.center, leg.radius_km, a0, a1)
    pts_list.append(_to_att(vp.points[-1]))
    return pts_list


# ---------------- 图9：禁飞区与绕行路径图 ----------------
def _draw_case_routes(ax, case: str, case_data: dict, arc3: dict, zones: list[dict],
                      annotate_zones: bool = True):
    """在子图上绘制某算例的禁飞区与全部航迹。

    路径绘制规则（几何诚实性）：
    - direct / wait_direct：直线——飞行时刻与禁飞窗不冲突（wait_direct
      已等待禁区解除），真实路径即直线；
    - detour / visibility：真实绕行路径——解档案只存起终点（其直线弦
      恰好穿过禁飞区圆），绘图用求解器同款几何（geometry.detour_path /
      visibility_path）重建"切点直线 + 沿禁飞区圆边弧线"的路径。
    """
    # 禁飞区
    for z in zones:
        ax.add_patch(Circle((z["x"], z["y"]), z["r"], facecolor="#C0392B",
                            alpha=0.15, edgecolor="#C0392B", linewidth=1.2,
                            zorder=1))
        if annotate_zones:
            ax.text(z["x"], z["y"] + z["r"] + 18, f"{z['id']}\n{z['win']}",
                    ha="center", va="bottom", fontsize=7.5, color="#8B1A1A")
    # 巡检点（等级着色）
    for pid, p in case_data["points"].items():
        ax.scatter(p["x"], p["y"], s=16, c=LEVEL_COLOR[p["level"]], alpha=0.7,
                   edgecolors="white", linewidths=0.3, zorder=3)
    ax.plot(*BASE, marker="*", markersize=13, color="gold",
            markeredgecolor="black", markeredgewidth=0.7, zorder=5)
    # 航迹（按 path_type 着色）
    drawn = set()
    zones_by_id = {z["id"]: z for z in zones}
    for sched in arc3["schedules"]:
        for seg in sched["segments"]:
            p0 = seg_point(case_data, seg["from_id"])
            p1 = seg_point(case_data, seg["to_id"])
            color, _ = PATH_STYLE[seg["path_type"]]
            if seg["path_type"] in ("detour", "visibility"):
                # 真实绕行路径：复用求解器同款几何重建（切点直线 + 禁飞区
                # 圆边弧线），detour 绕 affected_zones 声明的单圆、visibility
                # 绕受影响圆集——解档案只存起终点，直线弦恰好穿过禁飞区
                if seg["path_type"] == "detour" and seg.get("affected_zones"):
                    poly = _detour_polyline(
                        p0, p1, zones_by_id[seg["affected_zones"][0]])
                else:
                    active = [z for z in zones
                              if z["id"] in seg.get("affected_zones", [])]
                    poly = _visibility_polyline(p0, p1, active)
                xs, ys = zip(*poly)
                lw = 3.0 if seg["path_type"] == "detour" else 1.4
                ax.plot(xs, ys, "-", color=color, linewidth=lw, alpha=0.85,
                        zorder=2, solid_capstyle="round")
                if seg["path_type"] == "detour":
                    mid = poly[len(poly) // 2]
                    ax.text(mid[0], mid[1] + 5, "绕行", ha="center",
                            va="bottom", fontsize=7.5, color="#C44E52",
                            zorder=6)
            else:
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "-", color=color,
                        linewidth=1.2, alpha=0.75, zorder=2,
                        solid_capstyle="round")
            if seg["path_type"] not in drawn and seg["path_type"] != "direct":
                drawn.add(seg["path_type"])
    return drawn


def fig9_routes(archives: dict[str, dict], data: dict[str, dict],
                zones_all: dict[str, list]) -> plt.Figure:
    """图9：典型算例 Case2 禁飞区绕行路径大图（单图）。

    四算例中仅 Case2 存在真实绕行（detour 5 段 + visibility 6 段 +
    wait_direct 1 段；Case1 无任何非直飞段，Case3/4 仅各 1 段等待/可见性
    调整，绕行细节无展示价值），故选取 Case2 画单算例大图。图幅按数据
    纵横比动态调整（Case2 纵向跨度约为横向的 2.5 倍）。"""
    case = "Case2"
    arc3, case_data, zones = archives[case], data[case], zones_all[case]
    all_pts = list(case_data["points"].values())
    xs = [p["x"] for p in all_pts] + [z["x"] + z["r"] for z in zones] + \
         [z["x"] - z["r"] for z in zones]
    ys = [p["y"] for p in all_pts] + [z["y"] + z["r"] for z in zones] + \
         [z["y"] - z["r"] for z in zones]
    xspan = max(xs) - min(xs)
    yspan = max(ys) - min(ys)
    fig_h = 10.0
    fig_w = max(6.0, fig_h * xspan / yspan)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    _draw_case_routes(ax, case, case_data, arc3, zones, annotate_zones=True)
    ax.text(0.015, 0.98, case, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left", zorder=6,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2))
    ax.set_xlabel("X 坐标（单位：100 m）", fontsize=12)
    ax.set_ylabel("Y 坐标（单位：100 m）", fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=c, label=label)
               for _, (c, label) in PATH_STYLE.items()]
    handles += [plt.Line2D([0], [0], marker="s", ls="", color="#C0392B", alpha=0.25,
                           label="禁飞区")]
    ax.legend(handles=handles, loc="lower left", fontsize=10)
    fig.tight_layout()
    return fig


# ---------------- 图10：甘特图 ----------------
def fig10_gantt(archives: dict[str, dict], zones_all: dict[str, list]) -> plt.Figure:
    """图10：无人机任务甘特图（2×2，横轴为自 8:00 起的时间）。"""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9))
    for ax, case in zip(axes.flat, CASES):
        arc = archives[case]
        zones = zones_all[case]
        n = len(arc["schedules"])
        # 禁飞时段背景条（红色，贯穿所有无人机行）
        for z in zones:
            ax.axvspan(z["t0"] / 3600.0, z["t1"] / 3600.0, color="#C0392B",
                       alpha=0.12, zorder=0)
            ax.text((z["t0"] + z["t1"]) / 2 / 3600.0, n + 0.55, f"{z['id']} {z['win']}",
                    ha="center", fontsize=7, color="#8B1A1A")
        for i, sched in enumerate(arc["schedules"]):
            y = n - i  # 从上往下排列
            for seg in sched["segments"]:
                color, _ = PATH_STYLE[seg["path_type"]]
                ax.barh(y, (seg["arrive_s"] - seg["depart_s"]) / 3600.0,
                        left=seg["depart_s"] / 3600.0, height=0.62, color=color,
                        alpha=0.85, edgecolor="none", zorder=2)
                if seg["wait_s"] > 0:
                    ax.barh(y, seg["wait_s"] / 3600.0, left=seg["arrive_s"] / 3600.0,
                            height=0.62, color="#CCCCCC", edgecolor="none", zorder=2)
            for s, e in sched["service_intervals"]:
                ax.barh(y, (e - s) / 3600.0, left=s / 3600.0, height=0.62,
                        color="#F7B731", edgecolor="#8a6d1a", linewidth=0.4,
                        zorder=3)
        ax.text(0.015, 0.98, case, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="top", ha="left", zorder=6,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2))
        ax.set_xlabel("时刻（自 8:00 起，h）", fontsize=10)
        ax.set_ylabel("无人机编号（自上而下）", fontsize=10)
        ax.set_yticks(range(n, 0, -1))
        ax.set_yticklabels([f"UAV {i + 1}" for i in range(n)])
        ax.set_xlim(0, arc["metrics"]["S_max_s"] / 3600.0 * 1.06)
        ax.grid(axis="x", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=f"飞行（{label}）")
               for pt, (c, label) in PATH_STYLE.items()]
    handles += [plt.Rectangle((0, 0), 1, 1, color="#F7B731", label="巡检作业"),
                plt.Rectangle((0, 0), 1, 1, color="#CCCCCC", label="等待"),
                plt.Rectangle((0, 0), 1, 1, color="#C0392B", alpha=0.15,
                              label="禁飞时段")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    return fig


# ---------------- 图11：绕行代价对比 ----------------
def fig11_detour_cost(q1_all: dict[str, dict], q3_all: dict[str, dict]) -> plt.Figure:
    """图11：Q1 vs Q3 总飞行距离 + Q3 路径类型分布。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    x = np.arange(len(CASES))
    width = 0.35
    d1 = [sum(u["dist_km"] for u in q1_all[c]["uavs"]) for c in CASES]
    d3 = [q3_all[c]["metrics"]["total_distance_km"] for c in CASES]
    b1 = ax1.bar(x - width / 2, d1, width, color="#4C72B0", edgecolor="white",
                 label="问题1（无禁飞区）", zorder=3)
    b2 = ax1.bar(x + width / 2, d3, width, color="#55A868", edgecolor="white",
                 label="问题3（含禁飞区）", zorder=3)
    for b in list(b1) + list(b2):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 12,
                 f"{b.get_height():.0f}", ha="center", fontsize=9)
    for i, c in enumerate(CASES):
        inc = (d3[i] - d1[i]) / d1[i] * 100
        ax1.text(x[i], max(d1[i], d3[i]) * 1.14, f"+{inc:.1f}%", ha="center",
                 fontsize=10, color="#8B1A1A", fontweight="bold")
    ax1.set_ylabel("总飞行距离（km）", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)

    # 路径类型段数分布（堆叠条）
    bottom = np.zeros(len(CASES))
    for pt, (color, label) in PATH_STYLE.items():
        counts = [sum(1 for s in q3_all[c]["schedules"]
                      for seg in s["segments"] if seg["path_type"] == pt)
                  for c in CASES]
        ax2.bar(x, counts, bottom=bottom, width=0.55, color=color,
                edgecolor="white", label=label, zorder=3)
        bottom += counts
    ax2.set_xticks(x)
    ax2.set_xticklabels(CASES)
    ax2.set_ylabel("飞行段数量", fontsize=11)
    ax2.set_xlabel("测试算例", fontsize=11)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = {c: load_case(c) for c in CASES}
    zones_all = {c: load_zones(c) for c in CASES}
    q3_all = {c: load_q3(c) for c in CASES}
    q1_all = {c: load_q1(c) for c in CASES}

    jobs = [
        (fig9_routes(q3_all, data, zones_all), "fig9_禁飞区与绕行路径"),
        (fig10_gantt(q3_all, zones_all), "fig10_无人机任务甘特图"),
        (fig11_detour_cost(q1_all, q3_all), "fig11_绕行代价对比"),
    ]
    for fig, name in jobs:
        for ext in ("png", "pdf"):
            save_fig(fig, FIG_DIR / f"{name}.{ext}", dpi=300)
        plt.close(fig)
        print(f"完成：{name}")


if __name__ == "__main__":
    main()
