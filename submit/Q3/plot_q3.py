import json
import math
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import pandas as pd
from matplotlib.patches import Circle
from matplotlib.patheffects import withStroke
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False
REPO_ROOT = Path(__file__).resolve().parents[1]
CASES = ["Case1", "Case2", "Case3", "Case4"]
FIG_DIR = REPO_ROOT / "Q3" / "figs"
ARCHIVE = REPO_ROOT / "outputs" / "workbooks"
Q2_ARCHIVE = REPO_ROOT / "enhanced_run_20260817_e2000" / "q2" / "strict"
ATTACH_XLSX = REPO_ROOT / "attachment" / "附件1.xlsx"
ATTACH_ZONES = REPO_ROOT / "attachment" / "附件2.xlsx"
START_HOUR = 8.0
BASE = (0.0, 0.0)
LEVEL_VISITS = {"I": 3, "II": 2, "III": 1}
PT_COLOR = "#DDDDDD"
ANCHOR_COLOR = "#333333"
DIRECT_COLOR = "#E0E0E0"
DETOUR_COLOR = "#C0392B"
ZONE_FACE = "#E74C3C"
ZONE_EDGE = "#8B1A1A"
FLIGHT_COLOR = "#9DC3E6"
SERVICE_TICK = "#2F5D8C"
WAIT_COLOR = "#666666"
ROW_BG = "#F5F5F5"
Q2_BASE_COLOR = "#DD8452"
Q3_COLOR = "#55A868"
def save_fig(fig, path, dpi=300):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"图片已保存到：{path}（dpi={dpi}）")
def load_case(case: str) -> dict:
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
def seg_point(case_data: dict, node_id: int):
    if node_id == 0:
        return BASE
    pid = case_data["task_of"][node_id]
    p = case_data["points"][pid]
    return (p["x"], p["y"])
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "Q3"))
from geometry import detour_path, visibility_path, TAU
from domain import EPS_ARC_KM, SAFETY_MARGIN_KM
KM_UNIT = 0.1
def _to_km(p) -> tuple[float, float]:
    return (p[0] * KM_UNIT, p[1] * KM_UNIT)
def _to_att(p) -> tuple[float, float]:
    return (p[0] / KM_UNIT, p[1] / KM_UNIT)
def _arc_sample(center, r, a0, a1, n=36) -> list[tuple[float, float]]:
    forward = (a1 - a0) % TAU
    sweep = forward if forward <= TAU - forward else forward - TAU
    return [_to_att((center[0] + r * math.cos(a0 + sweep * j / n),
                     center[1] + r * math.sin(a0 + sweep * j / n)))
            for j in range(1, n)]
def _detour_polyline(p0, p1, zone):
    c = _to_km((zone["x"], zone["y"]))
    r = zone["r"] * KM_UNIT + SAFETY_MARGIN_KM + EPS_ARC_KM
    res = detour_path(_to_km(p0), _to_km(p1), c, r)
    if res is None:
        return [p0, p1], []
    points, _, _ = res
    a0 = math.atan2(points[1][1] - c[1], points[1][0] - c[0])
    a1 = math.atan2(points[2][1] - c[1], points[2][0] - c[0])
    poly = [p0] + _arc_sample(c, r, a0, a1) + [p1]
    return poly, [_to_att(points[1]), _to_att(points[2])]
def _visibility_polyline(p0, p1, zones_active):
    if not zones_active:
        return [p0, p1], []
    disks = [(_to_km((z["x"], z["y"])), z["r"] * KM_UNIT + SAFETY_MARGIN_KM)
             for z in zones_active]
    vp = visibility_path(_to_km(p0), _to_km(p1), disks, margin=EPS_ARC_KM)
    if vp is None:
        return [p0, p1], []
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
    tangs = [_to_att(vp.points[i]) for i in range(1, len(vp.points) - 1)]
    return pts_list, tangs
def _draw_case_routes(ax, case: str, case_data: dict, arc3: dict, zones: list[dict]):
    zones_by_id = {z["id"]: z for z in zones}
    for z in zones:
        ax.add_patch(Circle((z["x"], z["y"]), z["r"], facecolor=ZONE_FACE,
                            alpha=0.08, edgecolor=ZONE_EDGE, linewidth=1.0,
                            linestyle="--", zorder=1))
        ax.text(z["x"], z["y"], z["id"], ha="center", va="center",
                fontsize=7.5, color=ZONE_EDGE, fontweight="bold", zorder=4)
    for pid, p in case_data["points"].items():
        ax.scatter(p["x"], p["y"], s=8, c=PT_COLOR, alpha=0.9,
                   edgecolors="none", zorder=2)
    anchors = set()
    best_arc = None
    for sched in arc3["schedules"]:
        for seg in sched["segments"]:
            p0 = seg_point(case_data, seg["from_id"])
            p1 = seg_point(case_data, seg["to_id"])
            if seg["path_type"] in ("detour", "visibility"):
                if seg["path_type"] == "detour" and seg.get("affected_zones"):
                    poly, tangs = _detour_polyline(
                        p0, p1, zones_by_id[seg["affected_zones"][0]])
                else:
                    active = [z for z in zones
                              if z["id"] in seg.get("affected_zones", [])]
                    poly, tangs = _visibility_polyline(p0, p1, active)
                xs, ys = zip(*poly)
                ax.plot(xs, ys, "-", color=DETOUR_COLOR, linewidth=2.2,
                        alpha=0.95, zorder=3, solid_capstyle="round",
                        path_effects=[withStroke(linewidth=2.9,
                                                 foreground="white")])
                if p0 != BASE:
                    anchors.add(p0)
                if p1 != BASE:
                    anchors.add(p1)
                for t in tangs:
                    anchors.add(t)
                if seg["path_type"] == "detour":
                    arc_len = sum(math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
                                  for i in range(len(xs) - 1))
                    if best_arc is None or arc_len > best_arc[0]:
                        best_arc = (arc_len, poly[len(poly) // 2])
            else:
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "-", color=DIRECT_COLOR,
                        linewidth=0.5, alpha=0.25, zorder=2,
                        solid_capstyle="round")
    for a in anchors:
        ax.plot(*a, marker="o", markersize=4.5, color=ANCHOR_COLOR,
                markeredgecolor="white", markeredgewidth=0.3, zorder=4)
    ax.plot(*BASE, marker="*", markersize=15, color="black",
            markeredgecolor="white", markeredgewidth=0.6, zorder=5)
    if best_arc is not None:
        mid = best_arc[1]
        ax.annotate("绕行", xy=mid, xytext=(mid[0] + 22, mid[1] - 14),
                    fontsize=8, color=DETOUR_COLOR, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", lw=0.9, color=DETOUR_COLOR))
def fig9_routes(archives: dict[str, dict], data: dict[str, dict],
                zones_all: dict[str, list]) -> plt.Figure:
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
    _draw_case_routes(ax, case, case_data, arc3, zones)
    ax.text(0.015, 0.98, case, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left", zorder=6,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2))
    ax.set_xlabel("X 坐标（单位：100 m）", fontsize=12)
    ax.set_ylabel("Y 坐标（单位：100 m）", fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    handles = [plt.Line2D([0], [0], ls="-", color=DIRECT_COLOR, lw=0.5,
                          label="直飞"),
               plt.Line2D([0], [0], ls="-", color=DETOUR_COLOR, lw=2.2,
                          label="绕行"),
               plt.Line2D([0], [0], ls="--", color=ZONE_EDGE, lw=1.0,
                          label="禁飞区"),
               plt.Line2D([0], [0], marker="o", ls="", color=PT_COLOR,
                          label="巡检点"),
               plt.Line2D([0], [0], marker="o", ls="", color=ANCHOR_COLOR,
                          label="绕行锚点"),
               plt.Line2D([0], [0], marker="*", ls="", color="black",
                          label="基地")]
    ax.legend(handles=handles, loc="lower left", fontsize=8, ncol=2,
              frameon=True, framealpha=0.85)
    fig.tight_layout()
    return fig
def fig10_gantt(archives: dict[str, dict], zones_all: dict[str, list]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9))
    for ax, case in zip(axes.flat, CASES):
        arc = archives[case]
        zones = zones_all[case]
        n = len(arc["schedules"])
        smax_h = arc["metrics"]["S_max_s"] / 3600.0
        total_wait = sum(seg.get("wait_s", 0)
                         for s in arc["schedules"] for seg in s["segments"])
        for i in range(n):
            ax.axhspan(n - i - 0.31, n - i + 0.31, color=ROW_BG, zorder=0)
        spans = [(z["t0"], z["t1"]) for z in zones if z["t0"] < z["t1"]]
        over = []
        for i in range(len(spans)):
            for j in range(i + 1, len(spans)):
                a, b = max(spans[i][0], spans[j][0]), min(spans[i][1], spans[j][1])
                if a < b:
                    over.append([a, b])
        if over:
            over.sort()
            merged_o = [over[0]]
            for a, b in over[1:]:
                if a <= merged_o[-1][1]:
                    merged_o[-1][1] = max(merged_o[-1][1], b)
                else:
                    merged_o.append([a, b])
            for a, b in merged_o:
                ax.axvspan(a / 3600.0, b / 3600.0, color=ZONE_FACE,
                           alpha=0.14 if total_wait == 0 else 0.20,
                           zorder=0.5)
        for z in zones:
            if z["t0"] >= z["t1"]:
                continue
            ax.axvspan(z["t0"] / 3600.0, z["t1"] / 3600.0, color=ZONE_FACE,
                       alpha=0.08 if total_wait == 0 else 0.12,
                       zorder=0.5)
            ax.text(z["t0"] / 3600.0 + 0.04, n - 0.02, z["id"],
                    ha="left", va="top", fontsize=6.5, color=ZONE_EDGE,
                    fontweight="bold", zorder=4)
            for t in (z["t0"], z["t1"]):
                ax.axvline(t / 3600.0, color="#B0B0B0", ls="--", lw=0.5,
                           alpha=0.8, zorder=0.6)
        for i, sched in enumerate(arc["schedules"]):
            y = n - i
            ivs = [(seg["depart_s"], seg["arrive_s"])
                   for seg in sched["segments"]]
            ivs += [(s, e) for s, e in sched["service_intervals"]]
            ivs.sort()
            merged = []
            for s, e in ivs:
                if merged and s <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((s, e))
            for s, e in merged:
                ax.barh(y, (e - s) / 3600.0, left=s / 3600.0, height=0.62,
                        color=FLIGHT_COLOR, alpha=0.9, edgecolor="none",
                        zorder=2)
            for s, e in sched["service_intervals"]:
                ax.vlines((s + e) / 2 / 3600.0, y - 0.31, y + 0.31,
                          color=SERVICE_TICK, linewidth=0.5, zorder=2.5)
            for seg in sched["segments"]:
                if seg["wait_s"] > 0:
                    w = seg["wait_s"] / 3600.0
                    xw = seg["arrive_s"] / 3600.0
                    ax.barh(y, w, left=xw, height=0.62, color=WAIT_COLOR,
                            edgecolor="none", zorder=3)
                    ax.text(xw + w / 2, y + 0.42, f"{seg['wait_s']} s",
                            ha="center", va="bottom", fontsize=6.8,
                            color="#333333", zorder=5,
                            bbox=dict(facecolor="white", alpha=0.85,
                                      edgecolor="none", pad=1.5))
                    if any(z["t0"] / 3600.0 <= xw <= z["t1"] / 3600.0
                           for z in zones if z["t0"] < z["t1"]):
                        ax.plot([xw, xw], [y + 0.31, n - 0.02], ls="--",
                                color=ZONE_EDGE, lw=0.6, alpha=0.75,
                                zorder=3.5)
            end_h = merged[-1][1] / 3600.0
            x_t, ha_t = end_h + 0.05, "left"
            if x_t > smax_h - 0.05:
                x_t, ha_t = smax_h - 0.05, "right"
            ax.text(x_t, y, f"{end_h:.2f} h", ha=ha_t, va="center",
                    fontsize=6.8, color="#333333", zorder=5,
                    bbox=dict(facecolor="white", alpha=0.85,
                              edgecolor="none", pad=1))
        ax.text(0.015, 0.98, case, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="top", ha="left", zorder=6,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2))
        ax.text(0.985, 0.98, f"S_max = {smax_h:.2f} h\n总等待 = {total_wait} s",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                fontweight="bold", zorder=6,
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=3))
        if total_wait == 0:
            ax.text(0.015, 0.045, "无等待：任务时序完全避开禁飞窗口",
                    transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=6.8, color="#888888", zorder=6)
        ax.set_xlabel("时刻（自 8:00 起，h）", fontsize=10)
        ax.set_yticks(range(n, 0, -1))
        ax.set_yticklabels([f"UAV {i + 1}" for i in range(n)])
        ax.set_xlim(0, smax_h)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.grid(axis="x", which="major", alpha=0.15, color="#CCCCCC")
        for k in range(n + 1):
            ax.axhline(n - k + 0.5, color="#DDDDDD", lw=0.4, zorder=1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=FLIGHT_COLOR,
                             label="任务执行时段"),
               plt.Rectangle((0, 0), 1, 1, color=WAIT_COLOR,
                             label="必要等待（标注时长）"),
               plt.Rectangle((0, 0), 1, 1, color=ZONE_FACE, alpha=0.18,
                             label="禁飞时段")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    return fig
def q2_distance_km(case: str, case_data: dict) -> float:
    q2 = json.load(open(Q2_ARCHIVE / f"{case}.json", encoding="utf-8"))
    total = 0.0
    for route in q2["task_routes"]:
        prev = BASE
        for t in route:
            p = case_data["points"][case_data["task_of"][t]]
            total += math.hypot(p["x"] - prev[0], p["y"] - prev[1]) * KM_UNIT
            prev = (p["x"], p["y"])
        total += math.hypot(-prev[0], -prev[1]) * KM_UNIT
    return total
def fig11_detour_cost(q3_all: dict[str, dict], data: dict[str, dict]) -> plt.Figure:
    q2_tmax_h = []
    q2_dist_km = []
    for c in CASES:
        q2 = json.load(open(Q2_ARCHIVE / f"{c}.json", encoding="utf-8"))
        q2_tmax_h.append(q2["metrics"]["Tmax_s"] / 3600.0)
        q2_dist_km.append(q2_distance_km(c, data[c]))
    s3 = [q3_all[c]["metrics"]["S_max_s"] / 3600.0 for c in CASES]
    d3 = [q3_all[c]["metrics"]["total_distance_km"] for c in CASES]
    w3 = [sum(seg.get("wait_s", 0) for s in q3_all[c]["schedules"]
              for seg in s["segments"]) for c in CASES]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 11.5), sharex=True)
    x = np.arange(len(CASES))
    width = 0.35
    def _grouped(ax, base_vals, q3_vals, unit):
        b1 = ax.bar(x - width / 2, base_vals, width, color=Q2_BASE_COLOR,
                    edgecolor="white", label="问题 2 无禁飞区方案", zorder=3)
        b2 = ax.bar(x + width / 2, q3_vals, width, color=Q3_COLOR,
                    edgecolor="white", label="问题 3 方案", zorder=3)
        for b, v in zip(list(b1) + list(b2), base_vals + q3_vals):
            ax.text(b.get_x() + b.get_width() / 2, v + max(base_vals) * 0.015,
                    f"{v:.3f}" if unit == "h" else f"{v:.1f}",
                    ha="center", fontsize=8, zorder=4)
        for i in range(len(CASES)):
            inc = (q3_vals[i] - base_vals[i]) / base_vals[i] * 100
            ax.text(x[i], max(base_vals[i], q3_vals[i]) * 1.07,
                    f"+{inc:.1f}%", ha="center", fontsize=10.5,
                    color=ZONE_EDGE, fontweight="bold", zorder=4)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9, loc="upper left")
    _grouped(ax1, q2_tmax_h, s3, "h")
    ax1.set_ylabel("总体完成时间（h）", fontsize=11)
    _grouped(ax2, q2_dist_km, d3, "km")
    ax2.set_ylabel("总飞行距离（km）", fontsize=11)
    ax3.bar(x, w3, width * 1.1, color=Q3_COLOR, edgecolor="white", zorder=3)
    for i, w in enumerate(w3):
        ax3.text(x[i], w + max(w3) * 0.03, f"{w} s", ha="center",
                 fontsize=10, color=ZONE_EDGE, fontweight="bold", zorder=4)
    ax3.set_ylabel("总必要等待（s）", fontsize=11)
    ax3.set_xlabel("测试算例", fontsize=11)
    ax3.grid(axis="y", alpha=0.3)
    ax3.set_xticks(x)
    ax3.set_xticklabels(CASES)
    fig.tight_layout()
    return fig
def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = {c: load_case(c) for c in CASES}
    zones_all = {c: load_zones(c) for c in CASES}
    q3_all = {c: load_q3(c) for c in CASES}
    jobs = [
        (fig9_routes(q3_all, data, zones_all), "fig9_禁飞区与绕行路径"),
        (fig10_gantt(q3_all, zones_all), "fig10_无人机任务甘特图"),
        (fig11_detour_cost(q3_all, data), "fig11_绕行代价对比"),
    ]
    for fig, name in jobs:
        for ext in ("png", "pdf"):
            save_fig(fig, FIG_DIR / f"{name}.{ext}", dpi=300)
        plt.close(fig)
        print(f"完成：{name}")
if __name__ == "__main__":
    main()
