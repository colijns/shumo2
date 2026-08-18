import math
from dataclasses import dataclass
import numpy as np
from .domain import EPS_ARC_KM, EPS_VER_KM, SERVICE_S, SPEED_KMH, SegmentRecord, Task
from .geometry import (
    TAU,
    detour_path,
    point_inside,
    visibility_path,
)
from .io import load_problem
KM_PER_S = SPEED_KMH / 3600.0
STRAIGHT_SAMPLES = 50
ARC_SAMPLES = 100
@dataclass(frozen=True, slots=True)
class Sample:
    x_km: float
    y_km: float
    t_s: float
@dataclass(frozen=True, slots=True)
class Replay:
    samples: tuple[Sample, ...]
    length_km: float
    flight_s: int
def _node_position(problem, node_id: int) -> tuple[float, float]:
    if node_id == 0:
        return (0.0, 0.0)
    task: Task = problem.tasks[node_id - 1]
    return (task.x_km, task.y_km)
def _straight(start: np.ndarray, end: np.ndarray, t0: float, t1: float) -> list[Sample]:
    return [
        Sample(
            float(start[0] + frac * (end[0] - start[0])),
            float(start[1] + frac * (end[1] - start[1])),
            t0 + frac * (t1 - t0),
        )
        for frac in np.linspace(0.0, 1.0, STRAIGHT_SAMPLES)
    ]
def _arc(center, radius: float, theta_a: float, theta_b: float, direction: int,
         t0: float, t1: float) -> list[Sample]:
    if direction > 0:
        sweep = (theta_b - theta_a) % TAU
    else:
        sweep = -((theta_a - theta_b) % TAU)
    samples = []
    for frac in np.linspace(0.0, 1.0, ARC_SAMPLES):
        theta = theta_a + sweep * frac
        samples.append(
            Sample(
                float(center[0] + radius * math.cos(theta)),
                float(center[1] + radius * math.sin(theta)),
                t0 + frac * (t1 - t0),
            )
        )
    return samples
def _flight_duration(points, arc_lengths: list[float]) -> int:
    total = 0
    for index in range(len(points) - 1):
        length = arc_lengths[index] if arc_lengths[index] is not None else float(
            np.linalg.norm(points[index + 1] - points[index])
        )
        total += int(math.ceil(length / KM_PER_S))
    return total
def replay_segment(problem, seg: SegmentRecord) -> Replay | None:
    p = np.asarray(_node_position(problem, seg.from_id), dtype=float)
    q = np.asarray(_node_position(problem, seg.to_id), dtype=float)
    start_t = float(seg.depart_s)
    samples: list[Sample] = []
    if seg.path_type in ("direct", "wait_direct"):
        length = float(np.linalg.norm(q - p))
        duration = int(math.ceil(length / KM_PER_S))
        samples = _straight(p, q, start_t, start_t + duration)
        return Replay(tuple(samples), length, duration)
    if seg.path_type == "detour":
        if len(seg.affected_zones) != 1:
            return None
        zone = next(z for z in problem.zones if z.zone_id == seg.affected_zones[0])
        result = detour_path(p, q, (zone.cx_km, zone.cy_km), zone.safe_radius() + EPS_ARC_KM)
        if result is None:
            return None
        points, arc_angle, direction = result
        points = [np.asarray(pt, dtype=float) for pt in points]
        arc_radius = zone.safe_radius() + EPS_ARC_KM
        leg_lengths: list[float | None] = [None, arc_angle * arc_radius, None]
        seg_len = [float(np.linalg.norm(points[1] - points[0])),
                   arc_angle * arc_radius,
                   float(np.linalg.norm(points[3] - points[2]))]
        times = [0.0]
        for length in seg_len:
            times.append(times[-1] + int(math.ceil(length / KM_PER_S)))
        duration = times[-1]
        samples.extend(_straight(points[0], points[1], start_t + times[0], start_t + times[1]))
        c = np.asarray((zone.cx_km, zone.cy_km), dtype=float)
        theta_a = math.atan2(points[1][1] - c[1], points[1][0] - c[0])
        theta_b = math.atan2(points[2][1] - c[1], points[2][0] - c[0])
        samples.extend(_arc(c, zone.safe_radius() + EPS_ARC_KM, theta_a, theta_b, direction,
                            start_t + times[1], start_t + times[2]))
        samples.extend(_straight(points[2], points[3], start_t + times[2], start_t + times[3]))
        return Replay(tuple(samples), sum(seg_len), duration)
    if seg.path_type == "visibility":
        obstacles = [
            ((zone.cx_km, zone.cy_km), zone.safe_radius())
            for zone in problem.zones
            if zone.zone_id in seg.affected_zones
        ]
        path = visibility_path(p, q, obstacles, margin=EPS_ARC_KM)
        if path is None:
            return None
        length = float(path.length_km)
        leg_lengths: list[float | None] = []
        for index in range(len(path.points) - 1):
            arc = None
            for leg in path.arc_legs:
                if leg.start == index:
                    arc = leg
                    break
            if arc is None:
                leg_lengths.append(None)
            else:
                theta_a = math.atan2(path.points[index][1] - arc.center[1], path.points[index][0] - arc.center[0])
                theta_b = math.atan2(path.points[index + 1][1] - arc.center[1], path.points[index + 1][0] - arc.center[0])
                forward = (theta_b - theta_a) % TAU
                leg_lengths.append(arc.radius_km * min(forward, TAU - forward))
        duration = _flight_duration(path.points, leg_lengths)
        times = [0.0]
        for index in range(len(path.points) - 1):
            llen = leg_lengths[index] if leg_lengths[index] is not None else float(
                np.linalg.norm(path.points[index + 1] - path.points[index])
            )
            times.append(times[-1] + int(math.ceil(llen / KM_PER_S)))
        for index in range(len(path.points) - 1):
            leg = None
            for cand in path.arc_legs:
                if cand.start == index:
                    leg = cand
                    break
            if leg is None:
                samples.extend(_straight(path.points[index], path.points[index + 1],
                                         start_t + times[index], start_t + times[index + 1]))
            else:
                theta_a = math.atan2(path.points[index][1] - leg.center[1], path.points[index][0] - leg.center[0])
                theta_b = math.atan2(path.points[index + 1][1] - leg.center[1], path.points[index + 1][0] - leg.center[0])
                samples.extend(_arc(leg.center, leg.radius_km, theta_a, theta_b, leg.direction,
                                    start_t + times[index], start_t + times[index + 1]))
        return Replay(tuple(samples), length, duration)
    return None
def _zone_violations(problem, samples, where: str) -> list[str]:
    violations = []
    for zone in problem.zones:
        for sample in samples:
            if zone.start_s <= sample.t_s <= zone.end_s:
                distance = math.hypot(sample.x_km - zone.cx_km, sample.y_km - zone.cy_km)
                if distance <= zone.safe_radius() - EPS_VER_KM:
                    violations.append(
                        f"{where}: inside {zone.zone_id} at t={sample.t_s:.0f} "
                        f"(d={distance:.6f} <= safe {zone.safe_radius():.6f})"
                    )
                    break
    return violations
def verify_archive(problem, archive: dict) -> list[str]:
    violations: list[str] = []
    task_routes = archive.get("task_routes")
    schedules = archive.get("schedules")
    served = [task_id for route in (task_routes or []) for task_id in route]
    if sorted(served) != list(range(1, len(problem.tasks) + 1)):
        violations.append(
            f"1: served set mismatch (expected 1..{len(problem.tasks)}, got {sorted(served)})"
        )
    per_route = []
    for index, route in enumerate(task_routes or []):
        if not route:
            violations.append(f"2: uav {index}: empty route")
        prev_point = None
        for task_id in route:
            point_id = problem.tasks[task_id - 1].point_id
            if point_id == prev_point:
                violations.append(f"3: uav {index}: adjacent same point {point_id}")
            prev_point = point_id
        per_route.append(route)
    for schedule in (schedules or []):
        prev_arrive: int | None = None
        for segment in schedule.get("segments", []):
            record = SegmentRecord(
                segment["from_id"], segment["to_id"], segment["path_type"],
                segment["depart_s"], segment["arrive_s"], segment["wait_s"],
                segment["distance_km"], tuple(segment.get("affected_zones", [])),
            )
            replay = replay_segment(problem, record)
            if replay is None:
                violations.append(
                    f"4: uav {schedule['uav_id']}: cannot replay {segment['path_type']} "
                    f"{segment['from_id']}->{segment['to_id']}"
                )
                prev_arrive = segment["arrive_s"]
                continue
            if replay.flight_s != segment["arrive_s"] - segment["depart_s"]:
                violations.append(
                    f"6: uav {schedule['uav_id']}: flight time mismatch "
                    f"({replay.flight_s} != {segment['arrive_s'] - segment['depart_s']})"
                )
            expected = 0 if segment["from_id"] == 0 else (prev_arrive or 0) + SERVICE_S
            if segment["depart_s"] - segment["wait_s"] != expected:
                violations.append(
                    f"6: uav {schedule['uav_id']}: depart chain broken at "
                    f"{segment['from_id']}->{segment['to_id']}"
                )
            violations.extend(
                _zone_violations(problem, replay.samples,
                                 f"4: uav {schedule['uav_id']} {segment['from_id']}->{segment['to_id']}")
            )
            if segment["to_id"] != 0:
                task = problem.tasks[segment["to_id"] - 1]
                for zone in problem.zones:
                    if segment["arrive_s"] > zone.end_s or segment["arrive_s"] + SERVICE_S < zone.start_s:
                        continue
                    if point_inside((task.x_km, task.y_km), (zone.cx_km, zone.cy_km), zone.safe_radius()):
                        violations.append(
                            f"5: uav {schedule['uav_id']}: service at task {segment['to_id']} "
                            f"inside active {zone.zone_id} [{segment['arrive_s']}, {segment['arrive_s'] + SERVICE_S}]"
                        )
            if segment["wait_s"] > 0:
                wx, wy = _node_position(problem, segment["from_id"])
                for zone in problem.zones:
                    wait_lo = segment["depart_s"] - segment["wait_s"]
                    if zone.end_s < wait_lo or zone.start_s > segment["depart_s"]:
                        continue
                    if point_inside((wx, wy), (zone.cx_km, zone.cy_km), zone.safe_radius()):
                        violations.append(
                            f"5: uav {schedule['uav_id']}: waiting at node {segment['from_id']} "
                            f"inside active {zone.zone_id}"
                        )
            prev_arrive = segment["arrive_s"]
        last_segment = schedule["segments"][-1] if schedule["segments"] else None
        if last_segment is None or last_segment["to_id"] != 0:
            violations.append(f"2: uav {schedule['uav_id']}: route does not return to base")
    for schedule in (schedules or []):
        segments = schedule.get("segments", [])
        if not segments or segments[0]["depart_s"] - segments[0]["wait_s"] != 0:
            violations.append(f"6: uav {schedule['uav_id']}: first depart_s != 0")
        for previous, current in zip(segments, segments[1:]):
            expected = previous["arrive_s"] + SERVICE_S
            if current["depart_s"] - current["wait_s"] != expected:
                violations.append(
                    f"6: uav {schedule['uav_id']}: timeline gap between legs "
                    f"({current['depart_s'] - current['wait_s']} != {expected})"
                )
        if segments and schedule.get("S_k_s") != segments[-1]["arrive_s"]:
            violations.append(
                f"6: uav {schedule['uav_id']}: S_k {schedule['S_k_s']} != final arrival {segments[-1]['arrive_s']}"
            )
        flight = sum(
            seg["arrive_s"] - seg["depart_s"] for seg in segments
        )
        wait = sum(seg["wait_s"] for seg in segments)
        distance = sum(seg["distance_km"] for seg in segments)
        if flight != schedule.get("flight_s") or wait != schedule.get("wait_s"):
            violations.append(
                f"7: uav {schedule['uav_id']}: sums mismatch "
                f"(flight {flight} != {schedule.get('flight_s')}, wait {wait} != {schedule.get('wait_s')})"
            )
        if abs(distance - schedule.get("distance_km", 0.0)) > 1e-6:
            violations.append(f"7: uav {schedule['uav_id']}: distance mismatch")
    if schedules:
        s_ks = [schedule["S_k_s"] for schedule in schedules]
        s_max, s_min = max(s_ks), min(s_ks)
        metrics = archive.get("metrics", {})
        if metrics.get("S_max_s") != s_max or metrics.get("S_min_s") != s_min:
            violations.append(
                f"7: metrics S_max/S_min mismatch (archive {metrics.get('S_max_s')}/{metrics.get('S_min_s')} "
                f"!= computed {s_max}/{s_min})"
            )
        if metrics.get("delta_s") != s_max - s_min:
            violations.append("7: metrics delta mismatch")
        computed_sum = sum(s_ks)
        if metrics.get("sum_T_s") != computed_sum:
            violations.append(f"7: metrics sum_T mismatch ({metrics.get('sum_T_s')} != {computed_sum})")
        computed_wait = sum(schedule["wait_s"] for schedule in schedules)
        if metrics.get("total_wait_s") != computed_wait:
            violations.append(f"7: metrics total_wait mismatch ({metrics.get('total_wait_s')} != {computed_wait})")
        computed_distance = round(sum(schedule["distance_km"] for schedule in schedules), 6)
        if abs(metrics.get("total_distance_km", -1.0) - computed_distance) > 1e-6:
            violations.append(f"7: metrics total_distance mismatch")
    if task_routes:
        schedule_routes = [
            [seg["to_id"] for seg in schedule.get("segments", []) if seg["to_id"] != 0]
            for schedule in (schedules or [])
        ]
        if [list(route) for route in task_routes] != schedule_routes:
            violations.append("8: task_routes disagree with schedule task_route fields")
    cap = archive.get("config", {}).get("nine_hour_cap_s")
    if cap:
        for schedule in (schedules or []):
            if schedule["S_k_s"] > cap:
                violations.append(f"9: uav {schedule['uav_id']} S_k {schedule['S_k_s']} > {cap}")
    return violations
def _task_routes_from_csv(problem, csv_path) -> list[list[int]]:
    import pandas as pd
    frame = pd.read_csv(csv_path, encoding="utf-8")
    routes: dict[int, list[int]] = {}
    for row in frame.itertuples(index=False):
        uav_id = int(row.uav_id)
        routes.setdefault(uav_id, [])
        to_id = int(row.to_id)
        if to_id != 0:
            routes[uav_id].append(to_id)
    ordered = [routes[uav_id] for uav_id in sorted(routes)]
    if sorted(task_id for route in ordered for task_id in route) != list(range(1, len(problem.tasks) + 1)):
        raise ValueError(f"timetable {csv_path} does not cover tasks 1..{len(problem.tasks)}")
    return ordered
def _point_routes_from_workbook(problem, workbook_path, case: str) -> list[list[int]]:
    import pandas as pd
    max_visits: dict[int, int] = {}
    for task in problem.tasks:
        max_visits[task.point_id] = max_visits.get(task.point_id, 0) + 1
    frame = pd.read_excel(workbook_path, sheet_name=case)
    routes: list[list[int]] = []
    for _, row in frame.iterrows():
        points = [
            int(value) for column, value in row.items()
            if column != "UAV ID" and pd.notna(value)
        ]
        counts: dict[int, int] = {}
        for point_id in points:
            counts[point_id] = counts.get(point_id, 0) + 1
            if counts[point_id] > max_visits.get(point_id, 0):
                raise ValueError(f"workbook {case}: too many visits of point {point_id}")
        routes.append(points)
    return routes
def verify_all(cases, config, repository_root=None) -> list[str]:
    import json as _json
    from pathlib import Path as _Path
    from .io import OUT_RELATIVE, Q3_OUT_RELATIVE, archive_sha256, load_archive
    root = _Path(repository_root).resolve() if repository_root else _Path(__file__).resolve().parents[1]
    report_dir = root / Q3_OUT_RELATIVE / "reports"
    manifest_path = report_dir / "run_manifest.json"
    violations: list[str] = []
    report: dict = {"schema_version": "q3-verify-report-v1", "cases": {}}
    if not manifest_path.is_file():
        violations.append(f"manifest missing: {manifest_path}")
    else:
        manifest = load_archive(manifest_path)
        for case in cases:
            entry = (manifest.get("cases") or {}).get(case)
            if entry is None:
                violations.append(f"{case}: absent from manifest")
                report["cases"][case] = {"verified": False, "violations": ["absent from manifest"]}
                continue
            case_violations: list[str] = []
            archive_path = root / entry["archive"]
            problem = load_problem(case, repository_root=root)
            archive = load_archive(archive_path)
            case_violations.extend(verify_archive(problem, archive))
            if entry.get("archive_sha256") != archive_sha256(archive_path):
                case_violations.append(f"8: archive sha256 mismatch")
            try:
                csv_routes = _task_routes_from_csv(problem, root / entry["timetable"])
                if [list(route) for route in archive.get("task_routes", [])] != csv_routes:
                    case_violations.append("8: timetable disagrees with archive task_routes")
            except (OSError, ValueError) as exc:
                case_violations.append(f"8: timetable unreadable ({exc})")
            workbook = root / OUT_RELATIVE / "result3.xlsx"
            if workbook.is_file():
                try:
                    xlsx_points = _point_routes_from_workbook(problem, workbook, case)
                    point_by_task = {task.task_id: task.point_id for task in problem.tasks}
                    archive_points = [
                        [point_by_task[task_id] for task_id in route]
                        for route in archive.get("task_routes", [])
                    ]
                    if xlsx_points != archive_points:
                        case_violations.append("8: result3.xlsx disagrees with archive task_routes")
                except (ValueError, KeyError) as exc:
                    case_violations.append(f"8: workbook unreadable ({exc})")
            else:
                case_violations.append("8: result3.xlsx missing")
            violations.extend(f"{case}: {item}" for item in case_violations)
            report["cases"][case] = {"verified": not case_violations, "violations": case_violations}
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "verify_report.json").write_text(
        _json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8"
    )
    lines = ["# Q3 verify report", ""]
    for case in cases:
        entry = report["cases"].get(case)
        if entry is None:
            lines.append(f"- {case}: not in manifest")
            continue
        status = "PASS" if entry["verified"] else f"FAIL ({len(entry['violations'])})"
        lines.append(f"- {case}: {status}")
        for item in entry["violations"]:
            lines.append(f"    - {item}")
    lines.append(f"\nTotal violations: {len(violations)}")
    (report_dir / "verify_report.md").write_text("\n".join(lines), encoding="utf-8")
    return violations
