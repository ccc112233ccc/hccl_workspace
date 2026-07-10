#!/usr/bin/env python3
"""Render a single-rank HCCL V3-style task profiling timeline.

Rows are logical algorithm parallel units, not physical ports:
  - mesh-thread[i] for same-mesh peers, matching Mesh2DV3 neighborIdx.
  - clos-thread[i] for cross-mesh peers, matching MeshClosV3 linkIdx.

The bars use ns-3 task start/end times. If multiple tasks mapped to the same
logical unit overlap materially in ns-3, the row is split into visual sublanes
instead of inventing a serialization that is not present in task_statistics.csv.
"""

from __future__ import annotations

import argparse
import csv
import html
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


INVALID_GROUP_ID = -1


@dataclass
class Task:
    task_id: int
    src: int
    dst: int
    size: int
    start_us: float
    end_us: float
    throughput_gbps: float
    unit_kind: str
    unit_idx: int


@dataclass
class RenderStats:
    tx_bytes: int
    task_count: int
    synthetic_self_bytes: int


def pairwise_round_num(group_num: int) -> int:
    if group_num <= 1:
        return 0
    return group_num - 1 if group_num % 2 == 0 else group_num


def pair_group_in_round(group_num: int, my_group: int, round_id: int) -> tuple[int, bool]:
    if group_num <= 1 or my_group >= group_num:
        return INVALID_GROUP_ID, True

    schedule_group_num = group_num if group_num % 2 == 0 else group_num + 1
    round_num = schedule_group_num - 1
    dummy_group = group_num
    groups = list(range(schedule_group_num))

    for _ in range(round_id % round_num):
        last = groups[-1]
        for idx in range(schedule_group_num - 1, 1, -1):
            groups[idx] = groups[idx - 1]
        groups[1] = last

    for idx in range(schedule_group_num // 2):
        left = groups[idx]
        right = groups[schedule_group_num - 1 - idx]
        if left == my_group:
            return (INVALID_GROUP_ID if right == dummy_group else right), True
        if right == my_group:
            return (INVALID_GROUP_ID if left == dummy_group else left), False
    return INVALID_GROUP_ID, True


def infer_inter_channel_count(source_case: Path, rank_count: int, group_size: int, priority: int) -> int:
    counts: dict[tuple[int, int], int] = defaultdict(int)
    with (source_case / "transport_channel.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            if int(row["priority"]) != priority:
                continue
            a = int(row["nodeId1"])
            b = int(row["nodeId2"])
            if a >= rank_count or b >= rank_count:
                continue
            if a // group_size == b // group_size:
                continue
            counts[tuple(sorted((a, b)))] += 1
    if not counts:
        raise ValueError(f"could not infer inter channel count from {source_case}")
    return max(counts.values())


def logical_unit(src: int, dst: int, rank_count: int, group_size: int, clos_channels: int) -> tuple[str, int]:
    src_group, src_local = divmod(src, group_size)
    dst_group, dst_local = divmod(dst, group_size)
    if src_group == dst_group:
        ordered_peers = [
            src_group * group_size + ((src_local + 1 + idx) % group_size)
            for idx in range(group_size - 1)
        ]
        return "mesh", ordered_peers.index(dst)

    return "clos", (src ^ dst) % clos_channels


def hinted_logical_unit(
    row: dict[str, str],
    src: int,
    dst: int,
    group_size: int,
    clos_channels: int,
) -> tuple[str, int] | None:
    src_port_hint = row.get("srcPortHint", "").strip()
    if src_port_hint == "":
        return None
    try:
        hint = int(src_port_hint)
    except ValueError:
        return None

    if src // group_size == dst // group_size:
        return "mesh", hint % group_size

    first_clos_hint = group_size - 1
    return "clos", (hint - first_clos_hint) % clos_channels


def load_task_metadata(case_dir: Path) -> dict[int, dict[str, str]]:
    traffic_path = case_dir / "traffic.csv"
    if not traffic_path.exists():
        return {}

    with traffic_path.open(newline="") as f:
        return {int(row["taskId"]): row for row in csv.DictReader(f)}


def load_rank_tasks(
    case_dir: Path,
    rank: int,
    rank_count: int,
    group_size: int,
    clos_channels: int,
    priorities: set[int],
    unit_mode: str,
) -> list[Task]:
    def field(row: dict[str, str], *names: str) -> str:
        for name in names:
            if name in row:
                return row[name]
        raise KeyError(names[0])

    task_metadata = load_task_metadata(case_dir)
    tasks: list[Task] = []
    with (case_dir / "output" / "task_statistics.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            row = {**task_metadata.get(int(row["taskId"]), {}), **row}
            if int(row["priority"]) not in priorities:
                continue
            src = int(field(row, "sourceNodeId", "sourceNode"))
            if src != rank:
                continue
            dst = int(field(row, "destNodeId", "destNode"))
            priority = int(row["priority"])
            if unit_mode == "peer-thread":
                unit_kind, unit_idx = "peer", dst
            elif (
                unit_mode == "priority-plane"
                and src // group_size != dst // group_size
                and priority != 7
            ):
                unit_kind, unit_idx = "clos", priority - 3
            else:
                hinted = hinted_logical_unit(row, src, dst, group_size, clos_channels)
                if hinted is not None:
                    unit_kind, unit_idx = hinted
                else:
                    # Baseline traffic has no fixed-plane port hint. In that
                    # case each task is its own peer thread; mapping it into
                    # clos buckets would incorrectly merge independent peers.
                    unit_kind, unit_idx = "peer", dst
            tasks.append(
                Task(
                    task_id=int(row["taskId"]),
                    src=src,
                    dst=dst,
                    size=int(row["dataSize(Byte)"]),
                    start_us=float(row["taskStartTime(us)"]),
                    end_us=float(row["taskCompletesTime(us)"]),
                    throughput_gbps=float(row["taskThroughput(Gbps)"]),
                    unit_kind=unit_kind,
                    unit_idx=unit_idx,
                )
            )
    return sorted(tasks, key=lambda t: (t.unit_kind, t.unit_idx, t.task_id))


def infer_render_stats(tasks: list[Task], rank_count: int, include_self: bool) -> RenderStats:
    tx_bytes = sum(task.size for task in tasks)
    synthetic_self_bytes = 0
    if include_self and tasks:
        # AllToAll traffic omits the self-send task but bandwidth is normally
        # reported per rank including that rank's own shard.
        synthetic_self_bytes = max(task.size for task in tasks)
        tx_bytes += synthetic_self_bytes
    task_count = len(tasks) + (1 if synthetic_self_bytes else 0)
    return RenderStats(tx_bytes=tx_bytes, task_count=task_count, synthetic_self_bytes=synthetic_self_bytes)


def pack_sublanes(tasks: list[Task], overlap_epsilon_us: float) -> list[tuple[Task, int]]:
    ends: list[float] = []
    packed: list[tuple[Task, int]] = []
    for task in sorted(tasks, key=lambda t: (t.start_us, t.end_us, t.task_id)):
        sublane = None
        for idx, end_us in enumerate(ends):
            if task.start_us + overlap_epsilon_us >= end_us:
                sublane = idx
                break
        if sublane is None:
            sublane = len(ends)
            ends.append(task.end_us)
        else:
            ends[sublane] = task.end_us
        packed.append((task, sublane))
    return packed


def render_html(
    case_dir: Path,
    tasks: list[Task],
    rank: int,
    rank_count: int,
    group_size: int,
    clos_channels: int,
    title: str,
    sublane_epsilon_us: float,
    markers: list[tuple[float, str]],
    include_self: bool,
) -> str:
    max_end = max((task.end_us for task in tasks), default=1.0)
    stats = infer_render_stats(tasks, rank_count, include_self)
    direct_gbs = stats.tx_bytes / max_end / 1e3 if max_end else 0

    lanes: dict[tuple[str, int], list[Task]] = defaultdict(list)
    for task in tasks:
        lanes[(task.unit_kind, task.unit_idx)].append(task)

    if any(kind == "peer" for kind, _ in lanes):
        lane_order = [("peer", idx) for idx in range(rank_count)]
    else:
        lane_order = [("mesh", idx) for idx in range(group_size)]
        lane_order.extend(("clos", idx) for idx in range(clos_channels))

    left = 172
    right = 28
    top = 84
    row_gap = 8
    sublane_h = 15
    min_row_h = 24
    width = 1180
    plot_w = width - left - right
    x_scale = plot_w / max_end if max_end else 1

    row_entries = []
    y = top
    for lane in lane_order:
        packed = pack_sublanes(lanes[lane], sublane_epsilon_us)
        sublane_count = max((s for _, s in packed), default=0) + 1
        row_h = max(min_row_h, sublane_count * sublane_h + 9)
        row_entries.append((lane, packed, y, row_h))
        y += row_h + row_gap
    height = y + 52

    ticks = []
    for idx in range(9):
        t = max_end * idx / 8
        x = left + t * x_scale
        ticks.append(
            f'<line class="grid" x1="{x:.2f}" y1="{top - 28}" x2="{x:.2f}" y2="{height - 44}" />'
            f'<text class="tick" x="{x:.2f}" y="{top - 36}" text-anchor="middle">{t:.1f}us</text>'
        )

    marker_rows = []
    for marker_us, marker_label in markers:
        if marker_us < 0 or marker_us > max_end:
            continue
        x = left + marker_us * x_scale
        marker_rows.append(
            f'<line class="iter-marker" x1="{x:.2f}" y1="{top - 30}" '
            f'x2="{x:.2f}" y2="{height - 44}" />'
            f'<text class="iter-marker-label" x="{x + 6:.2f}" y="{top - 14}">'
            f'{html.escape(marker_label)} {marker_us:.1f}us</text>'
        )

    rows = []
    for lane, packed, row_y, row_h in row_entries:
        kind, idx = lane
        label = f"{kind}-thread[{idx}]"
        lane_tasks = lanes[lane]
        rows.append(
            f'<rect class="row-bg" x="0" y="{row_y - 4:.2f}" width="{width}" height="{row_h:.2f}" />'
            f'<text class="row-label" x="16" y="{row_y + 12:.2f}">{html.escape(label)}</text>'
            f'<text class="row-meta" x="124" y="{row_y + 12:.2f}">{len(lane_tasks)} task</text>'
        )
        by_sublane: dict[int, list[Task]] = defaultdict(list)
        for task, sublane in packed:
            by_sublane[sublane].append(task)
        for sublane, sublane_tasks in by_sublane.items():
            ordered = sorted(sublane_tasks, key=lambda t: (t.start_us, t.end_us, t.task_id))
            for prev, cur in zip(ordered, ordered[1:]):
                gap_us = cur.start_us - prev.end_us
                if gap_us <= 1e-6:
                    continue
                x = left + prev.end_us * x_scale
                w = max(1.0, gap_us * x_scale)
                bar_y = row_y + 3 + sublane * sublane_h
                tooltip = (
                    f"bubble: {gap_us:.3f}us\\n"
                    f"after task {prev.task_id}, before task {cur.task_id}\\n"
                    f"{prev.end_us:.3f}us - {cur.start_us:.3f}us"
                )
                rows.append(
                    f'<g class="bubble" tabindex="0" data-kind="bubble" '
                    f'data-gap="{gap_us:.6f} us" data-start="{prev.end_us:.6f} us" '
                    f'data-end="{cur.start_us:.6f} us" data-slot="{html.escape(label)}" '
                    f'data-prev="task {prev.task_id}" data-next="task {cur.task_id}">'
                    f'<title>{html.escape(tooltip)}</title>'
                    f'<rect class="gap" x="{x:.2f}" y="{bar_y:.2f}" width="{w:.2f}" height="11" rx="2" />'
                    f'</g>'
                )
        for task, sublane in packed:
            x = left + task.start_us * x_scale
            w = max(1.0, (task.end_us - task.start_us) * x_scale)
            bar_y = row_y + 3 + sublane * sublane_h
            cls = "bar mesh" if task.unit_kind == "mesh" else "bar clos"
            label_x = x + 4
            slot_label = f"{task.unit_kind}-thread[{task.unit_idx}]"
            duration_us = task.end_us - task.start_us
            gbps = task.throughput_gbps / 8
            tooltip = (
                f"task {task.task_id}: {task.src}->{task.dst}\\n"
                f"{task.start_us:.3f}us - {task.end_us:.3f}us\\n"
                f"{task.size} bytes\\n"
                f"{gbps:.2f} GB/s\\n"
                f"{slot_label}"
            )
            rows.append(
                f'<g class="task" tabindex="0" data-task-id="{task.task_id}" '
                f'data-flow="{task.src}->{task.dst}" data-slot="{html.escape(slot_label)}" '
                f'data-start="{task.start_us:.6f} us" data-end="{task.end_us:.6f} us" '
                f'data-duration="{duration_us:.6f} us" data-size="{task.size} B" '
                f'data-throughput="{task.throughput_gbps:.3f} Gbps / {gbps:.3f} GB/s">'
                f'<title>{html.escape(tooltip)}</title>'
                f'<rect class="{cls}" x="{x:.2f}" y="{bar_y:.2f}" width="{w:.2f}" height="11" rx="2" />'
                f'<text class="bar-label" x="{label_x:.2f}" y="{bar_y + 8.5:.2f}">{task.dst}</text>'
                f'</g>'
            )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin: 0; background: #f6f7f9; color: #1f2937; font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1240px, calc(100% - 32px)); margin: 24px auto 44px; }}
    h1 {{ margin: 0 0 6px; font-size: 24px; letter-spacing: 0; }}
    .muted {{ color: #667085; margin: 0 0 14px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-bottom: 14px; }}
    .metric {{ background: #fff; border: 1px solid #d8dee8; border-radius: 8px; padding: 12px; }}
    .k {{ color: #667085; font-size: 12px; }}
    .v {{ font-size: 22px; font-weight: 700; }}
    .panel {{ overflow-x: auto; background: #fff; border: 1px solid #d8dee8; border-radius: 8px; }}
    svg {{ display: block; min-width: {width}px; }}
    .grid {{ stroke: #e3e8f0; stroke-width: 1; }}
    .tick, .row-meta {{ fill: #667085; font-size: 11px; }}
    .iter-marker {{ stroke: #111827; stroke-width: 3; opacity: .82; }}
    .iter-marker-label {{ fill: #111827; font-size: 12px; font-weight: 700; }}
    .row-bg {{ fill: #fbfcfe; stroke: #edf1f6; }}
    .row-label {{ fill: #111827; font-size: 12px; font-weight: 650; }}
    .bar {{ stroke: rgba(17,24,39,.24); stroke-width: .5; opacity: .86; }}
    .task, .bubble {{ cursor: pointer; outline: none; }}
    .task:focus .bar, .task.selected .bar {{ stroke: #111827; stroke-width: 1.8; opacity: 1; }}
    .bubble:focus .gap, .bubble.selected .gap {{ stroke: #111827; stroke-width: 1.8; opacity: 1; }}
    .bar.mesh {{ fill: #db2777; }}
    .bar.clos {{ fill: #db2777; }}
    .gap {{ fill: #22c55e; stroke: rgba(17,24,39,.22); stroke-width: .5; opacity: .56; }}
    .bar-label {{ fill: #fff; font-size: 9px; pointer-events: none; }}
    .legend {{ display: flex; gap: 16px; color: #475467; font-size: 12px; margin-top: 10px; }}
    .swatch {{ display: inline-block; width: 12px; height: 8px; border-radius: 2px; margin-right: 5px; }}
    .details {{ margin-top: 12px; background: #fff; border: 1px solid #d8dee8; border-radius: 8px; padding: 14px; }}
    .details h2 {{ margin: 0 0 10px; font-size: 16px; }}
    .detail-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }}
    .detail-item {{ background: #fbfcfe; border: 1px solid #edf1f6; border-radius: 6px; padding: 8px 10px; }}
    .detail-label {{ color: #667085; font-size: 12px; }}
    .detail-value {{ color: #111827; font-weight: 650; font-variant-numeric: tabular-nums; }}
    @media (max-width: 760px) {{ .metrics, .detail-grid {{ grid-template-columns: 1fr 1fr; }} }}
  </style>
</head>
<body>
<main>
  <h1>{html.escape(title)}</h1>
  <p class="muted">单 rank TX 视角；行是 V3 源码逻辑并行单元，条形是 ns-3 task 的实际 start/end。条形上的数字是目的 rank。</p>
  <div class="metrics">
    <div class="metric"><div class="k">Case</div><div class="v">{html.escape(case_dir.name)}</div></div>
    <div class="metric"><div class="k">Rank</div><div class="v">{rank}</div></div>
    <div class="metric"><div class="k">Tasks</div><div class="v">{stats.task_count}</div></div>
    <div class="metric"><div class="k">TX GB/s</div><div class="v">{direct_gbs:.2f}</div></div>
  </div>
  <div class="panel">
    <svg width="{width}" height="{height}" role="img" aria-label="{html.escape(title)}">
      {''.join(ticks)}
      {''.join(marker_rows)}
      {''.join(rows)}
    </svg>
  </div>
  <div class="legend">
    <span><span class="swatch" style="background:#db2777"></span>真实数据流 task</span>
    <span><span class="swatch" style="background:#22c55e"></span>同一逻辑行上的空泡</span>
  </div>
  <section class="details" aria-live="polite">
    <h2>Task detail</h2>
    <div class="detail-grid" id="detail-grid">
      <div class="detail-item"><div class="detail-label">选择</div><div class="detail-value">点击上方任意条形流</div></div>
    </div>
  </section>
</main>
<script>
  const detailGrid = document.getElementById('detail-grid');
  const fields = [
    ['taskId', 'Task'],
    ['flow', 'Flow'],
    ['slot', 'Slot'],
    ['start', 'Start'],
    ['end', 'End'],
    ['duration', 'Duration'],
    ['size', 'Data'],
    ['throughput', 'Throughput'],
  ];
  function selectTask(node) {{
    document.querySelectorAll('.task.selected, .bubble.selected').forEach((el) => el.classList.remove('selected'));
    node.classList.add('selected');
    if (node.dataset.kind === 'bubble') {{
      detailGrid.innerHTML = [
        ['slot', 'Slot'],
        ['start', 'Start'],
        ['end', 'End'],
        ['gap', 'Bubble'],
        ['prev', 'After'],
        ['next', 'Before'],
      ].map(([key, label]) => {{
        const value = node.dataset[key] || '-';
        return `<div class="detail-item"><div class="detail-label">${{label}}</div><div class="detail-value">${{value}}</div></div>`;
      }}).join('');
      return;
    }}
    detailGrid.innerHTML = fields.map(([key, label]) => {{
      const value = node.dataset[key] || '-';
      return `<div class="detail-item"><div class="detail-label">${{label}}</div><div class="detail-value">${{value}}</div></div>`;
    }}).join('');
  }}
  document.querySelectorAll('.task, .bubble').forEach((node) => {{
    node.addEventListener('click', () => selectTask(node));
    node.addEventListener('keydown', (event) => {{
      if (event.key === 'Enter' || event.key === ' ') {{
        event.preventDefault();
        selectTask(node);
      }}
    }});
  }});
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--rank-count", type=int, required=True)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument(
        "--source-case",
        type=Path,
        help="Optional full-topology source case used to infer clos channel count.",
    )
    parser.add_argument("--clos-channels", type=int)
    parser.add_argument("--priority", type=int, default=7, help="Single priority to render. Kept for compatibility.")
    parser.add_argument(
        "--priorities",
        type=int,
        nargs="+",
        help="Render tasks whose priority is in this set. Overrides --priority.",
    )
    parser.add_argument(
        "--unit-mode",
        choices=("logical", "priority-plane", "peer-thread"),
        default="logical",
        help=(
            "logical uses MeshClosV3 logical thread mapping; priority-plane maps cross-rank "
            "priority 3/4/5/6 to clos-thread[0..3]; peer-thread renders one row per destination rank."
        ),
    )
    parser.add_argument(
        "--sublane-epsilon-us",
        type=float,
        default=5.0,
        help="Do not split a row into visual sublanes for overlaps up to this duration.",
    )
    parser.add_argument(
        "--marker-us",
        type=float,
        action="append",
        default=[],
        help="Draw a thick vertical marker at this timestamp. Can be repeated.",
    )
    parser.add_argument(
        "--marker-label",
        default="iter1 start",
        help="Label used for --marker-us vertical markers.",
    )
    parser.add_argument("--title")
    parser.add_argument(
        "--exclude-self",
        action="store_true",
        help="Do not add the omitted self-send shard to TX bytes/task count.",
    )
    args = parser.parse_args()

    case_dir = args.case_dir.resolve()
    clos_channels = args.clos_channels
    if clos_channels is None:
        if args.source_case is not None:
            clos_channels = infer_inter_channel_count(
                args.source_case.resolve(),
                args.rank_count,
                args.group_size,
                args.priority,
            )
        else:
            clos_channels = args.group_size

    priorities = set(args.priorities) if args.priorities else {args.priority}
    tasks = load_rank_tasks(
        case_dir,
        args.rank,
        args.rank_count,
        args.group_size,
        clos_channels,
        priorities,
        args.unit_mode,
    )
    if not tasks:
        raise ValueError(f"no TX tasks for rank {args.rank} in {case_dir}")

    title = args.title or f"Rank {args.rank} V3 Logical Timeline"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_html(
            case_dir,
            tasks,
            args.rank,
            args.rank_count,
            args.group_size,
            clos_channels,
            title,
            args.sublane_epsilon_us,
            [(marker, args.marker_label) for marker in args.marker_us],
            not args.exclude_self,
        )
    )
    print(f"wrote {args.output}")
    print(f"rank={args.rank} tasks={len(tasks)} clos_channels={clos_channels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
