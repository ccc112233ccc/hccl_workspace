#!/usr/bin/env python3
"""Apply HCCL sendrecv completion semantics to ns3ub CTP task statistics.

ns3ub records CTP traffic as unidirectional tasks.  HCCL's mesh all-to-all
issues SendRecv on each channel, so the two opposite directions of the same
rank/channel pair should complete together.  This postprocessor keeps the
network simulation result but adjusts task start/end times to the pairwise
sendrecv view:

    pair start = max(start(a->b), start(b->a))
    pair end   = max(end(a->b), end(b->a))

Both directions then use the pair start/end for profiling and bandwidth
summaries.  Rows without a matching reverse row are left unchanged.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path


def field(row: dict[str, str], *names: str) -> str:
    for name in names:
        if name in row:
            return row[name]
    raise KeyError(names[0])


def canonical_key(row: dict[str, str]) -> tuple[int, int, str, str, str, str]:
    src = int(field(row, "sourceNode", "sourceNodeId"))
    dst = int(field(row, "destNode", "destNodeId"))
    lo, hi = sorted((src, dst))
    src_port = row.get("srcPortHint", "")
    dst_port = row.get("dstPortHint", "")
    port_key = "|".join(sorted((src_port, dst_port)))
    entity_key = "|".join(sorted((row.get("srcEntityId", ""), row.get("dstEntityId", ""))))
    return (
        lo,
        hi,
        row.get("priority", ""),
        row.get("opType", ""),
        entity_key,
        port_key,
    )


def throughput_gbps(size_bytes: int, start_us: float, end_us: float) -> float:
    duration_us = end_us - start_us
    if duration_us <= 0:
        return 0.0
    return size_bytes * 8 / duration_us / 1000


def apply_barrier(input_case: Path, output_case: Path) -> None:
    output_case.mkdir(parents=True, exist_ok=True)
    for name in ["node.csv", "topology.csv", "routing_table.csv", "transport_channel.csv", "traffic.csv", "network_attribute.txt"]:
        src = input_case / name
        if src.exists():
            shutil.copy2(src, output_case / name)

    in_stats = input_case / "output" / "task_statistics.csv"
    out_dir = output_case / "output"
    out_dir.mkdir(exist_ok=True)
    rows: list[dict[str, str]]
    with in_stats.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    groups: dict[tuple[int, int, str, str, str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[canonical_key(row)].append(idx)

    adjusted = 0
    unmatched = 0
    for indices in groups.values():
        by_dir: dict[tuple[int, int], list[int]] = defaultdict(list)
        for idx in indices:
            row = rows[idx]
            src = int(field(row, "sourceNode", "sourceNodeId"))
            dst = int(field(row, "destNode", "destNodeId"))
            by_dir[(src, dst)].append(idx)

        if len(by_dir) != 2:
            unmatched += len(indices)
            continue
        dirs = list(by_dir)
        if dirs[0] != (dirs[1][1], dirs[1][0]):
            unmatched += len(indices)
            continue

        count = min(len(by_dir[dirs[0]]), len(by_dir[dirs[1]]))
        if count == 0:
            continue
        for left_idx, right_idx in zip(sorted(by_dir[dirs[0]]), sorted(by_dir[dirs[1]]), strict=False):
            pair_start = max(float(rows[left_idx]["taskStartTime(us)"]), float(rows[right_idx]["taskStartTime(us)"]))
            pair_end = max(float(rows[left_idx]["taskCompletesTime(us)"]), float(rows[right_idx]["taskCompletesTime(us)"]))
            for idx in (left_idx, right_idx):
                rows[idx]["taskStartTime(us)"] = f"{pair_start:.6f}".rstrip("0").rstrip(".")
                rows[idx]["taskCompletesTime(us)"] = f"{pair_end:.6f}".rstrip("0").rstrip(".")
                rows[idx]["taskThroughput(Gbps)"] = f"{throughput_gbps(int(rows[idx]['dataSize(Byte)']), pair_start, pair_end):.4f}"
                adjusted += 1

    with (out_dir / "task_statistics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    throughput_src = input_case / "output" / "throughput.csv"
    if throughput_src.exists():
        shutil.copy2(throughput_src, out_dir / "throughput.csv")

    print(f"wrote {out_dir / 'task_statistics.csv'}")
    print(f"adjusted_rows={adjusted} unmatched_rows={unmatched}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-case", required=True, type=Path)
    parser.add_argument("--output-case", required=True, type=Path)
    args = parser.parse_args()
    apply_barrier(args.input_case, args.output_case)


if __name__ == "__main__":
    main()
