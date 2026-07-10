#!/usr/bin/env python3
"""Compress ns-3-ub routing_table.csv rows with nodeId/dstNodeId ranges."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path


FIELDNAMES = ["nodeId", "dstNodeId", "dstPortId", "outPorts", "metrics"]


def parse_range(value: object) -> tuple[int, int]:
    text = str(value).strip()
    if ".." not in text:
        point = int(text)
        return point, point
    start_text, end_text = text.split("..", 1)
    start = int(start_text)
    end = int(end_text)
    if start > end:
        raise ValueError(f"range start must not exceed end: {text}")
    return start, end


def format_range(start: int, end: int) -> str:
    return str(start) if start == end else f"{start}..{end}"


def consecutive_ranges(values: Iterable[int]) -> list[tuple[int, int]]:
    sorted_values = sorted(set(values))
    if not sorted_values:
        return []
    ranges: list[tuple[int, int]] = []
    start = prev = sorted_values[0]
    for value in sorted_values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append((start, prev))
        start = prev = value
    ranges.append((start, prev))
    return ranges


def normalize_value(out_ports: object, metrics: object) -> tuple[str, str]:
    port_metric: dict[int, int] = {}
    ports = [int(part) for part in str(out_ports).split()]
    metric_values = [int(part) for part in str(metrics).split()]
    if len(ports) != len(metric_values):
        raise ValueError(f"outPorts and metrics length mismatch: {out_ports!r}, {metrics!r}")
    if not ports:
        raise ValueError("route row must contain at least one outPort/metric pair")
    for port, metric in zip(ports, metric_values, strict=True):
        previous = port_metric.get(port)
        if previous is not None and previous != metric:
            raise ValueError(f"outPort {port} has conflicting metrics {previous} and {metric}")
        port_metric[port] = metric
    pairs = sorted(port_metric.items())
    return " ".join(str(port) for port, _ in pairs), " ".join(str(metric) for _, metric in pairs)


def iter_csv_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        missing = [field for field in FIELDNAMES if field not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} missing columns: {', '.join(missing)}")
        yield from reader


def compress_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    by_node_and_value: dict[tuple[int, str, str, str], list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        node_start, node_end = parse_range(row["nodeId"])
        dst_start, dst_end = parse_range(row["dstNodeId"])
        out_ports, metrics = normalize_value(row["outPorts"], row["metrics"])
        dst_port = str(int(row["dstPortId"]))
        for node_id in range(node_start, node_end + 1):
            by_node_and_value[(node_id, dst_port, out_ports, metrics)].append((dst_start, dst_end))

    dst_range_rows: list[tuple[int, str, str, str, str]] = []
    for (node_id, dst_port, out_ports, metrics), dst_ranges in by_node_and_value.items():
        dst_ids = (dst for start, end in dst_ranges for dst in range(start, end + 1))
        for dst_start, dst_end in consecutive_ranges(dst_ids):
            dst_range_rows.append((node_id, format_range(dst_start, dst_end), dst_port, out_ports, metrics))

    by_signature: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    for node_id, dst_range, dst_port, out_ports, metrics in dst_range_rows:
        by_signature[(dst_range, dst_port, out_ports, metrics)].append(node_id)

    compressed: list[dict[str, object]] = []
    for (dst_range, dst_port, out_ports, metrics), node_ids in by_signature.items():
        for node_start, node_end in consecutive_ranges(node_ids):
            compressed.append(
                {
                    "nodeId": format_range(node_start, node_end),
                    "dstNodeId": dst_range,
                    "dstPortId": dst_port,
                    "outPorts": out_ports,
                    "metrics": metrics,
                }
            )

    return sorted(
        compressed,
        key=lambda row: (
            parse_range(row["nodeId"])[0],
            parse_range(row["dstNodeId"])[0],
            int(row["dstPortId"]),
            str(row["outPorts"]),
            str(row["metrics"]),
        ),
    )


def write_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def compress_file(input_path: Path, output_path: Path | None = None) -> tuple[int, int]:
    output_path = output_path or input_path
    input_count = 0

    def counted_rows() -> Iterable[dict[str, str]]:
        nonlocal input_count
        for row in iter_csv_rows(input_path):
            input_count += 1
            yield row

    compressed = compress_rows(counted_rows())
    write_rows(output_path, compressed)
    return input_count, len(compressed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("routing_table", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    before, after = compress_file(args.routing_table, args.output)
    target = args.output or args.routing_table
    print(f"compressed {args.routing_table} -> {target}: {before} rows -> {after} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
