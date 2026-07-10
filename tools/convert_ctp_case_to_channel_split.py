#!/usr/bin/env python3
"""Convert a CTP all-to-all case from per-peer tasks to per-channel tasks.

The HCCL CTP path splits one peer's payload across multiple channels before
issuing send/recv work. This utility mirrors that in ns3ub traffic.csv by
splitting each task across transport_channel.csv rows and pinning each child
task with srcPortHint/dstPortHint.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path


TRAFFIC_FIELDS = [
    "taskId",
    "sourceNode",
    "destNode",
    "dataSize(Byte)",
    "opType",
    "priority",
    "delay",
    "phaseId",
    "dependOnPhases",
    "srcEntityId",
    "dstEntityId",
    "srcPortHint",
    "dstPortHint",
]


def row_get(row: dict[str, str], *names: str) -> str:
    for name in names:
        if name in row:
            return row[name]
    raise KeyError(names[0])


def load_channels(path: Path) -> dict[tuple[int, int, int], list[tuple[int, int, int]]]:
    channels: dict[tuple[int, int, int], list[tuple[int, int, int]]] = defaultdict(list)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            a = int(row["nodeId1"])
            b = int(row["nodeId2"])
            pa = int(row["portId1"])
            pb = int(row["portId2"])
            priority = int(row["priority"])
            metric = int(row.get("metric") or 1)
            channels[(a, b, priority)].append((pa, pb, metric))
            channels[(b, a, priority)].append((pb, pa, metric))
    for value in channels.values():
        value.sort()
    return channels


def split_sizes(total: int, weights: list[int]) -> list[int]:
    weight_sum = sum(weights)
    if weight_sum <= 0:
        weights = [1 for _ in weights]
        weight_sum = len(weights)

    sizes: list[int] = []
    assigned = 0
    for idx, weight in enumerate(weights):
        if idx == len(weights) - 1:
            size = total - assigned
        else:
            size = total * weight // weight_sum
            assigned += size
        sizes.append(size)
    return sizes


def convert_network_attribute(text: str, packet_spray: bool) -> str:
    desired = "true" if packet_spray else "false"
    lines: list[str] = []
    saw_transport_mode = False
    saw_keys: set[str] = set()
    spray_keys = {
        "default ns3::UbApp::UsePacketSpray",
        "default ns3::UbTransportChannel::UsePacketSpray",
        "default ns3::UbLdstApi::UsePacketSpray",
    }

    for raw_line in text.splitlines():
        line = raw_line
        if line.startswith("default ns3::UbApp::TransportMode"):
            line = 'default ns3::UbApp::TransportMode "CTP"'
            saw_transport_mode = True
        for key in spray_keys:
            if line.startswith(key):
                line = f'{key} "{desired}"'
                saw_keys.add(key)
                break
        lines.append(line)

    if not saw_transport_mode:
        lines.insert(0, 'default ns3::UbApp::TransportMode "CTP"')
    for key in sorted(spray_keys - saw_keys):
        lines.append(f'{key} "{desired}"')
    return "\n".join(lines) + "\n"


def copy_case_files(source_case: Path, output_case: Path) -> None:
    output_case.mkdir(parents=True, exist_ok=True)
    for name in ["node.csv", "topology.csv", "routing_table.csv"]:
        shutil.copy2(source_case / name, output_case / name)


def convert_case(
    source_case: Path,
    channel_case: Path,
    output_case: Path,
    packet_spray: bool,
    max_channels_per_peer: int | None,
    entity_per_channel: bool,
    bytes_per_peer: int | None,
) -> None:
    copy_case_files(source_case, output_case)
    channels = load_channels(channel_case / "transport_channel.csv")
    shutil.copy2(channel_case / "transport_channel.csv", output_case / "transport_channel.csv")

    network_text = (source_case / "network_attribute.txt").read_text()
    (output_case / "network_attribute.txt").write_text(convert_network_attribute(network_text, packet_spray))

    out_rows: list[dict[str, str]] = []
    next_task_id = 0
    with (source_case / "traffic.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            src = int(row_get(row, "sourceNode", "sourceNodeId"))
            dst = int(row_get(row, "destNode", "destNodeId"))
            priority = int(row["priority"])
            size = bytes_per_peer if bytes_per_peer is not None else int(row["dataSize(Byte)"])
            candidates = channels.get((src, dst, priority))
            if not candidates:
                raise ValueError(f"no channels for {src}->{dst} priority {priority}")
            if max_channels_per_peer is not None:
                candidates = candidates[:max_channels_per_peer]

            # The current HCCL all-to-all transport_channel rows represent one
            # physical port per row. If metrics differ, use them as a stable
            # inverse cost only for ordering; split evenly like portGroupSize=1.
            sizes = split_sizes(size, [1 for _ in candidates])
            for channel_idx, ((src_port, dst_port, _metric), child_size) in enumerate(
                zip(candidates, sizes, strict=True)
            ):
                if child_size <= 0:
                    continue
                out_rows.append(
                    {
                        "taskId": str(next_task_id),
                        "sourceNode": str(src),
                        "destNode": str(dst),
                        "dataSize(Byte)": str(child_size),
                        "opType": row["opType"],
                        "priority": row["priority"],
                        "delay": row["delay"],
                        "phaseId": row["phaseId"],
                        "dependOnPhases": row.get("dependOnPhases", ""),
                        "srcEntityId": str(channel_idx) if entity_per_channel else (row.get("srcEntityId", "0") or "0"),
                        "dstEntityId": str(channel_idx) if entity_per_channel else (row.get("dstEntityId", "0") or "0"),
                        "srcPortHint": str(src_port),
                        "dstPortHint": str(dst_port),
                    }
                )
                next_task_id += 1

    with (output_case / "traffic.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRAFFIC_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-case", required=True, type=Path)
    parser.add_argument("--channel-case", required=True, type=Path)
    parser.add_argument("--output-case", required=True, type=Path)
    parser.add_argument("--packet-spray", action="store_true")
    parser.add_argument("--max-channels-per-peer", type=int)
    parser.add_argument("--entity-per-channel", action="store_true")
    parser.add_argument("--bytes-per-peer", type=int)
    args = parser.parse_args()
    convert_case(
        args.source_case,
        args.channel_case,
        args.output_case,
        args.packet_spray,
        args.max_channels_per_peer,
        args.entity_per_channel,
        args.bytes_per_peer,
    )


if __name__ == "__main__":
    main()
