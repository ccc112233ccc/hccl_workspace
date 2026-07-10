#!/usr/bin/env python3
"""Generate 8-plane POD topologies used by the alltoall experiments."""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NETWORK_ATTR = REPO_ROOT / "experiments/topologies/pod128/generated_topology/network_attribute.txt"

GROUP_SIZE = 8
PLANE_COUNT = 8
L1_TO_L2_PORTS = (8, 9, 10, 11)
L1_TO_TOP_PORTS = (12, 13, 14, 15)
TOP_SWITCH_COUNT = 32


def direct_port(local_src: int, local_dst: int) -> int:
    if local_src == local_dst:
        raise ValueError("self direct port is undefined")
    compact_peer = local_dst if local_dst < local_src else local_dst - 1
    return 8 + compact_peer


def endpoint_tpns(rank_count: int) -> tuple[dict[tuple[int, int], list[int]], int]:
    """Return priority-1 endpoint TPNs and the priority-7 offset."""

    tpn: dict[tuple[int, int], list[int]] = {}
    max_count = 0
    for src in range(rank_count):
        cur = 0
        src_group = src // GROUP_SIZE
        for dst in range(rank_count):
            if src == dst:
                continue
            dst_group = dst // GROUP_SIZE
            if dst_group == src_group:
                tpn[(src, dst)] = [cur]
                cur += 1
            else:
                tpn[(src, dst)] = list(range(cur, cur + PLANE_COUNT))
                cur += PLANE_COUNT
        max_count = max(max_count, cur)
    return tpn, max_count


class PodLayout:
    def __init__(self, rank_count: int) -> None:
        if rank_count % GROUP_SIZE != 0:
            raise ValueError("rank_count must be a multiple of 8")
        group_count = rank_count // GROUP_SIZE
        if group_count % GROUP_SIZE != 0:
            raise ValueError("group count must be a multiple of 8")
        if group_count > 128:
            raise ValueError("5808 switches in this model expose 128 ports")

        self.rank_count = rank_count
        self.group_count = group_count
        self.l1_count = PLANE_COUNT * group_count
        self.l2_blocks_per_plane = group_count // GROUP_SIZE
        self.l2_count = PLANE_COUNT * self.l2_blocks_per_plane * 2
        self.top_count = TOP_SWITCH_COUNT

        self.l1_base = rank_count
        self.l2_base = self.l1_base + self.l1_count
        self.top_base = self.l2_base + self.l2_count
        self.total_nodes = self.top_base + self.top_count

    def l1(self, plane: int, group: int) -> int:
        return self.l1_base + plane * self.group_count + group

    def l2(self, plane: int, group: int, half: int) -> int:
        block = group // GROUP_SIZE
        return self.l2_base + plane * self.l2_blocks_per_plane * 2 + block * 2 + half

    def top(self, plane: int, top_lane: int) -> int:
        return self.top_base + plane * 4 + top_lane


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def format_range(start: int, end: int) -> str:
    return str(start) if start == end else f"{start}..{end}"


def consecutive_ranges(values: list[int]) -> list[tuple[int, int]]:
    if not values:
        return []
    ranges: list[tuple[int, int]] = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append((start, prev))
        start = prev = value
    ranges.append((start, prev))
    return ranges


def compress_routing(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Compress exact route rows into range rows supported by ns-3-ub.

    The compression is semantic-only: rows with identical dstPortId, outPorts,
    and metrics are coalesced across consecutive dstNodeId values, then across
    consecutive nodeId values. Expanding the result produces the same route set.
    """

    by_node_and_value: dict[tuple[int, str, str, str], list[int]] = defaultdict(list)
    for row in rows:
        by_node_and_value[
            (
                int(row["nodeId"]),
                str(row["dstPortId"]),
                str(row["outPorts"]),
                str(row["metrics"]),
            )
        ].append(int(row["dstNodeId"]))

    dst_range_rows: list[tuple[int, str, str, str, str]] = []
    for (node_id, dst_port, out_ports, metrics), dst_ids in by_node_and_value.items():
        for dst_start, dst_end in consecutive_ranges(sorted(set(dst_ids))):
            dst_range_rows.append((node_id, format_range(dst_start, dst_end), dst_port, out_ports, metrics))

    by_signature: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    for node_id, dst_range, dst_port, out_ports, metrics in dst_range_rows:
        by_signature[(dst_range, dst_port, out_ports, metrics)].append(node_id)

    compressed: list[dict[str, object]] = []
    for (dst_range, dst_port, out_ports, metrics), node_ids in by_signature.items():
        for node_start, node_end in consecutive_ranges(sorted(set(node_ids))):
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
            int(str(row["nodeId"]).split("..", 1)[0]),
            int(str(row["dstNodeId"]).split("..", 1)[0]),
            int(row["dstPortId"]),
            str(row["outPorts"]),
            str(row["metrics"]),
        ),
    )


def generate_topology(layout: PodLayout) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add(a: int, pa: int, b: int, pb: int) -> None:
        rows.append(
            {
                "nodeId1": a,
                "portId1": pa,
                "nodeId2": b,
                "portId2": pb,
                "bandwidth": "400Gbps",
                "delay": "1ns",
            }
        )

    # Host to one L1 per plane.
    for host in range(layout.rank_count):
        group = host // GROUP_SIZE
        local = host % GROUP_SIZE
        for plane in range(PLANE_COUNT):
            add(host, plane, layout.l1(plane, group), local)

    # Direct fullmesh links inside each 8-rank group.
    for group in range(layout.group_count):
        base = group * GROUP_SIZE
        for i in range(GROUP_SIZE):
            for j in range(i + 1, GROUP_SIZE):
                add(base + i, direct_port(i, j), base + j, direct_port(j, i))

    # L1 to L2 and shared top-level 5808 switches.
    for plane in range(PLANE_COUNT):
        for group in range(layout.group_count):
            l1 = layout.l1(plane, group)
            local_group = group % GROUP_SIZE
            for half in (0, 1):
                l2 = layout.l2(plane, group, half)
                for lane in (0, 1):
                    l1_port = L1_TO_L2_PORTS[half * 2 + lane]
                    l2_port = local_group * 2 + lane
                    add(l1, l1_port, l2, l2_port)
            for top_lane in range(4):
                add(l1, L1_TO_TOP_PORTS[top_lane], layout.top(plane, top_lane), group)

    return rows


def generate_transport(layout: PodLayout) -> list[dict[str, object]]:
    tpn, ack_offset = endpoint_tpns(layout.rank_count)
    rows: list[dict[str, object]] = []

    def add(a: int, pa: int, ta: int, b: int, pb: int, tb: int, priority: int, metric: int) -> None:
        rows.append(
            {
                "nodeId1": a,
                "portId1": pa,
                "tpn1": ta,
                "nodeId2": b,
                "portId2": pb,
                "tpn2": tb,
                "priority": priority,
                "metric": metric,
            }
        )

    for src in range(layout.rank_count):
        src_group = src // GROUP_SIZE
        src_local = src % GROUP_SIZE
        for dst in range(src + 1, layout.rank_count):
            dst_group = dst // GROUP_SIZE
            dst_local = dst % GROUP_SIZE
            if src_group == dst_group:
                ps = direct_port(src_local, dst_local)
                pd = direct_port(dst_local, src_local)
                ts = tpn[(src, dst)][0]
                td = tpn[(dst, src)][0]
                add(src, ps, ts, dst, pd, td, 1, 1)
                add(src, ps, ts + ack_offset, dst, pd, td + ack_offset, 7, 1)
            else:
                for plane in range(PLANE_COUNT):
                    ts = tpn[(src, dst)][plane]
                    td = tpn[(dst, src)][plane]
                    add(src, plane, ts, dst, plane, td, 1, 4)
                for plane in range(PLANE_COUNT):
                    ts = tpn[(src, dst)][plane] + ack_offset
                    td = tpn[(dst, src)][plane] + ack_offset
                    add(src, plane, ts, dst, plane, td, 7, 4)

    return rows


def iter_routing(layout: PodLayout) -> Iterable[dict[str, object]]:
    def add(node: int, dst: int, dst_port: int, out_ports: list[int], metrics: list[int]) -> dict[str, object]:
        return {
            "nodeId": node,
            "dstNodeId": dst,
            "dstPortId": dst_port,
            "outPorts": " ".join(str(p) for p in out_ports),
            "metrics": " ".join(str(m) for m in metrics),
        }

    for node in range(layout.total_nodes):
        if node < layout.rank_count:
            src_group = node // GROUP_SIZE
            src_local = node % GROUP_SIZE
            for dst in range(layout.rank_count):
                if dst == node:
                    continue
                dst_group = dst // GROUP_SIZE
                dst_local = dst % GROUP_SIZE
                if dst_group == src_group:
                    yield add(node, dst, direct_port(dst_local, src_local), [direct_port(src_local, dst_local)], [1])
                else:
                    for plane in range(PLANE_COUNT):
                        yield add(node, dst, plane, [plane], [4])
        elif node < layout.l2_base:
            rel = node - layout.l1_base
            plane = rel // layout.group_count
            group = rel % layout.group_count
            block = group // GROUP_SIZE
            for dst in range(layout.rank_count):
                dst_group = dst // GROUP_SIZE
                dst_local = dst % GROUP_SIZE
                if dst_group == group:
                    yield add(node, dst, plane, [dst_local], [1])
                elif dst_group // GROUP_SIZE == block:
                    yield add(node, dst, plane, list(L1_TO_L2_PORTS) + list(L1_TO_TOP_PORTS), [3] * 8)
                else:
                    yield add(node, dst, plane, list(L1_TO_TOP_PORTS), [3] * 4)
        elif node < layout.top_base:
            rel = node - layout.l2_base
            plane = rel // (layout.l2_blocks_per_plane * 2)
            block_half = rel % (layout.l2_blocks_per_plane * 2)
            block = block_half // 2
            for dst in range(layout.rank_count):
                dst_group = dst // GROUP_SIZE
                if dst_group // GROUP_SIZE == block:
                    local_group = dst_group % GROUP_SIZE
                    yield add(node, dst, plane, [local_group * 2, local_group * 2 + 1], [2, 2])
        else:
            rel = node - layout.top_base
            plane = rel // 4
            for dst in range(layout.rank_count):
                dst_group = dst // GROUP_SIZE
                yield add(node, dst, plane, [dst_group], [2])


def generate_routing(layout: PodLayout) -> list[dict[str, object]]:
    return list(iter_routing(layout))


def write_node_file(layout: PodLayout, out: Path) -> None:
    lines = [
        "nodeId,nodeType,portNum,forwardDelay\n",
        f"0..{layout.rank_count - 1},DEVICE,15,1ns\n",
        f"{layout.l1_base}..{layout.top_base - 1},SWITCH,16,1ns\n",
        f"{layout.top_base}..{layout.total_nodes - 1},SWITCH,128,1ns\n",
    ]
    (out / "node.csv").write_text("".join(lines))


def write_node_mapping(layout: PodLayout, out: Path) -> None:
    rows: list[dict[str, object]] = []
    for node in range(layout.rank_count):
        rows.append({"nodeId": node, "originalNodeId": node, "role": "NPU"})
    for node in range(layout.l1_base, layout.l2_base):
        rows.append({"nodeId": node, "originalNodeId": node, "role": "L1"})
    for node in range(layout.l2_base, layout.top_base):
        rows.append({"nodeId": node, "originalNodeId": node, "role": "L2"})
    for node in range(layout.top_base, layout.total_nodes):
        rows.append({"nodeId": node, "originalNodeId": node, "role": "5808"})
    write_csv(out / "node_mapping.csv", ["nodeId", "originalNodeId", "role"], rows)


def write_topology_doc(layout: PodLayout, out: Path) -> None:
    physical_links = layout.rank_count * PLANE_COUNT + layout.l1_count * 8
    fullmesh_links = layout.group_count * (GROUP_SIZE * (GROUP_SIZE - 1) // 2)
    text = f"""# 8-plane {layout.rank_count}-NPU slice

Generated by `tools/generate_pod_topology.py`.

Construction rule:

- Keep NPU hosts `0..{layout.rank_count - 1}`.
- Use one L1 switch per `(plane, 8-rank group)`.
- Use two L2 switches per `(plane, 8-group block)`.
- Reuse the same 32 top-level 5808 switches for all pods; each 5808 uses one port per 8-rank group.

Counts:

- Hosts: {layout.rank_count}
- L1 switches: {layout.l1_count}
- L2 switches: {layout.l2_count}
- 5808 switches: {layout.top_count}
- Total nodes: {layout.total_nodes}
- Physical links: {physical_links}
- Host fullmesh links: {fullmesh_links}

Routing:

- Same 8-NPU group uses direct fullmesh shortest path.
- Other host pairs ignore host-host fullmesh as transit and route through L1/L2/5808.
- `routing_table.csv` is range-compressed by default. Use `--expanded-routing` to emit legacy per-node/per-destination rows.
- Compressed POD source topologies do not commit the generation-only `transport_channel.csv`; MeshClos fixed-plane cases still keep their reduced transport table to pin the selected plane.

Link model:

- Link bandwidth: `400Gbps`.
- Link delay: `1ns`.
"""
    (out / "TOPOLOGY.md").write_text(text)


def generate(rank_count: int, out: Path, network_attr: Path, compress_routes: bool = True) -> None:
    layout = PodLayout(rank_count)
    out.mkdir(parents=True, exist_ok=True)
    routing_rows: Iterable[dict[str, object]] = iter_routing(layout)
    if compress_routes:
        routing_rows = compress_routing(routing_rows)
    else:
        routing_rows = list(routing_rows)

    write_node_file(layout, out)
    write_node_mapping(layout, out)
    write_csv(
        out / "topology.csv",
        ["nodeId1", "portId1", "nodeId2", "portId2", "bandwidth", "delay"],
        generate_topology(layout),
    )
    write_csv(
        out / "transport_channel.csv",
        ["nodeId1", "portId1", "tpn1", "nodeId2", "portId2", "tpn2", "priority", "metric"],
        generate_transport(layout),
    )
    write_csv(
        out / "routing_table.csv",
        ["nodeId", "dstNodeId", "dstPortId", "outPorts", "metrics"],
        routing_rows,
    )
    (out / "traffic.csv").write_text("taskId,src,dst,opcode,opType,priority,dataSize(Byte),startTime\n")
    network_attr_out = out / "network_attribute.txt"
    if network_attr.resolve() != network_attr_out.resolve():
        shutil.copyfile(network_attr, network_attr_out)
    write_topology_doc(layout, out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank-count", type=int, default=256)
    parser.add_argument("--output-case", type=Path, default=REPO_ROOT / "experiments/topologies/pod256/generated_topology")
    parser.add_argument("--network-attribute", type=Path, default=DEFAULT_NETWORK_ATTR)
    parser.add_argument(
        "--expanded-routing",
        action="store_true",
        help="Write legacy fully expanded routing_table.csv instead of range-compressed routes.",
    )
    args = parser.parse_args()
    generate(args.rank_count, args.output_case, args.network_attribute, compress_routes=not args.expanded_routing)
    print(f"Generated {args.rank_count}-rank topology at {args.output_case}")


if __name__ == "__main__":
    main()
