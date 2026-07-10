#!/usr/bin/env python3
"""Generate a compact N-NPU slice from the 8-plane 1024-NPU topology.

The slice keeps original hosts 0..N-1 and all related switches/links:

- NPU to L1 links for the retained original hosts.
- Full-mesh links inside each 8-NPU group.
- L1 to L2 links for the retained L1 switches.
- L1 to 5808 links for the retained L1 switches.

Output node IDs are compacted to a dense 0..N-1 range for ns-3. The original
1024-case node IDs are preserved in node_mapping.csv.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from collections import defaultdict, deque
from pathlib import Path


CASE_DIR = Path(__file__).resolve().parent
A5_ROOT = CASE_DIR
LOCAL_SOURCE_TOPOLOGY = A5_ROOT / "generate_8plane_1024npu_topology.py"
LEGACY_SOURCE_TOPOLOGY = A5_ROOT / "generate_8plane_1024npu_topology.py"
DEFAULT_HOST_COUNT = 64
HOST_COUNT = DEFAULT_HOST_COUNT
DEFAULT_LINK_DELAY = "1ns"
CSV_LINE_TERMINATOR = "\n"


def load_source_topology():
    source_path = LOCAL_SOURCE_TOPOLOGY if LOCAL_SOURCE_TOPOLOGY.exists() else LEGACY_SOURCE_TOPOLOGY
    spec = importlib.util.spec_from_file_location("source_topology", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load source topology: {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def compact_mapping(source_topo, links):
    r = source_topo.ranges()
    retained_hosts = set(range(HOST_COUNT))
    retained_l1 = set()
    retained_l2 = set()
    retained_m5808 = set()

    for a, _ap, b, _bp, _bw, _delay in links:
        if a in retained_hosts and r.l1_start <= b < r.l2_start:
            retained_l1.add(b)
        if b in retained_hosts and r.l1_start <= a < r.l2_start:
            retained_l1.add(a)

    for a, _ap, b, _bp, _bw, _delay in links:
        if a in retained_l1 and r.l2_start <= b < r.m5808_start:
            retained_l2.add(b)
        if b in retained_l1 and r.l2_start <= a < r.m5808_start:
            retained_l2.add(a)
        if a in retained_l1 and r.m5808_start <= b < r.total_nodes:
            retained_m5808.add(b)
        if b in retained_l1 and r.m5808_start <= a < r.total_nodes:
            retained_m5808.add(a)

    original_ids = (
        sorted(retained_hosts)
        + sorted(retained_l1)
        + sorted(retained_l2)
        + sorted(retained_m5808)
    )
    return {old_id: new_id for new_id, old_id in enumerate(original_ids)}


def filter_and_remap_links(source_topo, links, mapping):
    r = source_topo.ranges()
    remapped = []
    for a, ap, b, bp, bw, delay in links:
        keep = False
        if a in mapping and b in mapping:
            a_is_host = a < source_topo.HOST_COUNT
            b_is_host = b < source_topo.HOST_COUNT
            a_is_l1 = r.l1_start <= a < r.l2_start
            b_is_l1 = r.l1_start <= b < r.l2_start
            a_is_l2 = r.l2_start <= a < r.m5808_start
            b_is_l2 = r.l2_start <= b < r.m5808_start
            a_is_5808 = r.m5808_start <= a < r.total_nodes
            b_is_5808 = r.m5808_start <= b < r.total_nodes
            keep = (
                (a_is_host and b_is_host)
                or (a_is_host and b_is_l1)
                or (b_is_host and a_is_l1)
                or (a_is_l1 and b_is_l2)
                or (b_is_l1 and a_is_l2)
                or (a_is_l1 and b_is_5808)
                or (b_is_l1 and a_is_5808)
            )
        if keep:
            remapped.append((mapping[a], ap, mapping[b], bp, bw, delay))
    return remapped


def contiguous_ranges(values):
    values = sorted(values)
    if not values:
        return []
    ranges = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append((start, prev))
        start = prev = value
    ranges.append((start, prev))
    return ranges


def range_text(start, end):
    return str(start) if start == end else f"{start}..{end}"


def write_node_csv(output_dir, source_topo, mapping):
    r = source_topo.ranges()
    groups = defaultdict(list)
    for old_id, new_id in mapping.items():
        if old_id < HOST_COUNT:
            groups[("DEVICE", source_topo.HOST_PORTS)].append(new_id)
        elif r.l1_start <= old_id < r.l2_start:
            groups[("SWITCH", source_topo.L1_PORTS)].append(new_id)
        elif r.l2_start <= old_id < r.m5808_start:
            groups[("SWITCH", source_topo.L2_PORTS)].append(new_id)
        else:
            groups[("SWITCH", source_topo.M5808_PORTS)].append(new_id)

    rows = []
    for (node_type, port_count), node_ids in groups.items():
        for start, end in contiguous_ranges(node_ids):
            rows.append((start, range_text(start, end), node_type, port_count))
    rows.sort()

    with (output_dir / "node.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator=CSV_LINE_TERMINATOR)
        writer.writerow(["nodeId", "nodeType", "portNum", "forwardDelay"])
        for _sort_key, node_id, node_type, port_count in rows:
            writer.writerow([node_id, node_type, port_count, source_topo.DEFAULT_FORWARD_DELAY])


def write_mapping_csv(output_dir, source_topo, mapping):
    r = source_topo.ranges()
    with (output_dir / "node_mapping.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator=CSV_LINE_TERMINATOR)
        writer.writerow(["nodeId", "originalNodeId", "role"])
        for old_id, new_id in sorted(mapping.items(), key=lambda item: item[1]):
            if old_id < HOST_COUNT:
                role = "NPU"
            elif r.l1_start <= old_id < r.l2_start:
                role = "L1"
            elif r.l2_start <= old_id < r.m5808_start:
                role = "L2"
            else:
                role = "M5808"
            writer.writerow([new_id, old_id, role])


def write_topology_csv(output_dir, links):
    with (output_dir / "topology.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator=CSV_LINE_TERMINATOR)
        writer.writerow(["nodeId1", "portId1", "nodeId2", "portId2", "bandwidth", "delay"])
        writer.writerows(links)


def build_adjacency(links):
    adjacency = defaultdict(list)
    for a, ap, b, bp, _bw, _delay in links:
        adjacency[a].append((b, ap, bp))
        adjacency[b].append((a, bp, ap))
    return adjacency


def npu_group_id(host_id):
    return host_id // 8


def shortest_paths(adjacency, src, dst):
    if npu_group_id(src) == npu_group_id(dst):
        return [[src, dst]]

    distances = {src: 0}
    parents = defaultdict(list)
    queue = deque([src])
    while queue:
        node = queue.popleft()
        for nxt, _out_port, _dst_port in adjacency[node]:
            if node < HOST_COUNT and nxt < HOST_COUNT:
                continue
            distance = distances[node] + 1
            if nxt not in distances:
                distances[nxt] = distance
                parents[nxt].append(node)
                queue.append(nxt)
            elif distances[nxt] == distance:
                parents[nxt].append(node)

    if dst not in distances:
        return []

    paths = []

    def backtrack(node, suffix):
        if node == src:
            paths.append([src] + suffix)
            return
        for parent in parents[node]:
            backtrack(parent, [node] + suffix)

    backtrack(dst, [])
    return paths


def route_port(adjacency, node, next_hop):
    return [out_port for neighbor, out_port, _dst_port in adjacency[node] if neighbor == next_hop]


def dst_ports(adjacency, dst, prev_hop):
    return [out_port for neighbor, out_port, _dst_port in adjacency[dst] if neighbor == prev_hop]


def add_route(routes, adjacency, curr, next_hop, prev_before_dst, dst, metric):
    for dst_port in dst_ports(adjacency, dst, prev_before_dst):
        out_ports = route_port(adjacency, curr, next_hop)
        route_set = routes[(curr, dst, dst_port)]
        for out_port in out_ports:
            old_metric = route_set.get(out_port)
            if old_metric is None or old_metric > metric:
                route_set[out_port] = metric


def write_routing_table(output_dir, links):
    adjacency = build_adjacency(links)
    routes = defaultdict(dict)

    for src in range(HOST_COUNT):
        for dst in range(src + 1, HOST_COUNT):
            for path in shortest_paths(adjacency, src, dst):
                for i, node in enumerate(path):
                    if i >= 1:
                        add_route(routes, adjacency, node, path[i - 1], path[1], src, i)
                    if i < len(path) - 1:
                        add_route(routes, adjacency, node, path[i + 1], path[-2], dst, len(path) - i - 1)

    with (output_dir / "routing_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator=CSV_LINE_TERMINATOR)
        writer.writerow(["nodeId", "dstNodeId", "dstPortId", "outPorts", "metrics"])
        for node, dst, dst_port in sorted(routes):
            port_metrics = sorted(routes[(node, dst, dst_port)].items())
            writer.writerow(
                [
                    node,
                    dst,
                    dst_port,
                    " ".join(str(port) for port, _metric in port_metrics),
                    " ".join(str(metric) for _port, metric in port_metrics),
                ]
            )


def write_traffic_csv(output_dir):
    with (output_dir / "traffic.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator=CSV_LINE_TERMINATOR)
        writer.writerow(
            [
                "taskId",
                "src",
                "dst",
                "opcode",
                "opType",
                "priority",
                "dataSize(Byte)",
                "startTime",
            ]
        )
        writer.writerow([0, 0, 63, "WRITE", "TA_LDST", 0, 65536, "0us"])


def write_transport_channel_csv(output_dir):
    (output_dir / "transport_channel.csv").write_text(
        "nodeId1,portId1,tpn1,nodeId2,portId2,tpn2,priority,metric\n",
        encoding="utf-8",
    )


def write_network_attribute(output_dir):
    text = """default ns3::UbApp::EnableMultiPath "true"
default ns3::UbApp::UseShortestPaths "true"
default ns3::UbLink::Delay "+0ns"
default ns3::UbPort::UbDataRate "400Gbps"
default ns3::UbPort::UbInterframeGap "+0ns"
default ns3::UbPort::CbfcFlitLenByte "20"
default ns3::UbPort::CbfcFlitsPerCell "8"
default ns3::UbPort::CbfcRetCellGrainDataPacket "4"
default ns3::UbPort::CbfcRetCellGrainControlPacket "32"
default ns3::UbPort::CbfcCtrlCrdRtrThldCell "1024"
default ns3::UbPort::CbfcInitCreditCell "6553"
default ns3::UbPort::PfcUpThld "819200"
default ns3::UbPort::PfcLowThld "655360"
default ns3::UbSwitch::FlowControl "CBFC"
default ns3::UbJetty::JettyOooAckThreshold "2048"
default ns3::UbJetty::UbInflightMax "10000"
default ns3::UbTransportChannel::EnableRetrans "false"
default ns3::UbTransportChannel::InitialRTO "+25600ns"
default ns3::UbTransportChannel::MaxRetransAttempts "7"
default ns3::UbTransportChannel::RetransExponentFactor "1"
default ns3::UbTransportChannel::DefaultMaxWqeSegNum "1000"
default ns3::UbTransportChannel::DefaultMaxInflightPacketSize "1000"
default ns3::UbTransportChannel::TpOooThreshold "2048"
default ns3::UbTransportChannel::UsePacketSpray "true"
default ns3::UbTransportChannel::UseShortestPaths "true"
default ns3::UbSwitchAllocator::AllocationTime "+10ns"
default ns3::UbLdstInstance::ThreadNum "10"
default ns3::UbLdstInstance::QueuePriority "1"
default ns3::UbLdstThread::LoadResponseSize "512"
default ns3::UbLdstThread::StoreRequestSize "512"
default ns3::UbLdstThread::LoadRequestSize "64"
default ns3::UbLdstThread::StoreOutstanding "64"
default ns3::UbLdstThread::LoadOutstanding "64"
default ns3::UbLdstApi::UsePacketSpray "true"
default ns3::UbLdstApi::UseShortestPaths "true"
default ns3::UbCaqm::UbCaqmAlpha "0.5"
default ns3::UbCaqm::UbCaqmBeta "0.5"
default ns3::UbCaqm::UbCaqmGamma "0.5"
default ns3::UbCaqm::UbCaqmLambda "0.5"
default ns3::UbCaqm::UbCaqmTheta "10"
default ns3::UbCaqm::UbCaqmQt "40960"
default ns3::UbCaqm::UbCaqmCcUint "32"
default ns3::UbCaqm::UbMarkProbability "0.1"
default ns3::UbHostCaqm::UbCaqmCwnd "40960"
default ns3::UbSwitchCaqm::UbCcUpdatePeriod "+500ns"
default ns3::UbQueueManager::ReservePerQueueBytes "1048576"
default ns3::UbFault::UbFaultUsePacketSpray "false"

global UB_FAULT_ENABLE "false"
global UB_PRIORITY_NUM "16"
global UB_VL_NUM "16"
global UB_CC_ALGO "CAQM"
global UB_CC_ENABLED "false"

global UB_TRACE_ENABLE "true"
global UB_TASK_TRACE_ENABLE "true"
global UB_PACKET_TRACE_ENABLE "true"
global UB_PORT_TRACE_ENABLE "false"
global UB_QUEUE_TRACE_ENABLE "false"
global UB_RECORD_PKT_TRACE "true"
global UB_FLOW_CONTROL_TRACE_ENABLE "false"
global UB_CONGESTION_CONTROL_TRACE_ENABLE "false"

global UB_PARSE_TRACE_ENABLE "true"
global UB_PYTHON_SCRIPT_PATH "scratch/ns-3-ub-tools/trace_analysis/parse_trace.py"
"""
    (output_dir / "network_attribute.txt").write_text(text, encoding="utf-8")


def write_summary(output_dir, source_topo, mapping, links):
    roles = defaultdict(int)
    r = source_topo.ranges()
    for old_id in mapping:
        if old_id < HOST_COUNT:
            roles["hosts"] += 1
        elif r.l1_start <= old_id < r.l2_start:
            roles["l1"] += 1
        elif r.l2_start <= old_id < r.m5808_start:
            roles["l2"] += 1
        else:
            roles["m5808"] += 1
    fullmesh_links = sum(1 for a, _ap, b, _bp, _bw, _delay in links if a < HOST_COUNT and b < HOST_COUNT)
    text = f"""# 8-plane {HOST_COUNT}-NPU slice

Generated from the 1024-NPU 8-plane topology script, not by manual CSV editing.

Source case:

- `../20260509-8plane-1024npu-topology`

Slice rule:

- Keep original NPU hosts `0..{HOST_COUNT - 1}`.
- Keep every switch directly related to those hosts: their L1 switches, those
  L1 switches' L2 switches, and those L1 switches' 5808 switches.
- Keep links among retained nodes only.
- Compact node IDs for ns-3 runtime; see `node_mapping.csv` for original IDs.

Counts:

- Hosts: {roles["hosts"]}
- L1 switches: {roles["l1"]}
- L2 switches: {roles["l2"]}
- 5808 switches: {roles["m5808"]}
- Total nodes: {len(mapping)}
- Physical links: {len(links)}
- Host fullmesh links: {fullmesh_links}

Routing:

- Same 8-NPU group uses direct fullmesh shortest path.
- Other host pairs ignore host-host fullmesh as transit and route through L1/L2/5808.

Link model:

- Link bandwidth: `400Gbps`.
- Link delay: `{DEFAULT_LINK_DELAY}`.
"""
    (output_dir / "TOPOLOGY.md").write_text(text, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=CASE_DIR)
    parser.add_argument(
        "--host-count",
        type=int,
        default=DEFAULT_HOST_COUNT,
        help="Number of original hosts to keep from the 1024-NPU topology.",
    )
    return parser.parse_args()


def main():
    global HOST_COUNT
    args = parse_args()
    HOST_COUNT = args.host_count
    if HOST_COUNT <= 0 or HOST_COUNT > 1024 or HOST_COUNT % 8 != 0:
        raise ValueError("--host-count must be a positive multiple of 8 and no larger than 1024")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_topo = load_source_topology()
    source_topo.validate_static_spec()
    source_links, _used_ports = source_topo.build_links(
        source_topo.DEFAULT_BANDWIDTH,
        DEFAULT_LINK_DELAY,
    )
    mapping = compact_mapping(source_topo, source_links)
    links = filter_and_remap_links(source_topo, source_links, mapping)

    write_node_csv(output_dir, source_topo, mapping)
    write_mapping_csv(output_dir, source_topo, mapping)
    write_topology_csv(output_dir, links)
    write_routing_table(output_dir, links)
    write_traffic_csv(output_dir)
    write_transport_channel_csv(output_dir)
    write_network_attribute(output_dir)
    write_summary(output_dir, source_topo, mapping, links)

    print(f"output_dir={output_dir}")
    print(f"nodes={len(mapping)} links={len(links)} hosts={HOST_COUNT}")
    print("wrote node.csv node_mapping.csv topology.csv routing_table.csv traffic.csv")
    print("wrote transport_channel.csv network_attribute.txt TOPOLOGY.md")


if __name__ == "__main__":
    main()
