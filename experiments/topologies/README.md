# 拓扑目录

| 目录 | 说明 |
| --- | --- |
| `ubx16/generated_topology_ubx16` | 当前 UBX16 实验默认拓扑。生成脚本默认也指向这里。 |
| `pod64/generated_topology` | POD 64P 拓扑，只保留拓扑和基础配置，仿真结果已清理。 |
| `pod128/generated_topology` | POD 128P 拓扑，复用 32 个顶层 5808。 |
| `pod256/generated_topology` | POD 256P 拓扑，复用 32 个顶层 5808。 |

UBX16 拓扑可用 `tools/generate_ubx16_topology.py` 重新生成；默认输出仍是 `ubx16/generated_topology_ubx16`。
POD 拓扑可用 `tools/generate_pod_topology.py --rank-count <64|128|256>` 生成。
