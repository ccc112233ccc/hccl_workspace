#!/usr/bin/env python3
"""Generate POD AllToAll baseline and MeshClos fixed-plane cases."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIZES = ("16MB", "64MB", "128MB", "256MB", "512MB", "1024MB")


def size_suffix(size: str) -> str:
    text = size.strip().lower()
    for suffix in ("mib", "mb"):
        if text.endswith(suffix):
            return text[: -len(suffix)] + "mb"
    for suffix in ("gib", "gb"):
        if text.endswith(suffix):
            return str(int(text[: -len(suffix)]) * 1024) + "mb"
    if text.isdigit():
        return f"{text}b"
    return text.replace("/", "_")


def run(cmd: list[str], dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def compress_case_routing(case_dir: Path, dry_run: bool) -> None:
    run(
        [
            sys.executable,
            "tools/compress_routing_table.py",
            str(case_dir / "routing_table.csv"),
        ],
        dry_run,
    )


def remove_case_transport(case_dir: Path, dry_run: bool) -> None:
    transport_csv = case_dir / "transport_channel.csv"
    print(f"remove {transport_csv}", flush=True)
    if dry_run:
        return
    transport_csv.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate POD AllToAll cases for baseline packet-spray and MeshClos fixed-plane models."
    )
    parser.add_argument("--rank-count", type=int, required=True, help="POD rank count, e.g. 64, 128, 256, 512.")
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=list(DEFAULT_SIZES),
        help="Per-rank payload sizes. Default: 16MB 64MB 128MB 256MB 512MB 1024MB.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=("baseline", "fixedplane"),
        default=("baseline", "fixedplane"),
        help="Case families to generate.",
    )
    parser.add_argument(
        "--topology-case",
        type=Path,
        help="Existing source topology case. Default: experiments/topologies/pod{N}/generated_topology.",
    )
    parser.add_argument(
        "--generate-topology",
        action="store_true",
        help="Generate the source POD topology before generating traffic cases.",
    )
    parser.add_argument(
        "--cases-root",
        type=Path,
        help="Output cases directory. Default: experiments/pod{N}/alltoall/cases.",
    )
    parser.add_argument("--baseline-concurrent", type=int, default=16)
    parser.add_argument("--baseline-dependency-mode", default="thread-serial", choices=("thread-serial", "phase-barrier", "none"))
    parser.add_argument("--fixedplane-dependency-mode", default="v3-plane-step-barrier")
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument(
        "--expanded-routing",
        action="store_true",
        help="Keep legacy fully expanded routing_table.csv in generated cases.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    topology_case = args.topology_case or REPO_ROOT / f"experiments/topologies/pod{args.rank_count}/generated_topology"
    cases_root = args.cases_root or REPO_ROOT / f"experiments/pod{args.rank_count}/alltoall/cases"

    if args.generate_topology:
        run(
            [
                sys.executable,
                "tools/generate_pod_topology.py",
                "--rank-count",
                str(args.rank_count),
                "--output-case",
                str(topology_case),
            ]
            + (["--expanded-routing"] if args.expanded_routing else []),
            args.dry_run,
        )
    elif not topology_case.exists() and not args.dry_run:
        raise FileNotFoundError(f"topology case does not exist: {topology_case}")

    for size in args.sizes:
        suffix = size_suffix(size)
        if "baseline" in args.algorithms:
            out = cases_root / f"generated_topology_pod{args.rank_count}_hccl_baseline_threadserial_a2a{args.rank_count}_{suffix}"
            run(
                [
                    sys.executable,
                    "tools/generate_hccl_mesh1d_alltoallv_case.py",
                    "--source-case",
                    str(topology_case),
                    "--output-case",
                    str(out),
                    "--rank-count",
                    str(args.rank_count),
                    "--per-rank-bytes",
                    size,
                    "--concurrent",
                    str(args.baseline_concurrent),
                    "--tp-mode",
                    "full",
                    "--dependency-mode",
                    args.baseline_dependency_mode,
                    "--no-port-trace",
                ],
                args.dry_run,
            )
            if not args.expanded_routing:
                compress_case_routing(out, args.dry_run)
                remove_case_transport(out, args.dry_run)

        if "fixedplane" in args.algorithms:
            out = cases_root / f"generated_topology_pod{args.rank_count}_hccl_meshclos2d_v3_strict_planestep_a2a{args.rank_count}_{suffix}"
            run(
                [
                    sys.executable,
                    "tools/generate_hccl_meshclos2d_v3_alltoall_case.py",
                    "--source-case",
                    str(topology_case),
                    "--output-case",
                    str(out),
                    "--rank-count",
                    str(args.rank_count),
                    "--group-size",
                    str(args.group_size),
                    "--direct-per-rank-bytes",
                    size,
                    "--mode",
                    "strict",
                    "--traffic-order",
                    "v3-logical",
                    "--dependency-mode",
                    args.fixedplane_dependency_mode,
                    "--no-port-trace",
                ],
                args.dry_run,
            )
            if not args.expanded_routing:
                compress_case_routing(out, args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
