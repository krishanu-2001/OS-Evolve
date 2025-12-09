#!/usr/bin/env python3
"""
Evaluation harness for ghOSt CFS scheduler experiments with sysbench.

This script performs the following steps:
1. Applies edits from the single evolvable file (plus optional `// FILE:` blocks)
   back into the `ghost-userspace` checkout.
2. Changes directory to the repo and runs (optionally via sudo):
     bazel build -c opt agent_cfs_test
     bazel build -c opt cfs_sysbench_test
     ./bazel-bin/cfs_sysbench_test --create_enclave_and_agent ...
   The sysbench binary writes stdout to `metrics/output.log`.
3. Parses `metrics/output.log` and the generated CSV (`metrics/cfs_sysbench.csv`) to
   extract CPU performance metrics and saves them into `metrics.json`.
4. Marks the run as correct only when every command returns 0.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

FILE_MARKER = "// FILE:"
DEFAULT_METRICS_LOG = "metrics/output.log"
DEFAULT_METRICS_CSV = "metrics/cfs_sysbench.csv"


class RepoFileManager:
    """Tracks and restores modifications inside the ghost-userspace repo."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root.resolve()
        self._backups: Dict[Path, str | None] = {}

    def _validate_path(self, rel_path: str) -> Path:
        candidate = (self.repo_root / rel_path).resolve()
        if not str(candidate).startswith(str(self.repo_root)):
            raise ValueError(
                f"Attempted to write outside the repo: {rel_path} -> {candidate}"
            )
        return candidate

    def write(self, rel_path: str, content: str) -> None:
        target = self._validate_path(rel_path)
        if target not in self._backups:
            self._backups[target] = (
                target.read_text(encoding="utf-8") if target.exists() else None
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def restore(self) -> None:
        for path, original in self._backups.items():
            if original is None:
                if path.exists():
                    path.unlink()
            else:
                path.write_text(original, encoding="utf-8")


def default_ghost_root() -> Path:
    """Infers the ghost-userspace checkout relative to this script."""
    this_file = Path(__file__).resolve()
    shinka_root = this_file.parents[2]
    return (shinka_root / "ghost-userspace").resolve()


def parse_program_sources(program_text: str) -> Dict[str, str]:
    """Split monolithic source into per-file payloads using `// FILE:` markers."""
    lines = program_text.splitlines(keepends=True)
    files: Dict[str, List[str]] = {}
    current_path: Optional[str] = None

    for line in lines:
        if line.startswith(FILE_MARKER):
            current_path = line.split(":", 1)[1].strip()
            if not current_path:
                raise ValueError("FILE marker missing relative path.")
            files.setdefault(current_path, [])
            continue

        target = current_path or "schedulers/test/cfs_scheduler.cc"
        files.setdefault(target, []).append(line)

    return {path: "".join(contents) for path, contents in files.items()}


@dataclass
class CommandResult:
    label: str
    command: List[str]
    returncode: int
    duration_sec: float
    log_path: Path


def prepend_sudo(cmd: List[str], use_sudo: bool) -> List[str]:
    return ["sudo"] + cmd if use_sudo else cmd


def run_command_with_logs(
    base_cmd: List[str],
    label: str,
    cwd: Path,
    results_dir: Path,
    use_sudo: bool,
    stdout_path: Optional[Path] = None,
) -> CommandResult:
    """Runs a command, storing stdout/stderr in a log file."""
    cmd = prepend_sudo(base_cmd, use_sudo)
    start = time.time()

    if stdout_path:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("w", encoding="utf-8") as capture:
            process = subprocess.run(
                cmd,
                cwd=str(cwd),
                stdout=capture,
                stderr=subprocess.STDOUT,
                text=True,
            )
        stdout_contents = stdout_path.read_text(encoding="utf-8")
        stderr_contents = ""
    else:
        process = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
        )
        stdout_contents = process.stdout
        stderr_contents = process.stderr

    duration = time.time() - start
    log_path = results_dir / f"{label}.log"
    log_path.write_text(
        stdout_contents + ("\n" + stderr_contents if stderr_contents else ""),
        encoding="utf-8",
    )

    return CommandResult(
        label=label,
        command=cmd,
        returncode=process.returncode,
        duration_sec=duration,
        log_path=log_path,
    )


def parse_sysbench_log(log_text: str) -> Dict[str, float]:
    """Extract metrics from cfs_sysbench_test stdout."""
    metrics: Dict[str, float] = {}
    
    # Check if sysbench was successfully moved to ghOSt
    moved_match = re.search(
        r"\[sysbenchCfs\] Moving sysbench \(pid=(\d+)\) and (\d+) threads into ghOSt",
        log_text,
    )
    if moved_match:
        metrics["sysbench_pid"] = float(moved_match.group(1))
        metrics["num_threads"] = float(moved_match.group(2))
    
    # Check for success messages
    success_match = re.search(
        r"\[sysbenchCfs\] SUCCESS: Moved main process",
        log_text,
    )
    if success_match:
        metrics["moved_to_ghost"] = 1.0
    else:
        metrics["moved_to_ghost"] = 0.0
    
    # Extract sysbench output (if present in log)
    events_match = re.search(
        r"events per second:\s+([\d\.]+)",
        log_text,
    )
    if events_match:
        metrics["events_per_sec"] = float(events_match.group(1))
    
    return metrics


def parse_sysbench_csv(csv_path: Path) -> Dict[str, float]:
    """Parse sysbench CSV output."""
    if not csv_path.exists():
        return {}

    rows: List[Dict[str, str]] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return {}

    def to_float(row: Dict[str, str], key: str) -> float:
        try:
            return float(row[key])
        except (ValueError, KeyError):
            return 0.0

    # Get the last row (most recent run)
    last_row = rows[-1] if rows else {}
    
    total_time = to_float(last_row, "total_time_sec")
    events_per_sec = to_float(last_row, "events_per_sec")
    total_events = to_float(last_row, "total_events")
    min_latency = to_float(last_row, "min_latency_ms")
    avg_latency = to_float(last_row, "avg_latency_ms")
    max_latency = to_float(last_row, "max_latency_ms")

    return {
        "csv_total_time_sec": total_time,
        "csv_events_per_sec": events_per_sec,
        "csv_total_events": total_events,
        "csv_min_latency_ms": min_latency,
        "csv_avg_latency_ms": avg_latency,
        "csv_max_latency_ms": max_latency,
    }


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=4), encoding="utf-8")


def compute_combined_score(
    fitness_metric: str,
    log_metrics: Dict[str, float],
    csv_metrics: Dict[str, float],
) -> float:
    """Returns a fitness score where larger is always better."""
    if fitness_metric == "events_per_sec":
        # Higher events/sec is better
        return float(csv_metrics.get("csv_events_per_sec", 0.0))
    if fitness_metric == "avg_latency":
        # Lower latency is better, so invert it
        avg_lat = csv_metrics.get("csv_avg_latency_ms", float('inf'))
        if avg_lat == 0.0 or avg_lat == float('inf'):
            return 0.0
        return 1000.0 / (1.0 + avg_lat)  # Inverse latency
    if fitness_metric == "throughput_latency_ratio":
        # Combined metric: events/sec / avg_latency
        events = csv_metrics.get("csv_events_per_sec", 0.0)
        avg_lat = csv_metrics.get("csv_avg_latency_ms", 1.0)
        if avg_lat == 0.0:
            return 0.0
        return events / (1.0 + avg_lat / 1000.0)  # Normalize latency
    raise ValueError(f"Unknown fitness metric '{fitness_metric}'")


def main(
    program_path: str,
    results_dir: str,
    ghost_root: Optional[str],
    use_sudo: bool,
    ghost_cpus: str,
    sysbench_threads: int,
    metrics_log_relpath: str,
    metrics_csv_relpath: str,
    fitness_metric: str,
) -> None:
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    program_text = Path(program_path).read_text(encoding="utf-8")
    sources = parse_program_sources(program_text)

    repo_root = Path(ghost_root).resolve() if ghost_root else default_ghost_root()

    if not repo_root.exists():
        raise FileNotFoundError(f"ghost-userspace repo not found at {repo_root}")

    manager = RepoFileManager(repo_root)
    metrics: Dict[str, object] = {}
    correct = False
    error = ""

    try:
        for rel_path, contents in sources.items():
            manager.write(rel_path, contents)

        build_results: List[CommandResult] = []
        for label, target in (
            ("build_agent_cfs_test", "agent_cfs_test"),
            ("build_cfs_sysbench_test", "cfs_sysbench_test"),
        ):
            cmd = ["bazel", "build", "-c", "opt", target]
            result = run_command_with_logs(
                cmd,
                label,
                repo_root,
                results_path,
                use_sudo=use_sudo,
            )
            build_results.append(result)
            if result.returncode != 0:
                raise RuntimeError(
                    f"{target} build failed with return code {result.returncode}"
                )

        sysbench_log_path = repo_root / metrics_log_relpath
        sysbench_cmd = [
            "./bazel-bin/cfs_sysbench_test",
            "--create_enclave_and_agent",
            f"--ghost_cpus={ghost_cpus}",
            f"--sysbench_threads={sysbench_threads}",
        ]
        sysbench_result = run_command_with_logs(
            sysbench_cmd,
            "run_cfs_sysbench_test",
            repo_root,
            results_path,
            use_sudo=use_sudo,
            stdout_path=sysbench_log_path,
        )
        sysbench_log_text = sysbench_log_path.read_text(encoding="utf-8")
        (results_path / "sysbench_output.log").write_text(
            sysbench_log_text, encoding="utf-8"
        )
        if sysbench_result.returncode != 0:
            raise RuntimeError(
                f"cfs_sysbench_test failed with return code {sysbench_result.returncode}"
            )

        log_metrics = parse_sysbench_log(sysbench_log_text)
        csv_metrics = parse_sysbench_csv(repo_root / metrics_csv_relpath)

        public_metrics = {
            "builds": [
                {
                    "label": r.label,
                    "return_code": r.returncode,
                    "duration_sec": r.duration_sec,
                    "log_path": str(r.log_path),
                }
                for r in build_results
            ],
            "sysbench_log_metrics": log_metrics,
            "sysbench_csv_metrics": csv_metrics,
        }
        private_metrics = {
            "commands": [
                {
                    "label": sysbench_result.label,
                    "return_code": sysbench_result.returncode,
                    "duration_sec": sysbench_result.duration_sec,
                    "log_path": str(sysbench_result.log_path),
                    "stdout_capture": str(sysbench_log_path),
                }
            ],
        }

        combined_score = compute_combined_score(
            fitness_metric, log_metrics, csv_metrics
        )
        metrics = {
            "combined_score": combined_score,
            "public": public_metrics,
            "private": private_metrics,
            "log_excerpt": sysbench_log_text[-2000:],
            "csv_path": str((repo_root / metrics_csv_relpath).resolve()),
            "fitness_metric": fitness_metric,
        }
        correct = True
    except Exception as exc:  # pylint: disable=broad-except
        error = f"{exc}\n{traceback.format_exc()}"
        metrics = {
            "combined_score": 0.0,
            "public": {},
            "private": {"error": error},
        }
    finally:
        manager.restore()

    write_json(results_path / "metrics.json", metrics)
    write_json(
        results_path / "correct.json",
        {"correct": correct, "error": error},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a candidate CFS scheduler implementation."
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.cc",
        help="Path to the monolithic program file produced by Shinka.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to store metrics and logs.",
    )
    parser.add_argument(
        "--ghost_root",
        type=str,
        default=None,
        help="Path to the ghost-userspace checkout (auto-detected if omitted).",
    )
    parser.add_argument(
        "--use_sudo",
        action="store_true",
        default=True,
        help="Run bazel/cfs_sysbench_test commands under sudo.",
    )
    parser.add_argument(
        "--ghost_cpus",
        type=str,
        default="1-5",
        help="CPU range passed to cfs_sysbench_test when creating the enclave.",
    )
    parser.add_argument(
        "--sysbench_threads",
        type=int,
        default=4,
        help="Number of threads to use for sysbench CPU test.",
    )
    parser.add_argument(
        "--metrics_log",
        type=str,
        default=DEFAULT_METRICS_LOG,
        help="Relative path (within ghost-userspace) of sysbench stdout capture.",
    )
    parser.add_argument(
        "--metrics_csv",
        type=str,
        default=DEFAULT_METRICS_CSV,
        help="Relative path of sysbench CSV output.",
    )
    parser.add_argument(
        "--fitness_metric",
        choices=("events_per_sec", "avg_latency", "throughput_latency_ratio"),
        default="events_per_sec",
        help="Optimization target used to compute combined_score.",
    )
    args = parser.parse_args()
    main(
        args.program_path,
        args.results_dir,
        args.ghost_root,
        use_sudo=args.use_sudo,
        ghost_cpus=args.ghost_cpus,
        sysbench_threads=args.sysbench_threads,
        metrics_log_relpath=args.metrics_log,
        metrics_csv_relpath=args.metrics_csv,
        fitness_metric=args.fitness_metric,
    )
