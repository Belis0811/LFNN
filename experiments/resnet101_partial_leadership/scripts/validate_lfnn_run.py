#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from lfnnexp.provenance import sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate completeness of an LFNN partial-leadership run."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--expected-epochs", type=int, default=90)
    parser.add_argument("--expected-validation-examples", type=int, default=50000)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return records


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    required_paths = {
        "config": run_dir / "resolved_config.yaml",
        "metadata": run_dir / "run_metadata.json",
        "metrics": run_dir / "metrics.jsonl",
        "trace": run_dir / "selection_trace.jsonl",
        "last_checkpoint": run_dir / "checkpoints" / "last.pt",
        "best_checkpoint": run_dir / "checkpoints" / "best.pt",
    }
    missing = [name for name, path in required_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing run artifacts: {missing}")
    with required_paths["config"].open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with required_paths["metadata"].open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    metrics = read_jsonl(required_paths["metrics"])
    trace = read_jsonl(required_paths["trace"])
    selection_records = [row for row in trace if row.get("event") == "selection"]
    epoch_records = [row for row in metrics if row.get("event") == "epoch_end"]

    issues = []
    if metadata.get("git", {}).get("dirty") is not False:
        issues.append("The recorded Git worktree was not clean.")
    if metadata.get("max_steps_per_epoch") is not None:
        issues.append("The run used max_steps_per_epoch and is not a full training run.")
    if len(epoch_records) != args.expected_epochs:
        issues.append(
            f"Found {len(epoch_records)} completed epochs; expected {args.expected_epochs}."
        )
    if not selection_records:
        issues.append("No leader/follower selection records were found.")

    empty_follower_steps = []
    zero_follower_loss_steps = []
    zero_follower_gradient_steps = []
    leader_sets = set()
    best_leaders = set()
    for row in selection_records:
        step = int(row["global_step"])
        followers = [int(index) for index in row.get("follower_ids", [])]
        leaders = tuple(int(index) for index in row.get("leader_ids", []))
        leader_sets.add(leaders)
        best_leaders.add(int(row["best_leader_id"]))
        if not followers:
            empty_follower_steps.append(step)
        if float(row.get("follower_alignment_loss", 0.0)) <= 0.0:
            zero_follower_loss_steps.append(step)
        norms = row.get("worker_gradient_norms", [])
        if followers and not all(
            index < len(norms) and float(norms[index]) > 0.0 for index in followers
        ):
            zero_follower_gradient_steps.append(step)
    if empty_follower_steps:
        issues.append(
            f"Follower set was empty in {len(empty_follower_steps)} recorded steps."
        )
    if selection_records and len(zero_follower_loss_steps) == len(selection_records):
        issues.append("Follower-alignment loss was zero in every recorded step.")
    if zero_follower_gradient_steps:
        issues.append(
            f"At least one follower had zero/missing gradient in "
            f"{len(zero_follower_gradient_steps)} recorded steps."
        )

    validation_counts = sorted(
        {int(row["validation"]["count"]) for row in epoch_records}
    )
    if validation_counts != [args.expected_validation_examples]:
        issues.append(
            f"Validation counts are {validation_counts}; expected only "
            f"{args.expected_validation_examples}."
        )
    if epoch_records and not any(row.get("is_best") for row in epoch_records):
        issues.append("No epoch was marked as the best checkpoint.")

    artifacts = {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in required_paths.items()
    }
    report = {
        "valid": not issues,
        "issues": issues,
        "run_dir": str(run_dir),
        "configured_leadership_fraction": config["lfnn"]["leadership_fraction"],
        "epoch_record_count": len(epoch_records),
        "selection_record_count": len(selection_records),
        "empty_follower_step_count": len(empty_follower_steps),
        "nonpositive_follower_loss_step_count": len(zero_follower_loss_steps),
        "zero_follower_gradient_step_count": len(zero_follower_gradient_steps),
        "unique_leader_sets": [list(values) for values in sorted(leader_sets)],
        "unique_best_leaders": sorted(best_leaders),
        "validation_counts": validation_counts,
        "artifacts": artifacts,
    }
    write_json(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

