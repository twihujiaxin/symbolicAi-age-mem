# -*- coding: utf-8 -*-
"""Command-line query and deterministic replay for AgeMem trajectory JSONL."""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from .trajectory import TrajectoryReplay, TrajectoryValidationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Trajectory JSONL path")
    parser.add_argument("--task-id")
    parser.add_argument("--rollout-id")
    parser.add_argument("--timestep", type=int)
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Replay one rollout; requires --task-id and --rollout-id",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Reject replay when the final step is not done",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        trajectory = TrajectoryReplay.from_jsonl(args.path)
        if args.replay:
            if not args.task_id or not args.rollout_id:
                raise TrajectoryValidationError(
                    "--replay requires --task-id and --rollout-id"
                )
            result = trajectory.replay(
                task_id=args.task_id,
                rollout_id=args.rollout_id,
                require_complete=args.require_complete,
            )
            output = result.model_dump(mode="json")
        else:
            output = [
                step.model_dump(mode="json")
                for step in trajectory.query(
                    task_id=args.task_id,
                    rollout_id=args.rollout_id,
                    timestep=args.timestep,
                )
            ]
        print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except TrajectoryValidationError as exc:
        print(f"Trajectory error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
