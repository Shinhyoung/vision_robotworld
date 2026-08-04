#!/usr/bin/env python3
"""End-to-end dry-run of the index cycle, without ROS and without hardware.

This is the Team Lead's Definition of Done gate (claude.md section 3.1: "stub
파이프라인만으로 E2E가 dry-run 된다") and the merge gate for every feature
branch (section 5).

It exercises the real state machine from :mod:`roboworld_core.pipeline` --
trigger, capture, inspect, NG-branch, pose, publish -- against mock frames, then
validates **every** emitted PartResult against the ICD contract.

    python3 tools/e2e_dryrun.py                         # stub backends, seconds
    python3 tools/e2e_dryrun.py --inspection statistical --pose icp
    python3 tools/e2e_dryrun.py --cycles 12 --json out.json

Exit code is 0 only when every cycle satisfied the contract and the expected
OK/NG branches were both taken.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from _bootstrap import bootstrap

bootstrap()

from roboworld_core.config import load_config  # noqa: E402
from roboworld_core.contract import check_result, check_schema_alignment  # noqa: E402
from roboworld_core.inspection import build_backend as build_inspection  # noqa: E402
from roboworld_core.mock_data import DEFECT_KINDS, MockStation, parts_from_config  # noqa: E402
from roboworld_core.pipeline import build_pipeline  # noqa: E402
from roboworld_core.pose import build_backend as build_pose  # noqa: E402
from roboworld_core.symmetry import group_for_part, pose_error  # noqa: E402
from roboworld_core.types import CameraIntrinsics, PartStatus  # noqa: E402

STATUS_SYMBOL = {
    PartStatus.OK: "OK  ",
    PartStatus.NG: "NG  ",
    PartStatus.NO_POSE: "NOPO",
    PartStatus.ERROR: "ERR ",
}


def build_capture(cfg, part_id: str, defect_every: int, seed: int):
    """A capture callable that injects a defect on every Nth cycle."""
    intrinsics = CameraIntrinsics.from_config(cfg.section("camera"))
    station = MockStation(parts_from_config(cfg), intrinsics)
    truth: dict[int, object] = {}

    def capture(requested_part: str, sequence: int):
        defect = None
        if defect_every > 0 and sequence % defect_every == 0:
            defect = DEFECT_KINDS[(sequence // defect_every - 1) % len(DEFECT_KINDS)]
        frame = station.sample_frame(
            requested_part,
            defect=defect,
            seed=seed + sequence * 17,
            sequence=sequence,
            stamp=time.time(),
        )
        truth[sequence] = frame
        return frame

    return capture, truth


def fit_if_needed(backend, cfg, part_id: str, frames: int, seed: int) -> None:
    """Fit the statistical backend on freshly rendered defect-free frames."""
    if backend.is_fitted:
        return
    intrinsics = CameraIntrinsics.from_config(cfg.section("camera"))
    station = MockStation(parts_from_config(cfg), intrinsics)
    print(f"  fitting '{backend.name}' on {frames} defect-free frames...", flush=True)
    backend.fit(
        [
            station.sample_frame(part_id, defect=None, seed=seed + 500_000 + i, sequence=i)
            for i in range(frames)
        ]
    )
    print(f"  threshold calibrated to {backend.settings.threshold:.4f}", flush=True)


def run(args: argparse.Namespace) -> int:
    cfg = load_config()
    part_id = args.part

    print("=" * 78)
    print(f"E2E dry-run  part={part_id}  inspection={args.inspection}  pose={args.pose}")
    print("=" * 78)

    # --- 1. schema alignment (the contract itself) ---------------------
    problems = check_schema_alignment()
    if problems:
        print("CONTRACT SCHEMA MISMATCH:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("schema: PartResult.msg and roboworld_core.types.PartResult are aligned")

    # --- 2. build the pipeline -----------------------------------------
    inspection_backend = build_inspection(cfg, part_id, backend=args.inspection)
    fit_if_needed(inspection_backend, cfg, part_id, args.fit_frames, args.seed)
    pose_backend = build_pose(cfg, part_id, backend=args.pose)
    capture, truth = build_capture(cfg, part_id, args.defect_every, args.seed)
    pipeline = build_pipeline(cfg, inspection_backend, pose_backend, capture)

    group = group_for_part(cfg, part_id)

    # --- 3. run the cycles ---------------------------------------------
    print(
        f"\n{'seq':>4} {'stat':<5} {'score':>7} {'pose_valid':>10} "
        f"{'t_err':>8} {'r_err':>7} {'tact':>8}  note"
    )
    print("-" * 78)

    rows: list[dict] = []
    violations: list[str] = []
    label_mismatches: list[str] = []
    statuses: list[PartStatus] = []
    translation_errors: list[float] = []
    rotation_errors: list[float] = []

    for _ in range(args.cycles):
        report = pipeline.run_cycle(part_id)
        result = report.result
        statuses.append(PartStatus(result.status))

        found = check_result(result, strict=False)
        if found:
            violations.extend(f"seq {result.sequence}: {problem}" for problem in found)

        row = result.to_dict()
        row["stage_times_ms"] = {k: round(v, 2) for k, v in report.stage_times_ms.items()}
        row["budget_exceeded"] = report.budget_exceeded

        translation_text, rotation_text = "-", "-"
        frame = truth.get(result.sequence)
        if result.pose_valid and frame is not None and frame.gt_pose is not None:
            error = pose_error(result.pose, frame.gt_pose, group)
            translation_errors.append(error["translation_mm"])
            rotation_errors.append(error["rotation_deg_symmetry_reduced"])
            row["pose_error"] = {k: round(v, 4) for k, v in error.items()}
            translation_text = f"{error['translation_mm']:.2f}mm"
            rotation_text = f"{error['rotation_deg_symmetry_reduced']:.2f}d"

        # Ground-truth label check. Reported, but NOT a gate: this tool verifies
        # the contract and the plumbing, and a handful of cycles is far too
        # small a sample to judge detector accuracy. That is tools/evaluate.py's
        # job, with proper rates and tolerances. Gating here would make CI fail
        # for reasons unrelated to the interface it is supposed to protect.
        if frame is not None and frame.gt_is_good is not None:
            expected_ng = not frame.gt_is_good
            actual_ng = PartStatus(result.status) == PartStatus.NG
            row["label_correct"] = expected_ng == actual_ng
            if expected_ng != actual_ng:
                kind = "MISSED DEFECT" if expected_ng else "FALSE REJECT"
                label_mismatches.append(f"seq {result.sequence}: {kind}")

        rows.append(row)
        print(
            f"{result.sequence:>4} {STATUS_SYMBOL[PartStatus(result.status)]:<5} "
            f"{result.anomaly_score:>7.3f} {str(result.pose_valid):>10} "
            f"{translation_text:>8} {rotation_text:>7} "
            f"{result.tact_time_ms:>7.0f}ms  {result.message[:28]}"
        )

    # --- 4. summary -----------------------------------------------------
    counts = {status.name: statuses.count(status) for status in PartStatus}
    print("-" * 78)
    print(f"statuses: {counts}")
    if translation_errors:
        print(
            f"pose error vs ground truth: translation median "
            f"{np.median(translation_errors):.2f} mm (max {max(translation_errors):.2f}), "
            f"rotation median {np.median(rotation_errors):.2f} deg "
            f"(max {max(rotation_errors):.2f}, symmetry-reduced, |G|={len(group)})"
        )
    tacts = [row["tact_time_ms"] for row in rows]
    print(f"tact time: median {np.median(tacts):.0f} ms, max {max(tacts):.0f} ms")
    if label_mismatches:
        print(
            f"note: {len(label_mismatches)} OK/NG label mismatch(es) "
            f"({', '.join(label_mismatches)}). Not a gate here -- run "
            "tools/evaluate.py for detection rates against tolerances."
        )

    if args.json:
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "part_id": part_id,
                    "inspection_backend": args.inspection,
                    "pose_backend": args.pose,
                    "cycles": rows,
                    "status_counts": counts,
                    "violations": violations,
                    "label_mismatches": label_mismatches,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {output}")

    # --- 5. gates -------------------------------------------------------
    failures: list[str] = list(violations)
    if args.defect_every > 0:
        # Both branches must actually have been taken, otherwise a dry-run that
        # never produced an NG would pass while the branch is broken.
        if counts["NG"] == 0:
            failures.append("no NG cycle occurred: the defect branch was never exercised")
        if counts["OK"] == 0:
            failures.append("no OK cycle occurred: the pose branch was never exercised")
    if counts["ERROR"]:
        failures.append(f"{counts['ERROR']} cycle(s) ended in STATUS_ERROR")

    if failures:
        print("\nFAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("\nPASSED: every cycle satisfied the ICD contract")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", default="guide_block")
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--inspection", default="stub",
                        choices=("stub", "statistical", "efficientad"))
    parser.add_argument("--pose", default="stub", choices=("stub", "icp", "foundationpose"))
    parser.add_argument("--defect-every", type=int, default=3,
                        help="inject a defect on every Nth cycle (0 = never)")
    parser.add_argument("--fit-frames", type=int, default=25,
                        help="frames used to fit the statistical backend")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", default=None, help="write the full report here")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
