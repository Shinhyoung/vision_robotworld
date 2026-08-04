#!/usr/bin/env python3
"""Measure inspection and pose accuracy against mock ground truth.

    python3 tools/evaluate.py --part all
    python3 tools/evaluate.py --part guide_block --pose icp --frames 30

Reports, per part:

* inspection -- false rejects (good parts called NG) and misses (defects called
  OK) at the configured threshold, plus the score margin between the two
  populations. The margin matters more than the counts: a run with zero errors
  but no margin will start failing on the first real capture.
* pose -- translation error in mm and rotation error in degrees against the
  rendered ground truth, reported both raw and reduced by the part's symmetry
  group (see roboworld_core.symmetry for why both are needed).

Numbers quoted in the config comments and in docs/architecture.md come from
this tool; re-run it after any threshold or backend change.
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
from roboworld_core.inspection import build_backend as build_inspection  # noqa: E402
from roboworld_core.mock_data import DEFECT_KINDS, MockStation, parts_from_config  # noqa: E402
from roboworld_core.pose import build_backend as build_pose  # noqa: E402
from roboworld_core.symmetry import group_for_part, pose_error  # noqa: E402
from roboworld_core.types import CameraIntrinsics  # noqa: E402


def evaluate_inspection(cfg, station, part_id: str, args) -> dict:
    backend = build_inspection(cfg, part_id, backend=args.inspection)
    train = [
        station.sample_frame(part_id, seed=args.seed + i, sequence=i)
        for i in range(args.train_frames)
    ]
    started = time.perf_counter()
    backend.fit(train)
    fit_seconds = time.perf_counter() - started

    good = [
        station.sample_frame(part_id, seed=args.seed + 50_000 + i, sequence=i)
        for i in range(args.frames)
    ]
    defective = [
        station.sample_frame(
            part_id,
            defect=DEFECT_KINDS[i % len(DEFECT_KINDS)],
            seed=args.seed + 70_000 + i,
            sequence=i,
        )
        for i in range(args.frames)
    ]

    good_results = [backend.infer(frame) for frame in good]
    defective_results = [backend.infer(frame) for frame in defective]
    good_scores = np.array([r.anomaly_score for r in good_results])
    defective_scores = np.array([r.anomaly_score for r in defective_results])
    threshold = backend.settings.threshold

    per_kind: dict[str, dict] = {}
    for index, result in enumerate(defective_results):
        kind = DEFECT_KINDS[index % len(DEFECT_KINDS)]
        entry = per_kind.setdefault(kind, {"n": 0, "missed": 0, "scores": []})
        entry["n"] += 1
        entry["scores"].append(float(result.anomaly_score))
        if result.anomaly_score <= threshold:
            entry["missed"] += 1

    return {
        "backend": backend.name,
        "threshold": float(threshold),
        "fit_seconds": round(fit_seconds, 2),
        "train_frames": len(train),
        "good_frames": len(good),
        "defective_frames": len(defective),
        "false_rejects": int((good_scores > threshold).sum()),
        "missed_defects": int((defective_scores <= threshold).sum()),
        "good_score_max": float(good_scores.max()),
        "defect_score_min": float(defective_scores.min()),
        # Negative margin means the two populations overlap: the threshold is
        # sitting inside the ambiguous band and will misclassify on new data.
        "margin": float(defective_scores.min() - good_scores.max()),
        "inference_ms_median": float(
            np.median([r.inference_time_ms for r in good_results + defective_results])
        ),
        "per_defect_kind": {
            kind: {
                "n": entry["n"],
                "missed": entry["missed"],
                "score_min": round(min(entry["scores"]), 4),
            }
            for kind, entry in sorted(per_kind.items())
        },
    }


def evaluate_pose(cfg, station, part_id: str, args) -> dict:
    backend = build_pose(cfg, part_id, backend=args.pose)
    group = group_for_part(cfg, part_id)

    translations, rotations_raw, rotations_symmetry, fitnesses, times = [], [], [], [], []
    invalid = 0
    #: Frames where the estimate is geometrically right but landed on a
    #: symmetric alternative: large raw error, small symmetry-reduced error.
    #: This is the number the robot department needs -- if their grasp is not
    #: symmetry-invariant, this is how often it would be handed a flipped part.
    flips = 0
    for index in range(args.frames):
        frame = station.sample_frame(
            part_id, seed=args.seed + 30_000 + index, sequence=index
        )
        estimate = backend.run(frame)
        if not estimate.valid:
            invalid += 1
            continue
        error = pose_error(estimate.pose, frame.gt_pose, group)
        translations.append(error["translation_mm"])
        rotations_raw.append(error["rotation_deg"])
        rotations_symmetry.append(error["rotation_deg_symmetry_reduced"])
        fitnesses.append(estimate.fitness)
        times.append(estimate.inference_time_ms)
        if error["rotation_deg"] > 90.0 and error["rotation_deg_symmetry_reduced"] < 10.0:
            flips += 1

    if not translations:
        return {"backend": backend.name, "frames": args.frames, "invalid": invalid,
                "error": "no valid pose produced"}

    return {
        "backend": backend.name,
        "symmetry_group_size": len(group),
        "frames": args.frames,
        "invalid": invalid,
        "translation_mm": {
            "median": round(float(np.median(translations)), 3),
            "p90": round(float(np.percentile(translations, 90)), 3),
            "max": round(float(np.max(translations)), 3),
        },
        "rotation_deg_symmetry_reduced": {
            "median": round(float(np.median(rotations_symmetry)), 3),
            "p90": round(float(np.percentile(rotations_symmetry, 90)), 3),
            "max": round(float(np.max(rotations_symmetry)), 3),
        },
        # A large raw error next to a small reduced one means the estimate landed
        # on a symmetric alternative -- correct geometry, different labelling.
        "rotation_deg_raw_median": round(float(np.median(rotations_raw)), 3),
        "symmetry_flips": flips,
        "symmetry_flip_rate": round(flips / max(1, len(translations)), 4),
        "fitness_median": round(float(np.median(fitnesses)), 3),
        "inference_ms_median": round(float(np.median(times)), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", default="all")
    parser.add_argument("--frames", type=int, default=25, help="evaluation frames per class")
    parser.add_argument("--train-frames", type=int, default=40)
    parser.add_argument("--inspection", default="statistical",
                        choices=("statistical", "efficientad", "stub", "none"))
    parser.add_argument("--pose", default="icp", choices=("icp", "foundationpose", "stub", "none"))
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--json", default=None)
    # Gates. A missed defect ships a bad part to the customer and is never
    # acceptable; a false reject costs one part and is a tunable trade-off, so
    # the two get different tolerances rather than a single pass/fail.
    parser.add_argument("--max-miss-rate", type=float, default=0.0,
                        help="allowed fraction of defective frames called OK")
    parser.add_argument("--max-false-reject-rate", type=float, default=0.05,
                        help="allowed fraction of good frames called NG")
    parser.add_argument("--max-translation-mm", type=float, default=5.0)
    parser.add_argument("--max-rotation-deg", type=float, default=5.0)
    args = parser.parse_args()

    cfg = load_config()
    intrinsics = CameraIntrinsics.from_config(cfg.section("camera"))
    station = MockStation(parts_from_config(cfg), intrinsics)

    catalogue = cfg.get("parts")
    known = sorted(catalogue)
    if args.part == "all":
        # "all" means the delivered catalogue. Parts registered from camera
        # captures have their own, unknown tolerances; holding them to the
        # shipped numbers would fail the gate for reasons unrelated to what it
        # protects. They stay evaluable by name.
        parts = [p for p in known if not catalogue[p].get("locally_registered")]
        skipped = [p for p in known if catalogue[p].get("locally_registered")]
        if skipped:
            print(f"note: skipping locally registered part(s): {', '.join(skipped)}\n"
                  f"      evaluate them explicitly, e.g. --part {skipped[0]}")
    else:
        parts = [args.part]
    for part_id in parts:
        if part_id not in known:
            print(f"error: unknown part '{part_id}'; known: {known}", file=sys.stderr)
            return 2

    report: dict[str, dict] = {}
    failures: list[str] = []

    for part_id in parts:
        print(f"\n=== {part_id} " + "=" * (62 - len(part_id)), flush=True)
        entry: dict = {}

        if args.inspection != "none":
            inspection = evaluate_inspection(cfg, station, part_id, args)
            entry["inspection"] = inspection
            print(
                f"inspection[{inspection['backend']}] thr={inspection['threshold']:.3f}  "
                f"false_rejects={inspection['false_rejects']}/{inspection['good_frames']}  "
                f"missed={inspection['missed_defects']}/{inspection['defective_frames']}  "
                f"margin={inspection['margin']:+.3f}  "
                f"{inspection['inference_ms_median']:.0f} ms"
            )
            for kind, stats in inspection["per_defect_kind"].items():
                print(f"    {kind:<8} missed {stats['missed']}/{stats['n']} "
                      f"(min score {stats['score_min']:.3f})")
            miss_rate = inspection["missed_defects"] / max(1, inspection["defective_frames"])
            false_reject_rate = inspection["false_rejects"] / max(1, inspection["good_frames"])
            inspection["miss_rate"] = round(miss_rate, 4)
            inspection["false_reject_rate"] = round(false_reject_rate, 4)
            if miss_rate > args.max_miss_rate:
                failures.append(
                    f"{part_id}: miss rate {miss_rate:.1%} exceeds "
                    f"{args.max_miss_rate:.1%} ({inspection['missed_defects']} defects called OK)"
                )
            if false_reject_rate > args.max_false_reject_rate:
                failures.append(
                    f"{part_id}: false reject rate {false_reject_rate:.1%} exceeds "
                    f"{args.max_false_reject_rate:.1%}"
                )

        if args.pose != "none":
            pose = evaluate_pose(cfg, station, part_id, args)
            entry["pose"] = pose
            if "error" in pose:
                print(f"pose[{pose['backend']}] FAILED: {pose['error']}")
                failures.append(f"{part_id}: {pose['error']}")
            else:
                print(
                    f"pose[{pose['backend']}] invalid={pose['invalid']}/{pose['frames']}  "
                    f"t_err med={pose['translation_mm']['median']:.2f} "
                    f"max={pose['translation_mm']['max']:.2f} mm  "
                    f"rot med={pose['rotation_deg_symmetry_reduced']['median']:.2f} "
                    f"max={pose['rotation_deg_symmetry_reduced']['max']:.2f} deg  "
                    f"fit={pose['fitness_median']:.2f}  "
                    f"{pose['inference_ms_median']:.0f} ms"
                )
                print(
                    f"    symmetry flips (right geometry, alternative orientation): "
                    f"{pose['symmetry_flips']}/{pose['frames'] - pose['invalid']} "
                    f"({pose['symmetry_flip_rate']:.0%})  |G|={pose['symmetry_group_size']}"
                )
                if pose["invalid"]:
                    failures.append(f"{part_id}: {pose['invalid']} pose(s) rejected")
                if pose["translation_mm"]["max"] > args.max_translation_mm:
                    failures.append(
                        f"{part_id}: max translation error "
                        f"{pose['translation_mm']['max']:.2f} mm exceeds "
                        f"{args.max_translation_mm:.2f} mm"
                    )
                if pose["rotation_deg_symmetry_reduced"]["max"] > args.max_rotation_deg:
                    failures.append(
                        f"{part_id}: max rotation error "
                        f"{pose['rotation_deg_symmetry_reduced']['max']:.2f} deg exceeds "
                        f"{args.max_rotation_deg:.2f} deg"
                    )

        report[part_id] = entry

    if args.json:
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {output}")

    if failures:
        print("\nFAILURES:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("\nall parts within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
