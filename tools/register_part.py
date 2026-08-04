#!/usr/bin/env python3
"""Register a captured part: fit its detector and write its config entry.

    python3 tools/register_part.py --part my_part
    python3 tools/register_part.py --part my_part --evaluate

Takes the frames from ``tools/capture_part.py``, fits the inspection model, and
appends the part to ``config/parts_local.yaml`` so the rest of the system can
use it. **No CAD is required** -- inspection and segmentation never touch part
geometry.

What this does and does not enable
----------------------------------
==================  ==========================================================
불량 검사             ✅ 동작. 정상품 사진만 있으면 됨
분할 (ROI)           ✅ 동작. depth 평면 제거 기반이라 형상 무관
6D 포즈              ❌ 형상 필요. 메시를 확보한 뒤 parts_local.yaml의
                       ``mesh:`` 를 채우면 활성화됨
목업 렌더링           ❌ CAD가 있어야 가능
==================  ==========================================================

So a camera-registered part flows through the pipeline as far as the OK/NG
decision, and reports ``STATUS_NO_POSE`` beyond it -- which is the contract's
existing, documented behaviour for "good part, no usable pose" (ICD section 3.1).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from _bootstrap import bootstrap

bootstrap()

from roboworld_core import paths  # noqa: E402
from roboworld_core.config import load_config  # noqa: E402
from roboworld_core.geometry import Pose  # noqa: E402
from roboworld_core.inspection import build_backend, model_path_for  # noqa: E402
from roboworld_core.types import CameraIntrinsics, Frame  # noqa: E402

#: Below this the fitted Gaussian is not meaningfully constrained.
_MIN_TRAIN_FRAMES = 10


def load_capture(path: Path, part_id: str) -> Frame:
    """Load a capture written by tools/capture_part.py."""
    with np.load(path, allow_pickle=False) as data:
        fx, fy, cx, cy, width, height = data["intrinsics"]
        intrinsics = CameraIntrinsics(
            width=int(width), height=int(height),
            fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
        )
        return Frame(
            color=data["color"],
            depth=data["depth"],
            intrinsics=intrinsics,
            stamp=float(data["stamp"]),
            sequence=int(data["sequence"]),
            part_id=part_id,
            gt_is_good=bool(data["is_good"]),
            gt_pose=Pose(data["gt_position"], data["gt_orientation"], intrinsics.frame_id),
        )


def load_split(part_id: str, split: str) -> list[Frame]:
    directory = paths.data_dir() / "captures" / part_id / split
    files = sorted(directory.glob("frame_*.npz")) if directory.is_dir() else []
    return [load_capture(path, part_id) for path in files]


def write_config_entry(cfg, part_id: str, frames: int, mesh: str | None) -> Path | None:
    """Append/update ``part_id`` in config/parts_local.yaml.

    Written as a separate file rather than edited into parts.yaml: that file is
    hand-curated with comments a YAML round-trip would destroy.

    Returns ``None`` when nothing needed writing.

    Re-registering an **existing** part (e.g. retraining guide_block on real
    captures instead of mock frames) must not touch its config. Writing an entry
    here would shadow parts.yaml with ``mesh: ""`` and silently disable 6D pose
    for a part that has perfectly good CAD -- the config would then disagree with
    the CAD sitting right next to it.
    """
    existing = cfg.get(f"parts.{part_id}", None)
    if existing is not None and mesh is None:
        return None

    config_path = paths.config_dir() / "parts_local.yaml"

    document: dict = {}
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            document = yaml.safe_load(handle) or {}

    node = document.setdefault("/**", {}).setdefault("ros__parameters", {})
    parts = node.setdefault("parts", {})
    entry = parts.setdefault(part_id, {})
    entry["description"] = (
        f"Registered from {frames} camera captures"
        + ("" if mesh else " (no CAD)")
    )
    # Marks the part as not belonging to the delivered catalogue. The accuracy
    # gate (tools/evaluate.py) is calibrated for the shipped parts' tolerances
    # and would otherwise fail a locally registered part for reasons that say
    # nothing about the shipped ones.
    entry["locally_registered"] = True
    entry.setdefault("color", [190, 185, 170])
    # `mesh` is intentionally present but empty when there is no CAD: an explicit
    # empty value documents "we know this is missing" rather than looking like an
    # accidental omission, and roboworld_core.pose.has_cad() reads it as absent.
    entry["mesh"] = mesh or ""
    entry["mesh_units"] = "m"
    entry.setdefault("symmetry", [])

    header = (
        "# Parts registered from camera captures (tools/register_part.py).\n"
        "# Generated -- safe to edit by hand, but it may be rewritten.\n"
        "# Merged on top of parts.yaml by roboworld_core.config.load_config.\n"
        "#\n"
        "# `mesh: \"\"` means no CAD: inspection works, 6D pose does not.\n"
        "# Fill in a mesh path to enable pose for that part.\n"
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        handle.write(header)
        yaml.safe_dump(document, handle, sort_keys=False, allow_unicode=True)
    return config_path


def evaluate(backend, normal: list[Frame], defective: list[Frame]) -> None:
    """Report the fitted model's separation on held-out captures."""
    threshold = backend.settings.threshold
    normal_scores = np.array([backend.infer(f).anomaly_score for f in normal])
    print(f"\n  normal   n={len(normal)}  score median={np.median(normal_scores):.3f} "
          f"max={normal_scores.max():.3f}  "
          f"false rejects={int((normal_scores > threshold).sum())}")

    if not defective:
        print("  defect   (none captured -- run capture_part.py --defect to check "
              "the model actually separates real defects)")
        return

    defect_scores = np.array([backend.infer(f).anomaly_score for f in defective])
    margin = float(defect_scores.min() - normal_scores.max())
    print(f"  defect   n={len(defective)}  score median={np.median(defect_scores):.3f} "
          f"min={defect_scores.min():.3f}  "
          f"missed={int((defect_scores <= threshold).sum())}")
    print(f"  margin   {margin:+.3f}  " + (
        "(양호)" if margin > 0.05 else
        "(빠듯함 — 정상 샘플을 더 찍거나 safety_factor 조정 필요)" if margin > 0 else
        "(겹침 — 이 특징으로는 구분 불가. EfficientAD 전환 검토)"
    ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", required=True)
    parser.add_argument("--mesh", default=None,
                        help="optional mesh path, enables 6D pose for this part")
    parser.add_argument("--evaluate", action="store_true",
                        help="hold out 20%% of normal captures to score the fit")
    parser.add_argument("--force", action="store_true",
                        help="fit even with fewer than the recommended frames")
    parser.add_argument("--mesh-only", action="store_true",
                        help="register geometry for 6D pose only; skip inspection fitting")
    args = parser.parse_args()

    cfg = load_config()
    normal = load_split(args.part, "normal")
    defective = load_split(args.part, "defect")

    # Pose-only registration: geometry is enough for 6D pose, and inspection
    # needs a proper normal set that a reconstruction session does not produce
    # (its views are all the same pose, which teaches the detector nothing).
    if args.mesh_only:
        if not args.mesh:
            print("error: --mesh-only requires --mesh", file=sys.stderr)
            return 2
        config_path = write_config_entry(cfg, args.part, len(normal), args.mesh)
        target = config_path or paths.config_dir() / "parts_local.yaml"
        print(f"'{args.part}' 등록 (메시만) -> {target}")
        print(f"  mesh: {args.mesh}")
        print("\n  ※ 검사 모델은 학습하지 않았습니다. 검사를 쓰려면:")
        print(f"     python3 tools/capture_part.py  --part {args.part}")
        print(f"     python3 tools/register_part.py --part {args.part} --mesh {args.mesh}")
        print("\n포즈 확인:")
        print(f"  python3 tools/live_view.py --source realsense --part {args.part} --pose")
        return 0

    if not normal:
        directory = paths.data_dir() / "captures" / args.part / "normal"
        print(f"error: no captures at {directory}\n"
              f"  python3 tools/capture_part.py --part {args.part}", file=sys.stderr)
        return 4
    if len(normal) < _MIN_TRAIN_FRAMES and not args.force:
        print(f"error: only {len(normal)} normal frames; {_MIN_TRAIN_FRAMES} is the "
              f"minimum and 40+ is recommended. Capture more, or pass --force.",
              file=sys.stderr)
        return 5

    print(f"part '{args.part}': {len(normal)} normal, {len(defective)} defect captures")

    holdout: list[Frame] = []
    train = normal
    if args.evaluate and len(normal) >= _MIN_TRAIN_FRAMES + 5:
        split = max(1, len(normal) // 5)
        holdout, train = normal[:split], normal[split:]
        print(f"  holding out {len(holdout)} frames for scoring")

    backend = build_backend(cfg, args.part, backend="statistical")
    backend.fit(train)
    model_path = model_path_for(cfg, "statistical", args.part)
    backend.save(model_path)
    print(f"  fitted on {len(train)} frames -> {model_path}")
    print(f"  threshold {backend.settings.threshold:.3f}")

    evaluate(backend, holdout or normal, defective)

    from roboworld_core.pose import has_cad

    already_known = cfg.get(f"parts.{args.part}", None) is not None
    config_path = write_config_entry(cfg, args.part, len(normal), args.mesh)
    if config_path is None:
        print(f"\n  '{args.part}' 는 이미 등록된 부품 — 설정은 그대로 두고 "
              "검사 모델만 갱신했습니다.")
    else:
        print(f"\n  config entry -> {config_path}")

    print("\n등록 완료. 확인:")
    print(f"  python3 tools/live_view.py --source realsense --part {args.part} --inspect")

    if args.mesh or (already_known and has_cad(cfg, args.part)):
        print(f"  python3 tools/evaluate.py --part {args.part}   # 포즈 정확도 재측정")
    else:
        print("\n  ※ CAD가 없어 6D 포즈는 비활성입니다 (검사는 동작).")
        print("     메시 확보 후:  --mesh <경로> 로 다시 등록하면 포즈가 켜집니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
