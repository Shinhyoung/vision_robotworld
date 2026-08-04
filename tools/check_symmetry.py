#!/usr/bin/env python3
"""Measure how symmetric each part actually is.

    python3 tools/check_symmetry.py

The three blocks have a 55 x 55 mm square cross-section, so a single-view pose
estimate can settle on a rotated-but-indistinguishable orientation. This tool
quantifies "indistinguishable": for every rotation declared in the ``symmetry``
section of parts.yaml, it reports the mean chamfer distance between the mesh and
its own rotated copy.

Reading the output:

* chamfer well **below** the ICP inlier gate (``pose.icp.max_correspondence_end_m``,
  6 mm by default) -> the rotation is not resolvable from one view; it belongs
  in the symmetry group and the robot department must be told (ICD section 6.1).
* chamfer well **above** the gate -> features break the symmetry and the pose is
  unique; the entry could be removed from parts.yaml.

The numbers quoted in parts.yaml and docs/architecture.md come from here.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from _bootstrap import bootstrap

bootstrap()

from roboworld_core.config import load_config  # noqa: E402
from roboworld_core.mesh_io import load_ply  # noqa: E402
from roboworld_core.paths import resolve_path  # noqa: E402
from roboworld_core.symmetry import rotation_about  # noqa: E402


def chamfer_mm(a: np.ndarray, b: np.ndarray) -> float:
    """Mean nearest-neighbour distance from ``a`` to ``b``, in millimeters."""
    distances_sq = (
        (a * a).sum(axis=1)[:, None] + (b * b).sum(axis=1)[None, :] - 2.0 * (a @ b.T)
    )
    return float(np.sqrt(np.maximum(distances_sq.min(axis=1), 0.0)).mean() * 1000.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", default="all")
    parser.add_argument("--samples", type=int, default=1500)
    args = parser.parse_args()

    cfg = load_config()
    gate_mm = float(cfg.get("pose.icp.max_correspondence_end_m", 0.006)) * 1000.0
    parts = cfg.get("parts")
    selected = sorted(parts) if args.part == "all" else [args.part]

    for part_id in selected:
        if part_id not in parts:
            print(f"error: unknown part '{part_id}'", file=sys.stderr)
            return 2

    print(f"ICP inlier gate: {gate_mm:.1f} mm "
          "(chamfer below this = not resolvable from a single view)\n")

    for part_id in selected:
        entry = parts[part_id]
        mesh = load_ply(resolve_path(entry["mesh"]), units=entry.get("mesh_units", "m"))
        points = mesh.sample_surface(args.samples, seed=0)

        print(f"{part_id}  extents = {np.round(mesh.extents * 1000.0, 1)} mm")
        for generator in entry.get("symmetry", []):
            axis = np.asarray(generator["axis"], dtype=np.float64)
            for angle in generator.get("angles_deg", []):
                if float(angle) % 360.0 == 0.0:
                    continue
                rotation = rotation_about(axis, np.radians(float(angle)))
                distance = chamfer_mm(points @ rotation.T, points)
                verdict = "ambiguous" if distance < gate_mm else "resolvable"
                print(
                    f"    axis {np.array2string(axis, precision=0)} "
                    f"{float(angle):5.1f} deg -> chamfer {distance:6.2f} mm  [{verdict}]"
                )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
