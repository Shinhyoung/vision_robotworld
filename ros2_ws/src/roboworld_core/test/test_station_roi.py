"""Only the part standing at the station may be selected.

Size identification (test_identification.py) answers "is this the right part".
It cannot answer "is this the one that stopped in front of the camera": a second
identical block further down the belt matches the registered dimensions just as
well. Measured on a two-part scene at 200 mm spacing, segmentation selected the
*following* part -- centre y = +195 mm -- and the pipeline would have published
its pose for the robot to pick.
"""

from __future__ import annotations

import numpy as np
import pytest

from roboworld_core.config import Config, ConfigError
from roboworld_core.geometry import euler_to_matrix, make_transform
from roboworld_core.inspection import _segmentation_kwargs as inspection_kwargs
from roboworld_core.mesh_io import load_ply
from roboworld_core.paths import resolve_path
from roboworld_core.pose import _segmentation_kwargs as pose_kwargs
from roboworld_core.pose import build_backend
from roboworld_core.render import Renderer, RenderItem, plane_item
from roboworld_core.segmentation import (
    StationRoi,
    segment_from_config,
    station_roi_from_config,
)
from roboworld_core.types import Frame

PART_ID = "guide_block"
#: Part centre at the stop position: on the belt, half a block thickness up.
_STATION_POSITION = np.array([0.0, 0.0, 0.5725])
#: Spacing that produced the measured mis-selection. Below the 500 mm minimum
#: agreed with the conveyor side, so the ROI is what has to catch it.
_SPACING_M = 0.20


def build_belt(cfg, intrinsics, positions):
    """Render identical blocks at ``positions`` on the belt.

    Returns the frame plus the render label image, so a test can say *which*
    block was selected instead of only that something was.
    """
    mesh = load_ply(resolve_path(cfg.get(f"parts.{PART_ID}.mesh")))
    rotation = euler_to_matrix(0.0, 0.0, 0.05)
    items = [
        plane_item(make_transform(np.eye(3), np.array([0.0, 0.0, 0.60])),
                   (1.4, 0.7), (58, 62, 70), 0),
    ]
    for index, position in enumerate(positions, start=1):
        items.append(RenderItem(
            mesh.vertices, mesh.faces, np.array([196, 188, 170], np.uint8),
            make_transform(rotation, np.asarray(position, dtype=np.float64)), index,
        ))
    rendered = Renderer(intrinsics).render(items)
    frame = Frame(rendered.color, rendered.depth, intrinsics, 0.0, part_id=PART_ID)
    return frame, rendered.mask


#: Box used by the scenario tests below. Deliberately NOT the shipped one: those
#: numbers are measured off a physical rig and move whenever the camera moves,
#: and a mechanism test that breaks every time the rig is re-surveyed teaches
#: nothing. ``test_shipped_config_*`` pins the shipped values instead.
TEST_ROI = {"center_m": [0.0, 0.0, 0.60], "half_extents_m": [0.15, 0.15, 0.12]}


def with_roi(cfg, enabled: bool, **overrides):
    roi = {"enabled": enabled, **TEST_ROI, **overrides}
    return cfg.merged_with({"pose": {"segmentation": {"station_roi": roi}}})


def selected_label(segmentation, labels: np.ndarray, count: int) -> int:
    """Which rendered block the selected mask actually covers."""
    coverage = [(labels == label)[segmentation.mask].mean() for label in range(1, count + 1)]
    return int(np.argmax(coverage)) + 1


# --- the box itself ------------------------------------------------------
def test_offset_is_zero_inside_and_grows_outside():
    roi = StationRoi([0.0, 0.0, 0.60], [0.15, 0.15, 0.12])

    assert roi.contains([0.0, 0.0, 0.60])
    assert roi.contains([0.149, -0.149, 0.71])  # just inside a corner
    assert roi.offset_m([0.10, 0.05, 0.58]) == 0.0

    # 50 mm past the +y face, and nothing else out of bounds.
    assert roi.offset_m([0.0, 0.20, 0.60]) == pytest.approx(0.05)
    assert not roi.contains([0.0, 0.20, 0.60])
    # Outside on two axes -> the diagonal distance to the edge, not the max.
    assert roi.offset_m([0.18, 0.19, 0.60]) == pytest.approx(0.05, abs=1e-9)


def test_rejects_a_malformed_box():
    with pytest.raises(ValueError, match="3 elements"):
        StationRoi([0.0, 0.0], [0.15, 0.15, 0.12])
    with pytest.raises(ValueError, match="positive"):
        StationRoi([0.0, 0.0, 0.60], [0.15, 0.0, 0.12])


# --- config plumbing -----------------------------------------------------
def test_config_roundtrip_and_disable(cfg):
    roi = station_roi_from_config(cfg)
    assert roi is not None, "the shipped config should enable the station ROI"
    assert station_roi_from_config(with_roi(cfg, False)) is None


def test_shipped_config_contains_the_measured_stop_position(cfg):
    """The shipped box must hold where the part was actually measured.

    Measured on the roller conveyor with a D455, 10 frames, centre stable to
    [4, 1, 2] mm. This is the assertion that catches "someone tuned the box and
    the part no longer fits" -- the exact failure that made detection 0/10.
    """
    roi = station_roi_from_config(cfg)
    measured_center_m = np.array([0.016, -0.009, 0.626])

    assert roi.contains(measured_center_m), (
        f"the part sits {roi.offset_m(measured_center_m) * 1000:.0f} mm outside "
        f"the shipped box (centre {roi.center_m}, half {roi.half_extents_m})"
    )
    # Margin, not a hairline pass: placement scatter must not push it out.
    slack = roi.half_extents_m - np.abs(measured_center_m - roi.center_m)
    assert slack.min() > 0.05, f"only {slack.min() * 1000:.0f} mm of margin left"


def test_shipped_box_still_rejects_a_neighbour_at_the_agreed_spacing(cfg):
    """A follower at the agreed 500 mm minimum spacing must stay excluded.

    This is the whole point of the gate, and it bounds how wide the box may be
    made: a follower at spacing S is only excluded while half_extent < S.
    """
    roi = station_roi_from_config(cfg)
    spacing_m = 0.500

    assert roi.half_extents_m.max() < spacing_m, (
        "the box is wider than the agreed part spacing -- it can no longer "
        "separate the station part from its neighbour"
    )
    for axis in range(3):
        neighbour = roi.center_m.copy()
        neighbour[axis] += spacing_m
        assert not roi.contains(neighbour), f"neighbour on axis {axis} is inside"


def test_enabled_but_incomplete_config_fails_loudly():
    """A half-written ROI must not silently degrade to "no ROI".

    Built from scratch rather than merged onto the shipped config, because a
    deep merge cannot remove the key that is meant to be missing.
    """
    broken = Config({"pose": {"segmentation": {
        "station_roi": {"enabled": True, "center_m": [0.0, 0.0, 0.60]},
    }}})
    with pytest.raises(ConfigError, match="half_extents_m"):
        station_roi_from_config(broken)


def test_both_backend_factories_receive_the_roi(cfg):
    """The pitfall from the identification work, repeated for the ROI.

    Wiring only ``segment_from_config`` leaves the backends untouched, because
    they call ``segment_part`` directly -- and the backends are what the
    pipeline runs.
    """
    for kwargs in (pose_kwargs(cfg, PART_ID), inspection_kwargs(cfg, PART_ID)):
        assert isinstance(kwargs["station_roi"], StationRoi)
    for kwargs in (pose_kwargs(with_roi(cfg, False), PART_ID),
                   inspection_kwargs(with_roi(cfg, False), PART_ID)):
        assert kwargs["station_roi"] is None


# --- selection -----------------------------------------------------------
def test_neighbour_at_200_mm_is_selected_without_the_roi(cfg, intrinsics):
    """The regression this exists to prevent, with the ROI switched off."""
    frame, labels = build_belt(
        cfg, intrinsics, [_STATION_POSITION, _STATION_POSITION + [0.0, _SPACING_M, 0.0]]
    )
    segmentation = segment_from_config(frame, with_roi(cfg, False), part_id=PART_ID)

    assert segmentation.ok, segmentation.reason
    assert selected_label(segmentation, labels, 2) == 2, (
        "expected the documented failure: without the ROI the follower wins"
    )


def test_roi_selects_the_part_at_the_station(cfg, intrinsics):
    frame, labels = build_belt(
        cfg, intrinsics, [_STATION_POSITION, _STATION_POSITION + [0.0, _SPACING_M, 0.0]]
    )
    segmentation = segment_from_config(frame, with_roi(cfg, True), part_id=PART_ID)

    assert segmentation.ok, segmentation.reason
    assert selected_label(segmentation, labels, 2) == 1
    assert segmentation.selected.roi_offset_m == 0.0


def test_the_neighbour_stays_visible_as_a_rejected_candidate(cfg, intrinsics):
    """Diagnosis needs "it was there and I refused it", not silence."""
    frame, _ = build_belt(
        cfg, intrinsics, [_STATION_POSITION, _STATION_POSITION + [0.0, _SPACING_M, 0.0]]
    )
    segmentation = segment_from_config(frame, with_roi(cfg, True), part_id=PART_ID)

    assert len(segmentation.candidates) == 2
    rejected = [c for c in segmentation.candidates if not c.in_roi]
    assert len(rejected) == 1
    # It is a dimensionally perfect match -- only its position disqualifies it.
    assert rejected[0].size_error < 0.05
    assert rejected[0].roi_offset_m == pytest.approx(0.045, abs=0.02)


def test_an_empty_station_is_refused_even_with_a_part_in_view(cfg, intrinsics):
    """Nothing at the stop position means "not arrived yet", not "pick that one"."""
    frame, _ = build_belt(cfg, intrinsics, [_STATION_POSITION + [0.0, 0.25, 0.0]])
    segmentation = segment_from_config(frame, with_roi(cfg, True), part_id=PART_ID)

    assert not segmentation.ok
    assert "station volume" in segmentation.reason
    assert segmentation.candidates, "the refused object should still be reported"
    # Without the ROI the same frame yields a confident, wrong selection.
    assert segment_from_config(frame, with_roi(cfg, False), part_id=PART_ID).ok


# --- what the robot department actually receives --------------------------
def test_pose_backend_publishes_the_station_part_not_the_neighbour(cfg, intrinsics):
    frame, _ = build_belt(
        cfg, intrinsics, [_STATION_POSITION, _STATION_POSITION + [0.0, _SPACING_M, 0.0]]
    )

    without = build_backend(with_roi(cfg, False), PART_ID, backend="icp").run(frame)
    with_gate = build_backend(with_roi(cfg, True), PART_ID, backend="icp").run(frame)

    assert with_gate.valid, with_gate.message
    assert np.linalg.norm(with_gate.pose.position - _STATION_POSITION) < 0.010

    # The failure being prevented: a *valid* pose 200 mm from the part the robot
    # was told to pick is worse than no pose at all.
    if without.valid:
        assert np.linalg.norm(without.pose.position - _STATION_POSITION) > 0.10


def test_pose_backend_reports_no_pose_when_the_station_is_empty(cfg, intrinsics):
    frame, _ = build_belt(cfg, intrinsics, [_STATION_POSITION + [0.0, 0.25, 0.0]])
    estimate = build_backend(with_roi(cfg, True), PART_ID, backend="icp").run(frame)

    assert not estimate.valid
    assert "station volume" in estimate.message
