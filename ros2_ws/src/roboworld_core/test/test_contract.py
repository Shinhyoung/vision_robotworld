"""Contract tests for the robot-department interface (claude.md section 4).

These are the tests that must never be weakened without an ICD revision.
"""

from __future__ import annotations

import numpy as np
import pytest

from roboworld_core.contract import (
    ContractViolation,
    check_result,
    check_schema_alignment,
    interfaces_dir,
    parse_msg,
)
from roboworld_core.geometry import Pose
from roboworld_core.types import PIPELINE_VERSION, PartResult, PartStatus


def make_result(**overrides) -> PartResult:
    defaults = {
        "sequence": 1,
        "part_id": "guide_block",
        "stamp": 1234.5,
        "frame_id": "camera_color_optical_frame",
        "status": PartStatus.OK,
        "is_good": True,
        "anomaly_score": 0.05,
        "anomaly_threshold": 0.5,
        "pose_valid": True,
        "pose": Pose(np.array([0.01, -0.02, 0.573]), np.array([0.0, 0.0, 0.0, 1.0]),
                     "camera_color_optical_frame"),
        "pose_fitness": 0.97,
        "tact_time_ms": 250.0,
        "pipeline_version": PIPELINE_VERSION,
        "message": "",
    }
    defaults.update(overrides)
    return PartResult(**defaults)


def test_msg_and_dataclass_stay_aligned():
    """PartResult.msg and roboworld_core.types.PartResult must not drift apart."""
    problems = check_schema_alignment()
    assert problems == [], "schema drift: " + "; ".join(problems)


def test_status_constants_match_the_enum():
    schema = parse_msg(interfaces_dir() / "msg" / "PartResult.msg")
    for status in PartStatus:
        assert int(schema.constants[f"STATUS_{status.name}"]) == int(status)


def test_every_interface_file_parses():
    """A malformed .msg would fail at build time; catch it in CI instead."""
    root = interfaces_dir()
    files = list((root / "msg").glob("*.msg")) + list((root / "srv").glob("*.srv"))
    assert files, "no interface definitions found"
    for path in (root / "msg").glob("*.msg"):
        schema = parse_msg(path)
        assert schema.fields, f"{path.name} declares no fields"


def test_nominal_result_passes():
    assert check_result(make_result(), strict=False) == []


def test_position_in_millimeters_is_rejected():
    """The classic unit slip: 573 instead of 0.573."""
    result = make_result(
        pose=Pose(np.array([10.0, -20.0, 573.0]), np.array([0.0, 0.0, 0.0, 1.0]), "camera")
    )
    problems = check_result(result, strict=False)
    assert any("millimeters" in problem for problem in problems)


def test_ng_must_not_carry_a_pose():
    """claude.md: an NG part reports the result only, pose is skipped."""
    result = make_result(status=PartStatus.NG, is_good=False, pose_valid=True)
    problems = check_result(result, strict=False)
    assert any("must not carry a valid pose" in problem for problem in problems)


def test_ok_requires_a_valid_pose():
    result = make_result(status=PartStatus.OK, pose_valid=False)
    problems = check_result(result, strict=False)
    assert any("pose_valid is false" in problem for problem in problems)


def test_no_pose_status_must_not_carry_a_pose():
    result = make_result(status=PartStatus.NO_POSE, pose_valid=True)
    problems = check_result(result, strict=False)
    assert any("must not carry a valid pose" in problem for problem in problems)


def test_empty_frame_id_on_valid_pose_is_rejected():
    result = make_result(pose=Pose(np.array([0.0, 0.0, 0.5]),
                                   np.array([0.0, 0.0, 0.0, 1.0]), ""))
    problems = check_result(result, strict=False)
    assert any("frame_id is empty" in problem for problem in problems)


@pytest.mark.parametrize("score", [-0.1, 1.5])
def test_anomaly_score_must_be_normalized(score):
    problems = check_result(make_result(anomaly_score=score), strict=False)
    assert any("anomaly_score" in problem for problem in problems)


def test_missing_pipeline_version_is_rejected():
    problems = check_result(make_result(pipeline_version=""), strict=False)
    assert any("pipeline_version" in problem for problem in problems)


def test_strict_mode_raises():
    with pytest.raises(ContractViolation):
        check_result(make_result(anomaly_score=9.0), strict=True)


def test_to_dict_is_json_serializable():
    import json

    payload = json.dumps(make_result().to_dict())
    restored = json.loads(payload)
    assert restored["status_name"] == "OK"
    assert restored["pose"]["frame_id"] == "camera_color_optical_frame"
    assert len(restored["pose"]["orientation"]) == 4
