"""Inspection backend tests against mock frames (claude.md section 3.2 DoD)."""

from __future__ import annotations

import numpy as np
import pytest

from roboworld_core.inspection import StatisticalBackend, build_backend
from roboworld_core.inspection.base import InspectionBackend, InspectionSettings

PART_ID = "guide_block"


class ConstantBackend(InspectionBackend):
    """Emits a caller-supplied map so decide() can be tested in isolation."""

    name = "constant"

    def __init__(self, anomaly_map, roi, settings=None):
        super().__init__(settings or InspectionSettings(threshold=0.5, min_defect_area_px=12))
        self._map = anomaly_map
        self._roi = roi

    def score_map(self, frame):
        return self._map, self._roi


# --- shared decision logic ---------------------------------------------
def test_speckle_below_min_area_does_not_trigger_ng():
    """A single hot pixel must not reject a good part."""
    roi = np.ones((50, 50), dtype=bool)
    anomaly_map = np.zeros((50, 50), dtype=np.float32)
    anomaly_map[25, 25] = 1.0  # 1 px, below min_defect_area_px

    backend = ConstantBackend(anomaly_map, roi)
    score, mask = backend.decide(anomaly_map, roi)
    assert mask.sum() == 0
    assert score < backend.settings.threshold


def test_region_above_min_area_triggers_ng():
    roi = np.ones((50, 50), dtype=bool)
    anomaly_map = np.zeros((50, 50), dtype=np.float32)
    anomaly_map[20:30, 20:30] = 0.9  # 100 px

    backend = ConstantBackend(anomaly_map, roi)
    score, mask = backend.decide(anomaly_map, roi)
    assert mask.sum() == 100
    assert score == pytest.approx(0.9)


def test_empty_roi_is_treated_as_ng():
    """No part visible must fail safe, not silently pass."""
    roi = np.zeros((20, 20), dtype=bool)
    backend = ConstantBackend(np.zeros((20, 20), dtype=np.float32), roi)
    score, mask = backend.decide(np.zeros((20, 20), dtype=np.float32), roi)
    assert score == 1.0
    assert mask.sum() == 0


def test_anomalies_outside_the_roi_are_ignored():
    roi = np.zeros((50, 50), dtype=bool)
    roi[:25] = True
    anomaly_map = np.zeros((50, 50), dtype=np.float32)
    anomaly_map[30:45, 30:45] = 1.0  # entirely outside the ROI

    backend = ConstantBackend(anomaly_map, roi)
    score, mask = backend.decide(anomaly_map, roi)
    assert mask.sum() == 0
    assert score < backend.settings.threshold


def test_calibration_never_lowers_the_configured_threshold():
    backend = ConstantBackend(np.zeros((4, 4), np.float32), np.ones((4, 4), bool))
    backend.settings.threshold = 0.5
    assert backend.calibrate_threshold([0.01] * 20) == pytest.approx(0.5)


def test_calibration_raises_the_threshold_when_normals_score_high():
    settings = InspectionSettings(threshold=0.5, calibration_percentile=99.0,
                                  calibration_margin=0.05, threshold_max=0.95)
    backend = ConstantBackend(np.zeros((4, 4), np.float32), np.ones((4, 4), bool), settings)
    assert backend.calibrate_threshold([0.7] * 20) == pytest.approx(0.75)


def test_calibration_respects_the_clamp():
    settings = InspectionSettings(threshold=0.5, calibration_margin=0.05, threshold_max=0.8)
    backend = ConstantBackend(np.zeros((4, 4), np.float32), np.ones((4, 4), bool), settings)
    assert backend.calibrate_threshold([0.99] * 20) == pytest.approx(0.8)


# --- stub backend -------------------------------------------------------
def test_stub_follows_ground_truth(cfg, good_frame, defective_frame):
    backend = build_backend(cfg, PART_ID, backend="stub")
    assert backend.infer(good_frame).is_good is True
    assert backend.infer(defective_frame).is_good is False


def test_stub_result_fields_are_populated(cfg, good_frame):
    result = build_backend(cfg, PART_ID, backend="stub").infer(good_frame)
    assert result.part_id == PART_ID
    assert result.sequence == good_frame.sequence
    assert result.frame_id == good_frame.intrinsics.frame_id
    assert 0.0 <= result.anomaly_score <= 1.0
    assert result.anomaly_map.shape == good_frame.depth.shape
    assert result.defect_mask.dtype == np.uint8
    assert result.inference_time_ms >= 0.0


# --- statistical backend ------------------------------------------------
@pytest.fixture(scope="module")
def fitted_backend(cfg, station):
    backend = build_backend(cfg, PART_ID, backend="statistical")
    backend.fit(
        [station.sample_frame(PART_ID, seed=900 + i, sequence=i) for i in range(12)]
    )
    return backend


def test_statistical_requires_fitting(good_frame):
    """Constructed directly, not via the factory.

    ``build_backend`` auto-loads a saved model when one exists, so going through
    it would make this test pass or skip depending on whether the developer had
    run tools/train_inspection.py -- the test must not depend on data/ state.
    """
    backend = StatisticalBackend(InspectionSettings())
    assert not backend.is_fitted
    with pytest.raises(RuntimeError, match="before fit"):
        backend.infer(good_frame)


def test_statistical_scores_defects_above_good_parts(fitted_backend, station):
    good = station.sample_frame(PART_ID, seed=4321, sequence=1)
    defective = station.sample_frame(PART_ID, defect="chip", seed=4321, sequence=2)
    assert (
        fitted_backend.infer(defective).anomaly_score
        > fitted_backend.infer(good).anomaly_score
    )


def test_statistical_marks_the_defect_location(fitted_backend, station):
    """The mask must land on the defect, not merely somewhere on the part."""
    defective = station.sample_frame(PART_ID, defect="chip", seed=77, sequence=1)
    clean = station.render_frame(PART_ID, defective.gt_pose, defect=None, seed=78)
    changed = np.abs(
        defective.color.astype(int) - clean.color.astype(int)
    ).sum(axis=2) > 60

    result = fitted_backend.infer(defective)
    assert result.is_good is False
    mask = result.defect_mask > 0
    assert mask.any()
    # Allow generous slack: patch features respond around the defect too.
    from roboworld_core.imageops import binary_dilate

    assert (mask & binary_dilate(changed, 12)).sum() / mask.sum() > 0.5


def test_statistical_score_is_normalized(fitted_backend, station):
    for seed in (1, 2, 3):
        for defect in (None, "scratch"):
            result = fitted_backend.infer(
                station.sample_frame(PART_ID, defect=defect, seed=seed, sequence=seed)
            )
            assert 0.0 <= result.anomaly_score <= 1.0


def test_statistical_model_roundtrips(fitted_backend, cfg, station, tmp_path):
    path = tmp_path / "model.npz"
    fitted_backend.save(path)

    restored = build_backend(cfg, PART_ID, backend="statistical")
    restored.load(path)
    frame = station.sample_frame(PART_ID, defect="dent", seed=555, sequence=1)
    assert restored.infer(frame).anomaly_score == pytest.approx(
        fitted_backend.infer(frame).anomaly_score, abs=1e-6
    )


def test_load_rejects_mismatched_patch_geometry(fitted_backend, tmp_path):
    """A model trained at one patch geometry must not load into another."""
    path = tmp_path / "model.npz"
    fitted_backend.save(path)

    other = StatisticalBackend(InspectionSettings(), patch_size=32, stride=8)
    with pytest.raises(ValueError, match="patch/stride"):
        other.load(path)


def test_factory_surfaces_a_stale_model(fitted_backend, cfg, tmp_path):
    """The same mismatch must fail loudly at node startup, not silently.

    Changing patch_size in the config after training leaves a stale model file;
    the node has to refuse to start rather than infer with mismatched geometry.
    """
    model_path = tmp_path / "statistical_stale.npz"
    fitted_backend.save(model_path)
    stale_cfg = cfg.merged_with(
        {"inspection": {"statistical": {"patch_size": 32, "model_path": str(model_path)}}}
    )
    with pytest.raises(ValueError, match="patch/stride"):
        build_backend(stale_cfg, PART_ID, backend="statistical")


def test_unknown_backend_name_raises(cfg):
    with pytest.raises(ValueError, match="unknown inspection backend"):
        build_backend(cfg, PART_ID, backend="does_not_exist")
