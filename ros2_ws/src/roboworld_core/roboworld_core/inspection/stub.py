"""Deterministic stub inspection backend.

This is the Team Lead's stub node payload (claude.md section 3.1): it lets the
whole pipeline be wired up and dry-run before EfficientAD exists. It needs no
training, no GPU and no model file.

Two modes:

* ``use_ground_truth=True`` (default) -- follow ``Frame.gt_is_good``, which mock
  frames carry. Used by the E2E dry-run so expected OK/NG paths are exercised.
* ``use_ground_truth=False`` -- always report OK with a fixed score. Used when
  replaying a rosbag that has no labels.
"""

from __future__ import annotations

import numpy as np

from ..segmentation import segment_part
from ..types import Frame
from .base import InspectionBackend, InspectionSettings


class StubBackend(InspectionBackend):
    """Fixed-response inspector for dry-runs and interface tests."""

    name = "stub"

    def __init__(
        self,
        settings: InspectionSettings,
        use_ground_truth: bool = True,
        ok_score: float = 0.10,
        ng_score: float = 0.90,
        segmentation_kwargs: dict | None = None,
    ) -> None:
        super().__init__(settings)
        self.use_ground_truth = bool(use_ground_truth)
        self.ok_score = float(ok_score)
        self.ng_score = float(ng_score)
        self.segmentation_kwargs = dict(segmentation_kwargs or {})

    def score_map(self, frame: Frame) -> tuple[np.ndarray, np.ndarray]:
        roi = segment_part(frame, **self.segmentation_kwargs).mask
        height, width = frame.depth.shape

        is_good = True if frame.gt_is_good is None else bool(frame.gt_is_good)
        if not self.use_ground_truth:
            is_good = True
        score = self.ok_score if is_good else self.ng_score

        anomaly_map = np.zeros((height, width), dtype=np.float32)
        if not roi.any():
            return anomaly_map, roi

        if is_good:
            anomaly_map[roi] = score
            return anomaly_map, roi

        # Paint a blob large enough to clear min_defect_area_px so the shared
        # decision logic in InspectionBackend.decide is genuinely exercised.
        rows, cols = np.nonzero(roi)
        center_y = int(rows.mean())
        center_x = int(cols.mean())
        radius = max(6, int(np.sqrt(self.settings.min_defect_area_px)) + 4)
        yy, xx = np.mgrid[0:height, 0:width]
        blob = roi & (((yy - center_y) ** 2 + (xx - center_x) ** 2) < radius ** 2)
        anomaly_map[roi] = self.ok_score
        anomaly_map[blob] = score
        return anomaly_map, roi
