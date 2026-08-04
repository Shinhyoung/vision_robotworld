"""Inspection backend interface.

Every backend takes a :class:`~roboworld_core.types.Frame` and returns an
:class:`~roboworld_core.types.InspectionResult`. The OK/NG decision, the mask
extraction and the minimum-defect-area rule live here so all backends decide
identically and only the anomaly map differs.
"""

from __future__ import annotations

import abc
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..imageops import connected_components
from ..types import Frame, InspectionResult


@dataclass
class InspectionSettings:
    """Decision parameters shared by all backends (see inspection.yaml)."""

    threshold: float = 0.50
    mask_relative_threshold: float = 0.55
    min_defect_area_px: int = 12
    publish_anomaly_map: bool = True
    publish_defect_mask: bool = True
    calibration_percentile: float = 99.0
    calibration_margin: float = 0.05
    threshold_min: float = 0.10
    threshold_max: float = 0.95

    @classmethod
    def from_config(cls, cfg) -> InspectionSettings:
        section = cfg.section("inspection")
        return cls(
            threshold=float(section.get("threshold", 0.50)),
            mask_relative_threshold=float(section.get("mask_relative_threshold", 0.55)),
            min_defect_area_px=int(section.get("min_defect_area_px", 12)),
            publish_anomaly_map=bool(section.get("publish_anomaly_map", True)),
            publish_defect_mask=bool(section.get("publish_defect_mask", True)),
            calibration_percentile=float(section.get("calibration_percentile", 99.0)),
            calibration_margin=float(section.get("calibration_margin", 0.05)),
            threshold_min=float(section.get("threshold_min", 0.10)),
            threshold_max=float(section.get("threshold_max", 0.95)),
        )

    def clamp_threshold(self, value: float) -> float:
        return float(np.clip(value, self.threshold_min, self.threshold_max))


class InspectionBackend(abc.ABC):
    """Base class for surface defect detectors."""

    #: Value published in ``InspectionResult.backend``.
    name = "base"

    def __init__(self, settings: InspectionSettings) -> None:
        self.settings = settings

    # -- backend specific ------------------------------------------------
    @abc.abstractmethod
    def score_map(self, frame: Frame) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(anomaly_map, roi_mask)`` for ``frame``.

        ``anomaly_map`` is float32 with the same shape as the color image and is
        already normalised so that ``> threshold`` means NG. ``roi_mask`` marks
        the pixels the map is meaningful on (the part surface); pixels outside
        it are ignored by the decision logic.
        """

    def fit(self, frames: Iterable[Frame]) -> None:  # noqa: B027 - optional hook
        """Train on defect-free frames. No-op for backends that need no fitting."""

    def save(self, path: str | Path) -> None:  # noqa: B027 - optional hook
        """Persist the fitted model."""

    def load(self, path: str | Path) -> None:  # noqa: B027 - optional hook
        """Restore a fitted model."""

    @property
    def is_fitted(self) -> bool:
        return True

    # -- shared decision logic ------------------------------------------
    def infer(self, frame: Frame) -> InspectionResult:
        """Run the backend and apply the shared OK/NG rule."""
        started = time.perf_counter()
        anomaly_map, roi = self.score_map(frame)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        anomaly_map = np.asarray(anomaly_map, dtype=np.float32)
        roi = np.asarray(roi, dtype=bool)
        score, mask = self.decide(anomaly_map, roi)
        threshold = self.settings.threshold
        is_good = bool(score <= threshold)

        return InspectionResult(
            part_id=frame.part_id,
            sequence=frame.sequence,
            stamp=frame.stamp,
            frame_id=frame.intrinsics.frame_id,
            is_good=is_good,
            anomaly_score=float(score),
            threshold=float(threshold),
            anomaly_map=anomaly_map if self.settings.publish_anomaly_map else None,
            defect_mask=(mask.astype(np.uint8) * 255) if self.settings.publish_defect_mask
            else None,
            inference_time_ms=elapsed_ms,
            backend=self.name,
        )

    def decide(self, anomaly_map: np.ndarray, roi: np.ndarray) -> tuple[float, np.ndarray]:
        """Reduce an anomaly map to ``(score, defect_mask)``.

        A raw maximum makes a single noisy pixel able to reject a good part, so
        a candidate region must also survive the ``min_defect_area_px`` filter
        before it counts towards the score.
        """
        settings = self.settings
        if not roi.any():
            # Nothing to inspect: report the maximum score so the pipeline
            # treats a missing/occluded part as NG rather than silently OK.
            return 1.0, np.zeros(anomaly_map.shape, dtype=bool)

        masked = np.where(roi, anomaly_map, 0.0)
        peak = float(masked.max())
        if peak <= 0.0:
            return 0.0, np.zeros(anomaly_map.shape, dtype=bool)

        candidate = masked >= max(peak * settings.mask_relative_threshold,
                                  settings.threshold * settings.mask_relative_threshold)
        labels, sizes = connected_components(candidate)
        keep = np.zeros(len(sizes), dtype=bool)
        keep[1:] = sizes[1:] >= settings.min_defect_area_px
        defect_mask = keep[labels]

        if not defect_mask.any():
            # Peak exists but is speckle: report the strongest *supported*
            # evidence instead, i.e. the map value at the sub-threshold area.
            return float(np.percentile(masked[roi], 99.9)) * 0.5, defect_mask
        return float(masked[defect_mask].max()), defect_mask

    def calibrate_threshold(self, normal_scores: Sequence[float]) -> float:
        """Adjust the threshold from the score distribution of a normal-only set.

        The calibrated value can only ever *raise* the configured operating
        point, never lower it. Training data contains no defects, so it cannot
        tell us how tight the threshold may safely be -- it can only warn that
        the normal set itself already scores close to the configured threshold,
        in which case the threshold must move up or the line will false-reject.
        """
        if len(normal_scores) == 0:
            return self.settings.threshold
        percentile = float(
            np.percentile(np.asarray(normal_scores, dtype=np.float64),
                          self.settings.calibration_percentile)
        )
        candidate = percentile + self.settings.calibration_margin
        threshold = self.settings.clamp_threshold(max(self.settings.threshold, candidate))
        self.settings.threshold = threshold
        return threshold
