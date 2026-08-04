"""CPU patch-statistics anomaly detector.

EfficientAD (anomalib) is the production backend, but it needs a GPU, a CUDA
build for Blackwell and a trained checkpoint -- none of which exist while the
team is still developing against mocks. This backend fills that gap with a
genuine, trainable detector rather than a fake one, so the Inspection agent's
pipeline (fit -> calibrate -> infer -> OK/NG) is exercised for real in CI.

Method (a position-agnostic PaDiM variant):

1. Split the image into overlapping patches on a regular grid.
2. Describe each patch with 7 illumination- and rotation-tolerant statistics.
3. Fit ONE multivariate Gaussian over all patches lying on defect-free part
   surface -- one global Gaussian, not per-position, because the part arrives at
   an arbitrary yaw and per-position statistics would be meaningless.
4. Score a patch by its Mahalanobis distance to that Gaussian.

Feature vector (7-D), all computed within a patch window:
``[mean_r, mean_g, mean_b, std_gray, mean_grad, max_grad, mean_gray - min_gray]``
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ..imageops import gaussian_blur, sobel_magnitude, to_gray
from ..segmentation import segment_part
from ..types import Frame
from .base import InspectionBackend, InspectionSettings

#: Fraction of a patch that must lie on the part before it is used.
_MIN_PATCH_COVERAGE = 0.95
#: Number of strongest patches averaged into the image-level raw score used to
#: derive the normalisation anchor.
#:
#: A single-cell maximum was measured to be too volatile: anchoring on the worst
#: cell of the training set puts the anchor on a noise spike, and defects then
#: normalise below the threshold (measured: 4 missed defects across the three
#: parts, versus 0 with the top-5 mean). The published score still comes from
#: InspectionBackend.decide's maximum -- the anchor is a training-set statistic
#: and wants to be robust, the score is a detection and wants to be sensitive.
_TOP_K = 5
#: Brightness the part's median is normalised to before feature extraction.
_REFERENCE_LEVEL = 128.0


class StatisticalBackend(InspectionBackend):
    """Patch-Mahalanobis anomaly detector implemented on numpy alone."""

    name = "statistical"

    def __init__(
        self,
        settings: InspectionSettings,
        patch_size: int = 16,
        stride: int = 8,
        blur_sigma_px: float = 2.0,
        regularization: float = 1e-3,
        norm_low_percentile: float = 50.0,
        norm_high_percentile: float = 95.0,
        safety_factor: float = 1.6,
        segmentation_kwargs: dict | None = None,
    ) -> None:
        super().__init__(settings)
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.blur_sigma_px = float(blur_sigma_px)
        self.regularization = float(regularization)
        self.norm_low_percentile = float(norm_low_percentile)
        self.norm_high_percentile = float(norm_high_percentile)
        self.safety_factor = float(safety_factor)
        self.segmentation_kwargs = dict(segmentation_kwargs or {})

        self._mean: np.ndarray | None = None
        self._inv_cov: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._raw_low: float = 0.0
        self._raw_high: float = 1.0

    # -- feature extraction ---------------------------------------------
    def _patch_features(
        self, frame: Frame, roi: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        """Return ``(features, coverage, grid_shape)`` for one frame.

        ``features`` is ``(cells, 7)``; ``coverage`` is the per-cell fraction of
        pixels inside ``roi``.
        """
        from numpy.lib.stride_tricks import sliding_window_view

        color = np.asarray(frame.color, dtype=np.float32)
        # Illumination normalisation: scale the frame so the part's own median
        # brightness is a fixed reference. Without it the features encode
        # absolute brightness, and anything that changes it globally -- ambient
        # light drifting over a shift, exposure auto-adjusting, or simply the
        # part stopping nearer the edge of the frame where the lambertian term
        # falls off -- reads as a defect. Defects survive because they are
        # *local contrast*, which the ratio preserves.
        reference = float(np.median(to_gray(color)[roi])) if roi.any() else 0.0
        if reference > 1.0:
            color = color * (_REFERENCE_LEVEL / reference)

        gray = to_gray(color)
        grad = sobel_magnitude(gaussian_blur(gray, 1.0))
        patch, stride = self.patch_size, self.stride

        channels = np.stack(
            [color[..., 0], color[..., 1], color[..., 2], gray, grad,
             roi.astype(np.float32)],
            axis=0,
        )
        # (C, cells_y, cells_x, patch, patch) -- strided views, no copy yet.
        windows = sliding_window_view(channels, (patch, patch), axis=(1, 2))
        windows = windows[:, ::stride, ::stride]
        cells_y, cells_x = windows.shape[1], windows.shape[2]
        flat = windows.reshape(6, cells_y * cells_x, patch * patch)

        mean_rgb = flat[0:3].mean(axis=2)  # (3, cells)
        gray_win = flat[3]
        grad_win = flat[4]
        coverage = flat[5].mean(axis=1)

        features = np.stack(
            [
                mean_rgb[0], mean_rgb[1], mean_rgb[2],
                gray_win.std(axis=1),
                grad_win.mean(axis=1),
                grad_win.max(axis=1),
                gray_win.mean(axis=1) - gray_win.min(axis=1),
            ],
            axis=1,
        ).astype(np.float64)
        return features, coverage, (cells_y, cells_x)

    def _roi_for(self, frame: Frame) -> np.ndarray:
        segmentation = segment_part(frame, **self.segmentation_kwargs)
        return segmentation.mask

    # -- training --------------------------------------------------------
    def fit(self, frames: Iterable[Frame]) -> None:
        """Fit the Gaussian on defect-free frames and set the normalisation."""
        collected: list[np.ndarray] = []
        cached: list[tuple[Frame, np.ndarray]] = []
        for frame in frames:
            roi = self._roi_for(frame)
            if not roi.any():
                continue
            features, coverage, _ = self._patch_features(frame, roi)
            usable = features[coverage >= _MIN_PATCH_COVERAGE]
            if len(usable):
                collected.append(usable)
                cached.append((frame, roi))

        if not collected:
            raise ValueError(
                "no usable training patches -- check that the training frames "
                "actually contain a segmentable part"
            )
        stacked = np.concatenate(collected, axis=0)
        if len(stacked) < stacked.shape[1] + 2:
            raise ValueError(
                f"need more than {stacked.shape[1] + 2} training patches, got {len(stacked)}"
            )

        # Standardise so features with large numeric ranges (gradients) do not
        # dominate the covariance, then fit in the standardised space.
        self._scale = np.maximum(stacked.std(axis=0), 1e-6)
        normalized = stacked / self._scale
        self._mean = normalized.mean(axis=0)
        covariance = np.cov(normalized, rowvar=False)
        covariance += np.eye(covariance.shape[0]) * self.regularization
        self._inv_cov = np.linalg.inv(covariance)

        # Normalisation anchor from the *image-level* raw scores of the normal
        # set: score 0.5 lands at `safety_factor * high`, where `high` is the
        # configured upper percentile of the training scores. A training set
        # never contains the worst good part the line will ever produce, so
        # anchoring 0.5 directly on the training maximum guarantees false
        # rejects on unseen-but-good parts; the factor is that headroom.
        # `_raw_low` is retained for diagnostics only.
        raws = [self._raw_score(*self._distance_map(frame, roi)) for frame, roi in cached]
        raws = np.asarray([r for r in raws if np.isfinite(r)], dtype=np.float64)
        self._raw_low = float(np.percentile(raws, self.norm_low_percentile))
        self._raw_high = float(np.percentile(raws, self.norm_high_percentile))
        if self._raw_high <= self._raw_low:
            self._raw_high = self._raw_low + 1e-6

        self.calibrate_threshold([self._normalize(r) for r in raws])

    @property
    def is_fitted(self) -> bool:
        return self._mean is not None and self._inv_cov is not None

    # -- inference -------------------------------------------------------
    def _distance_map(self, frame: Frame, roi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-cell Mahalanobis distances and the usable-cell mask."""
        if not self.is_fitted:
            raise RuntimeError("StatisticalBackend used before fit()/load()")
        features, coverage, grid_shape = self._patch_features(frame, roi)
        delta = features / self._scale - self._mean
        distances = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", delta, self._inv_cov, delta), 0.0))
        usable = coverage >= _MIN_PATCH_COVERAGE
        distances = np.where(usable, distances, 0.0)
        return distances.reshape(grid_shape), usable.reshape(grid_shape)

    @staticmethod
    def _raw_score(distance_grid: np.ndarray, usable_grid: np.ndarray) -> float:
        """Image-level raw score: mean of the strongest usable cells."""
        values = distance_grid[usable_grid]
        if values.size == 0:
            return float("nan")
        k = min(_TOP_K, values.size)
        return float(np.sort(values)[-k:].mean())

    def _normalize(self, raw: float) -> float:
        """Map a raw Mahalanobis score into the contract's [0, 1] range.

        Multiplicative, not span-based: ``score = 0.5 * raw / (k * high)``.

        An earlier version normalised by the training *spread*
        ``(high - low)``, which made the scale depend on how much pose variation
        the training set happened to cover -- adding training frames widened the
        spread and quietly pushed genuine defects below the threshold. Anchoring
        on the level instead of the spread makes the score stable as the normal
        set grows, which is the property a line needs when operators keep adding
        good samples.
        """
        anchor = max(self.safety_factor * self._raw_high, 1e-9)
        return float(np.clip(0.5 * raw / anchor, 0.0, 1.0))

    def score_map(self, frame: Frame) -> tuple[np.ndarray, np.ndarray]:
        roi = self._roi_for(frame)
        height, width = frame.depth.shape
        if not roi.any():
            return np.zeros((height, width), dtype=np.float32), roi

        distance_grid, usable_grid = self._distance_map(frame, roi)
        cell_scores = np.vectorize(self._normalize)(distance_grid).astype(np.float32)
        cell_scores = np.where(usable_grid, cell_scores, 0.0).astype(np.float32)

        # Expand the cell grid back to image resolution: every pixel takes the
        # score of the nearest cell center.
        half = self.patch_size // 2
        rows = np.clip(
            np.round((np.arange(height) - half) / self.stride), 0, cell_scores.shape[0] - 1
        ).astype(np.int64)
        cols = np.clip(
            np.round((np.arange(width) - half) / self.stride), 0, cell_scores.shape[1] - 1
        ).astype(np.int64)
        full = cell_scores[rows[:, None], cols[None, :]]
        full = gaussian_blur(full, self.blur_sigma_px)
        return (full * roi).astype(np.float32), roi

    # -- persistence -----------------------------------------------------
    def save(self, path: str | Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("cannot save an unfitted StatisticalBackend")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            target,
            mean=self._mean,
            inv_cov=self._inv_cov,
            scale=self._scale,
            raw_low=self._raw_low,
            raw_high=self._raw_high,
            safety_factor=self.safety_factor,
            threshold=self.settings.threshold,
            patch_size=self.patch_size,
            stride=self.stride,
        )

    def load(self, path: str | Path) -> None:
        with np.load(Path(path)) as data:
            self._mean = data["mean"]
            self._inv_cov = data["inv_cov"]
            self._scale = data["scale"]
            self._raw_low = float(data["raw_low"])
            self._raw_high = float(data["raw_high"])
            self.safety_factor = float(data["safety_factor"])
            self.settings.threshold = float(data["threshold"])
            stored_patch = int(data["patch_size"])
            stored_stride = int(data["stride"])
        if (stored_patch, stored_stride) != (self.patch_size, self.stride):
            raise ValueError(
                f"model was trained with patch/stride {stored_patch}/{stored_stride} "
                f"but config says {self.patch_size}/{self.stride}; retrain or fix the config"
            )
