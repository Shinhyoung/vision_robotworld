"""EfficientAD backend built on Intel anomalib.

This is the production detector fixed by claude.md section 1. anomalib, torch
and a Blackwell-capable CUDA build are *optional* imports: the module must be
importable on a CPU-only CI runner so the factory can report a clear error
instead of crashing at import time.

Training data are defect-free images only (EfficientAD is unsupervised); see
``docs/agent_tickets.md`` ticket INS-3 for the capture procedure.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from ..imageops import resize_nearest
from ..segmentation import segment_part
from ..types import Frame
from .base import InspectionBackend, InspectionSettings


class EfficientAdUnavailable(RuntimeError):
    """Raised when anomalib/torch are missing or no checkpoint is present."""


def _import_anomalib() -> tuple[Any, Any]:
    try:
        import torch
        from anomalib.models import EfficientAd
    except ImportError as exc:  # pragma: no cover - depends on deployment env
        raise EfficientAdUnavailable(
            "anomalib/torch not importable. Install the GPU extras "
            "(`pip install -r requirements-gpu.txt`) or set "
            "inspection.backend to 'statistical'. Original error: " + str(exc)
        ) from exc
    return torch, EfficientAd


class EfficientAdBackend(InspectionBackend):
    """anomalib EfficientAD wrapper."""

    name = "efficientad"

    def __init__(
        self,
        settings: InspectionSettings,
        model_path: str | Path | None = None,
        image_size: tuple[int, int] = (256, 256),
        device: str = "cuda",
        model_size: str = "small",
        segmentation_kwargs: dict | None = None,
    ) -> None:
        super().__init__(settings)
        self.model_path = Path(model_path) if model_path else None
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.device = str(device)
        self.model_size = str(model_size)
        self.segmentation_kwargs = dict(segmentation_kwargs or {})
        self._model: Any = None
        self._torch: Any = None

    # -- lifecycle -------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if self.model_path is None or not self.model_path.exists():
            raise EfficientAdUnavailable(
                f"EfficientAD checkpoint not found at {self.model_path}. Train one with "
                "`python tools/train_inspection.py --backend efficientad` or switch "
                "inspection.backend to 'statistical'."
            )
        self.load(self.model_path)

    def load(self, path: str | Path) -> None:
        torch, EfficientAd = _import_anomalib()
        self._torch = torch
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise EfficientAdUnavailable(f"checkpoint not found: {checkpoint_path}")
        device = self.device if torch.cuda.is_available() else "cpu"
        model = EfficientAd.load_from_checkpoint(str(checkpoint_path), map_location=device)
        model.eval()
        model.to(device)
        self.device = device
        self._model = model

    def fit(self, frames: Iterable[Frame]) -> None:
        """Training runs through anomalib's Engine, not in-process.

        anomalib expects a folder dataset and a Lightning trainer; wiring that
        here would duplicate ``tools/train_inspection.py``. Keeping training out
        of the runtime backend also keeps the inference node free of a torch
        import when the statistical backend is selected.
        """
        raise NotImplementedError(
            "train EfficientAD with tools/train_inspection.py --backend efficientad, "
            "then point inspection.efficientad.model_path at the checkpoint"
        )

    # -- inference -------------------------------------------------------
    def score_map(self, frame: Frame) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_model()
        torch = self._torch

        roi = segment_part(frame, **self.segmentation_kwargs).mask
        height, width = frame.depth.shape

        # anomalib expects a normalised NCHW float tensor.
        resized = resize_nearest(frame.color, self.image_size).astype(np.float32) / 255.0
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self._model(tensor)

        anomaly_map = _extract_map(output)
        anomaly_map = anomaly_map.astype(np.float32)
        full = resize_nearest(anomaly_map, (height, width)).astype(np.float32)

        # anomalib emits an unbounded map; squash it into the [0, 1] contract
        # using the model's own adaptive threshold when it exposes one.
        pixel_threshold = _adaptive_threshold(self._model)
        if pixel_threshold is not None and pixel_threshold > 0:
            # Map the model threshold onto our configured decision threshold so
            # inspection.threshold keeps its meaning across backends.
            full = full / (2.0 * pixel_threshold)
        full = np.clip(full, 0.0, 1.0)
        return (full * roi).astype(np.float32), roi


def _extract_map(output: Any) -> np.ndarray:
    """Pull the anomaly map out of the several shapes anomalib returns."""
    candidate = output
    for attr in ("anomaly_map", "pred_score"):
        if hasattr(candidate, attr):
            candidate = getattr(candidate, attr)
            break
    else:
        if isinstance(output, dict) and "anomaly_map" in output:
            candidate = output["anomaly_map"]
        elif isinstance(output, (tuple, list)) and output:
            candidate = output[0]

    array = candidate.detach().cpu().numpy() if hasattr(candidate, "detach") else np.asarray(
        candidate
    )
    return np.squeeze(array)


def _adaptive_threshold(model: Any) -> float | None:
    for attr in ("pixel_threshold", "image_threshold"):
        threshold = getattr(model, attr, None)
        value = getattr(threshold, "value", None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                continue
    return None
