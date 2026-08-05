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
from ..segmentation import part_crop_box, segment_part
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
        crop_margin: float = 0.15,
    ) -> None:
        super().__init__(settings)
        self.model_path = Path(model_path) if model_path else None
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.device = str(device)
        self.model_size = str(model_size)
        self.segmentation_kwargs = dict(segmentation_kwargs or {})
        self.crop_margin = float(crop_margin)
        self._model: Any = None
        self._torch: Any = None
        self._norm_anchor: float | None = None

    # -- lifecycle -------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        """Whether a usable model exists -- loaded, or waiting on disk.

        Reports the checkpoint too, not just the loaded module. Callers use this
        to decide whether to train, and this backend is never trained in-process
        (``fit`` raises); a bare ``self._model is not None`` made every caller
        try to fit a model that was already sitting next to them.
        """
        return self._model is not None or (
            self.model_path is not None and self.model_path.exists()
        )

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

    @staticmethod
    def anchor_path(checkpoint: str | Path) -> Path:
        """Sidecar holding the normalisation anchor for a checkpoint.

        Kept beside the .ckpt rather than inside it: anomalib owns that file's
        format, and a checkpoint copied without its anchor should fail loudly
        rather than score against a silent default.
        """
        return Path(checkpoint).with_suffix(".norm.json")

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

        anchor_file = self.anchor_path(checkpoint_path)
        if anchor_file.is_file():
            import json

            self._norm_anchor = float(json.loads(anchor_file.read_text())["anchor"])

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

        # Crop to the part first: resizing the whole frame spends the model's
        # input on belt and shrinks defects below what it can see. Must match
        # what the training images were cropped with -- see part_crop_box.
        r0, r1, c0, c1 = part_crop_box(roi, self.crop_margin)
        window = np.ascontiguousarray(frame.color[r0:r1, c0:c1])

        # anomalib expects a normalised NCHW float tensor.
        resized = resize_nearest(window, self.image_size).astype(np.float32) / 255.0
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = _inference_module(self._model)(tensor)

        anomaly_map = _extract_map(output).astype(np.float32)
        # Back to the crop's own size, then into the frame it came from.
        full = np.zeros((height, width), dtype=np.float32)
        full[r0:r1, c0:c1] = resize_nearest(anomaly_map, (r1 - r0, c1 - c0))

        # score = 0.5 * raw / anchor -- the statistical backend's contract, so
        # inspection.threshold means the same thing whichever backend is loaded.
        scaled = np.clip(0.5 * full / self._anchor(), 0.0, 1.0)
        return (scaled * roi).astype(np.float32), roi

    def _anchor(self) -> float:
        """Divisor that puts score 0.5 at ``safety_factor`` x the normal level.

        anomalib emits an unbounded map, so it has to be anchored on something
        the line will actually see. Calibrated on the *training* score
        distribution and stored beside the checkpoint by
        ``tools/train_inspection.py`` -- the same rule the statistical backend
        uses, and for the same reason (see its ``normalise``): anchoring on the
        training **level** keeps the scale stable as the normal set grows, where
        anchoring on its spread quietly pushes defects under the threshold.

        Two earlier attempts failed, both silently:
        * dividing by the model's own ``pixel_threshold``. anomalib 2.x does not
          expose it -- the attribute is ``None``, the normalisation was skipped
          and the raw map became the score.
        * using the raw map as-is. Measured on mock guide_block: a good part
          scored 0.665 against a 0.5 threshold, i.e. a false reject on every
          frame, while a chipped one scored 0.872. Separable, but not calibrated.
        """
        if self._norm_anchor is None:
            raise EfficientAdUnavailable(
                f"no normalisation anchor beside {self.model_path}. Retrain with "
                "`tools/train_inspection.py --backend efficientad`, which "
                "calibrates it on the training set."
            )
        return max(self._norm_anchor, 1e-9)


def _inference_module(model: Any) -> Any:
    """The sub-module that returns *raw* anomaly scores.

    anomalib 2.x's Lightning module post-processes its output with a per-image
    min-max normalisation, which makes every frame's map span exactly [0, 1] --
    good and defective alike. Measured on mock guide_block: the Lightning module
    returned max 1.0000 for both a clean part and a chipped one, so every defect
    scored identically (0.639) and the detector was blind. The inner torch model
    returns the scores the normalisation is derived from: 0.2095 clean vs 0.2903
    chipped, which is what a threshold can actually separate.

    Per-image normalisation is right for a picture and wrong for a decision: it
    removes exactly the between-image level that OK/NG depends on. Same trap as
    normalising on the training spread instead of its level.
    """
    inner = getattr(model, "model", None)
    return inner if callable(inner) else model


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
    """The model's own threshold, when it publishes one.

    anomalib 2.x does not: both attributes read ``None``. Kept because 1.x did
    and a future release may again, but nothing may depend on it -- see
    :meth:`EfficientAdBackend._anchor`.
    """
    for attr in ("pixel_threshold", "image_threshold"):
        threshold = getattr(model, attr, None)
        value = getattr(threshold, "value", None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                continue
    return None
