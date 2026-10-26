"""Intensity normalization for one 3D MRI modality at a time."""

from __future__ import annotations

from typing import Literal

import numpy as np

NormalizationStrategy = Literal["zscore", "minmax"]


def normalize_mri(
    volume: np.ndarray,
    *,
    strategy: NormalizationStrategy = "zscore",
    background_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Normalize a single 3D MRI volume without modifying the input.

    Supported strategies are population z-score normalization and min-max
    scaling to [0, 1]. By default, finite zero-valued voxels are treated as
    background and remain zero. To specify background explicitly, pass a
    boolean mask where True marks background; in that case zero-valued voxels
    not masked out are included in the statistics. Non-finite values are
    excluded from the statistics and set to zero in the output. A constant
    foreground (or a volume with no usable foreground voxels) returns zeros.

    This function is for MRI intensities only; do not pass segmentation masks.
    It preserves the input dimensions and returns float32 values.

    Args:
        volume: One numeric, real-valued 3D MRI modality.
        strategy: "zscore" or "minmax".
        background_mask: Optional same-shape boolean mask; True means
            background. If omitted, finite zero-valued voxels are background.

    Raises:
        TypeError: If the volume is non-numeric/complex or the mask is not bool.
        ValueError: If the volume is not 3D, strategy is unsupported, or mask
            dimensions do not match the volume.
    """
    values = np.asarray(volume)
    if values.ndim != 3:
        raise ValueError(f"volume must be 3D; received shape {values.shape}")
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise TypeError("volume must contain real numeric MRI intensities")
    if strategy not in ("zscore", "minmax"):
        raise ValueError(
            f"Unsupported normalization strategy {strategy!r}; "
            "choose 'zscore' or 'minmax'"
        )

    finite = np.isfinite(values)
    if background_mask is None:
        foreground = finite & (values != 0)
    else:
        mask = np.asarray(background_mask)
        if mask.dtype != np.bool_:
            raise TypeError("background_mask must have boolean dtype")
        if mask.shape != values.shape:
            raise ValueError(
                "background_mask shape must match volume shape; "
                f"received {mask.shape} and {values.shape}"
            )
        foreground = finite & ~mask

    normalized = np.zeros(values.shape, dtype=np.float32)
    if not np.any(foreground):
        return normalized

    selected = values[foreground].astype(np.float64, copy=False)
    if strategy == "zscore":
        center = selected.mean()
        scale = selected.std()
        if scale == 0:
            return normalized
        normalized[foreground] = ((selected - center) / scale).astype(np.float32)
    else:
        minimum = selected.min()
        value_range = selected.max() - minimum
        if value_range == 0:
            return normalized
        normalized[foreground] = ((selected - minimum) / value_range).astype(
            np.float32
        )

    return normalized
