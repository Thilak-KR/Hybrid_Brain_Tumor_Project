"""Deterministic slice-level MRI features and mask-positive labels."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import numpy as np

from config import settings

FEATURE_NAMES = ("mean", "std", "median", "iqr")
SUPPORTED_SEGMENTATION_LABELS = frozenset({0, 1, 2, 3})


def _validate_slice_index(slice_index: int, depth: int) -> int:
    if isinstance(slice_index, bool) or not isinstance(slice_index, Integral):
        raise TypeError("slice_index must be an integer")
    if slice_index < 0 or slice_index >= depth:
        raise IndexError(
            f"slice_index {slice_index} is outside valid range 0..{depth - 1}"
        )
    return int(slice_index)


def _validate_modalities(
    modality_volumes: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], tuple[int, int, int]]:
    missing = [name for name in settings.MRI_MODALITIES if name not in modality_volumes]
    if missing:
        raise ValueError(f"Missing required MRI modalities: {', '.join(missing)}")

    volumes: dict[str, np.ndarray] = {}
    reference_shape: tuple[int, ...] | None = None
    for modality in settings.MRI_MODALITIES:
        volume = np.asarray(modality_volumes[modality])
        if volume.ndim != 3:
            raise ValueError(
                f"{modality} volume must be 3D; received shape {volume.shape}"
            )
        if not np.issubdtype(volume.dtype, np.number) or np.iscomplexobj(volume):
            raise TypeError(f"{modality} volume must contain real numeric values")
        if reference_shape is None:
            reference_shape = volume.shape
        elif volume.shape != reference_shape:
            raise ValueError(
                f"{modality} shape {volume.shape} does not match "
                f"{settings.MRI_MODALITIES[0]} shape {reference_shape}"
            )
        volumes[modality] = volume

    assert reference_shape is not None
    return volumes, reference_shape


def extract_slice_features(
    modality_volumes: Mapping[str, np.ndarray],
    slice_index: int,
    *,
    foreground_masks: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Return 16 ordered statistics for one axial slice.

    The feature order is modality order from settings (t1c, t1n, t2w, t2f),
    with mean, population standard deviation, median, and interquartile range
    (75th minus 25th percentile) per modality.

    By default, all finite pixels are included, including zero-intensity
    pixels. This avoids assuming that zero means background. Callers may
    instead provide a boolean foreground mask for each modality, where True
    selects pixels for feature computation. Non-finite intensities are always
    excluded. If a modality has no selected finite pixels, its four features
    are zeros.

    This function does not inspect or use segmentation masks. Input arrays are
    not modified.

    Args:
        modality_volumes: Mapping containing all four configured 3D modalities.
        slice_index: Zero-based axial slice index shared by all volumes.
        foreground_masks: Optional mapping of modality names to 3D boolean
            masks matching their volumes. True means include the voxel.

    Returns:
        A float64 array of shape (16,) in the documented feature order.

    Raises:
        TypeError: If image volumes or foreground masks have invalid dtypes.
        ValueError: If a modality is missing, dimensions do not match, or a
            foreground mask is missing, extra, or has the wrong shape.
        IndexError: If slice_index is outside the volume depth.
    """
    volumes, shape = _validate_modalities(modality_volumes)
    index = _validate_slice_index(slice_index, shape[2])

    if foreground_masks is not None:
        required = set(settings.MRI_MODALITIES)
        supplied = set(foreground_masks)
        if supplied != required:
            missing = sorted(required - supplied)
            extra = sorted(supplied - required)
            details = []
            if missing:
                details.append(f"missing masks for {', '.join(missing)}")
            if extra:
                details.append(f"unexpected masks for {', '.join(extra)}")
            raise ValueError("Invalid foreground_masks: " + "; ".join(details))

    features: list[float] = []
    for modality in settings.MRI_MODALITIES:
        slice_data = volumes[modality][:, :, index]
        selected = np.isfinite(slice_data)

        if foreground_masks is not None:
            mask = np.asarray(foreground_masks[modality])
            if mask.dtype != np.bool_:
                raise TypeError(f"{modality} foreground mask must be boolean")
            if mask.shape != shape:
                raise ValueError(
                    f"{modality} foreground mask shape {mask.shape} "
                    f"does not match volume shape {shape}"
                )
            selected &= mask[:, :, index]

        values = slice_data[selected].astype(np.float64, copy=False)
        if values.size == 0:
            features.extend((0.0, 0.0, 0.0, 0.0))
            continue

        lower_quartile, upper_quartile = np.percentile(
            values, (25, 75), method="linear"
        )
        features.extend(
            (
                float(np.mean(values)),
                float(np.std(values, ddof=0)),
                float(np.median(values)),
                float(upper_quartile - lower_quartile),
            )
        )

    return np.asarray(features, dtype=np.float64)


def create_slice_label(
    segmentation: np.ndarray,
    slice_index: int,
    *,
    expected_shape: tuple[int, int, int],
) -> int:
    """Return 1 for a mask-positive slice and 0 for an all-zero mask slice.

    Segmentation values are validated across the supplied volume and must be
    finite integral values in {0, 1, 2, 3}. This function makes no assumptions
    about the specific meanings of labels 1, 2, or 3. A zero-mask slice is only
    a slice without positive mask pixels; it does not imply a healthy patient.

    Args:
        segmentation: Three-dimensional segmentation label volume.
        slice_index: Zero-based axial slice index.
        expected_shape: Shape of the associated MRI volumes, used to verify
            dimensions match exactly.

    Returns:
        Binary mask-positive slice label (0 or 1).

    Raises:
        TypeError: If segmentation contains non-real-numeric values.
        ValueError: If dimensions or segmentation values are invalid.
        IndexError: If slice_index is outside the volume depth.
    """
    mask = np.asarray(segmentation)
    if mask.ndim != 3:
        raise ValueError(
            f"segmentation must be 3D; received shape {mask.shape}"
        )
    if mask.shape != expected_shape:
        raise ValueError(
            f"segmentation shape {mask.shape} does not match "
            f"MRI shape {expected_shape}"
        )
    if not np.issubdtype(mask.dtype, np.number) or np.iscomplexobj(mask):
        raise TypeError("segmentation must contain real numeric label values")
    if not np.isfinite(mask).all():
        raise ValueError("segmentation contains non-finite label values")
    if not np.equal(mask, np.floor(mask)).all():
        raise ValueError("segmentation contains non-integral label values")
    if not np.isin(mask, tuple(SUPPORTED_SEGMENTATION_LABELS)).all():
        raise ValueError(
            "segmentation contains unsupported labels; expected values "
            "in {0, 1, 2, 3}"
        )

    index = _validate_slice_index(slice_index, mask.shape[2])
    return int(np.any(mask[:, :, index] != 0))
