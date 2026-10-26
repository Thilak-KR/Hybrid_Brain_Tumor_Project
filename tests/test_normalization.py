"""Focused synthetic tests for MRI intensity normalization."""

import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

normalization_path = (
    Path(__file__).resolve().parents[1] / "preprocessing" / "normalization.py"
)
normalization_spec = spec_from_file_location(
    "phase2_normalization", normalization_path
)
if normalization_spec is None or normalization_spec.loader is None:
    raise ImportError(f"Unable to load normalization module from {normalization_path}")
normalization_module = module_from_spec(normalization_spec)
normalization_spec.loader.exec_module(normalization_module)
normalize_mri = normalization_module.normalize_mri


class NormalizeMriTests(unittest.TestCase):
    def test_zscore_uses_nonzero_foreground(self) -> None:
        volume = np.array([[[0, 1, 3]]], dtype=np.int16)
        normalized = normalize_mri(volume, strategy="zscore")
        np.testing.assert_allclose(normalized, [[[0, -1, 1]]])

    def test_minmax_scales_foreground_and_preserves_background(self) -> None:
        volume = np.array([[[0, 1, 3]]], dtype=np.float32)
        normalized = normalize_mri(volume, strategy="minmax")
        np.testing.assert_allclose(normalized, [[[0, 0, 1]]])

    def test_explicit_background_mask_can_include_zero_intensities(self) -> None:
        volume = np.array([[[0, 2, 4]]], dtype=np.float32)
        background = np.array([[[False, False, True]]])
        normalized = normalize_mri(
            volume, strategy="minmax", background_mask=background
        )
        np.testing.assert_allclose(normalized, [[[0, 1, 0]]])

    def test_nonfinite_values_are_zero_and_excluded_from_statistics(self) -> None:
        volume = np.array([[[0, 1, np.nan, np.inf, 3]]], dtype=np.float32)
        normalized = normalize_mri(volume, strategy="zscore")
        np.testing.assert_allclose(normalized, [[[0, -1, 0, 0, 1]]])

    def test_constant_foreground_returns_zeros(self) -> None:
        volume = np.full((2, 2, 2), 7, dtype=np.int16)
        normalized = normalize_mri(volume)
        np.testing.assert_array_equal(normalized, np.zeros((2, 2, 2)))

    def test_preserves_dimensions_and_does_not_modify_input(self) -> None:
        volume = np.arange(8, dtype=np.int16).reshape((2, 2, 2))
        original = volume.copy()
        normalized = normalize_mri(volume, strategy="minmax")
        self.assertEqual(normalized.shape, volume.shape)
        np.testing.assert_array_equal(volume, original)

    def test_rejects_non_3d_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be 3D"):
            normalize_mri(np.ones((2, 2)))

    def test_rejects_unknown_strategy(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            normalize_mri(np.ones((2, 2, 2)), strategy="other")  # type: ignore[arg-type]

    def test_rejects_mismatched_background_mask(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape must match"):
            normalize_mri(
                np.ones((2, 2, 2)),
                background_mask=np.zeros((2, 2, 1), dtype=bool),
            )


if __name__ == "__main__":
    unittest.main()
