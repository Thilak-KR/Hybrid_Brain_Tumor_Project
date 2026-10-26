"""Synthetic unit tests for slice-level MRI features and labels."""

import unittest

import numpy as np

from feature_extraction.slice_features import (
    create_slice_label,
    extract_slice_features,
)


MODALITIES = ("t1c", "t1n", "t2w", "t2f")


def make_volumes() -> dict[str, np.ndarray]:
    base = np.array(
        [
            [[1, 5], [2, 6]],
            [[3, 7], [4, 8]],
        ],
        dtype=np.float32,
    )
    return {
        modality: base + offset
        for modality, offset in zip(MODALITIES, (0, 10, 20, 30))
    }


class SliceFeatureTests(unittest.TestCase):
    def test_returns_expected_statistics_in_modality_order(self) -> None:
        features = extract_slice_features(make_volumes(), 0)
        expected = np.array(
            [
                2.5, np.sqrt(1.25), 2.5, 1.5,
                12.5, np.sqrt(1.25), 12.5, 1.5,
                22.5, np.sqrt(1.25), 22.5, 1.5,
                32.5, np.sqrt(1.25), 32.5, 1.5,
            ]
        )
        self.assertEqual(features.shape, (16,))
        np.testing.assert_allclose(features, expected)

    def test_default_policy_includes_zero_intensity_pixels(self) -> None:
        volumes = {
            modality: np.array([[[0.0], [2.0]], [[4.0], [6.0]]])
            for modality in MODALITIES
        }
        features = extract_slice_features(volumes, 0)
        self.assertEqual(features[0], 3.0)

    def test_explicit_foreground_masks_control_selected_pixels(self) -> None:
        volumes = {
            modality: np.array([[[0.0], [2.0]], [[4.0], [6.0]]])
            for modality in MODALITIES
        }
        masks = {
            modality: np.array([[[False], [True]], [[True], [False]]])
            for modality in MODALITIES
        }
        features = extract_slice_features(
            volumes, 0, foreground_masks=masks
        )
        self.assertEqual(features[0], 3.0)
        self.assertEqual(features[2], 3.0)

    def test_empty_foreground_has_four_zero_features_per_modality(self) -> None:
        volumes = make_volumes()
        masks = {
            modality: np.zeros(volume.shape, dtype=bool)
            for modality, volume in volumes.items()
        }
        features = extract_slice_features(
            volumes, 0, foreground_masks=masks
        )
        np.testing.assert_array_equal(features, np.zeros(16))

    def test_nonfinite_image_pixels_are_excluded(self) -> None:
        volumes = {
            modality: np.array([[[1.0], [np.nan]], [[3.0], [np.inf]]])
            for modality in MODALITIES
        }
        features = extract_slice_features(volumes, 0)
        self.assertEqual(features[0], 2.0)
        self.assertEqual(features[2], 2.0)
        self.assertEqual(features[3], 1.0)

    def test_missing_modality_is_rejected(self) -> None:
        volumes = make_volumes()
        volumes.pop("t2f")
        with self.assertRaisesRegex(ValueError, "Missing required"):
            extract_slice_features(volumes, 0)

    def test_mismatched_modality_shapes_are_rejected(self) -> None:
        volumes = make_volumes()
        volumes["t2w"] = np.zeros((2, 2, 3))
        with self.assertRaisesRegex(ValueError, "does not match"):
            extract_slice_features(volumes, 0)

    def test_invalid_slice_indices_are_rejected(self) -> None:
        volumes = make_volumes()
        for index in (-1, 2):
            with self.subTest(index=index):
                with self.assertRaises(IndexError):
                    extract_slice_features(volumes, index)

    def test_foreground_mask_shape_and_dtype_are_validated(self) -> None:
        volumes = make_volumes()
        wrong_shape = {
            modality: np.zeros((2, 2, 3), dtype=bool)
            for modality in MODALITIES
        }
        with self.assertRaisesRegex(ValueError, "mask shape"):
            extract_slice_features(volumes, 0, foreground_masks=wrong_shape)

        wrong_dtype = {
            modality: np.ones(volume.shape, dtype=np.uint8)
            for modality, volume in volumes.items()
        }
        with self.assertRaisesRegex(TypeError, "must be boolean"):
            extract_slice_features(volumes, 0, foreground_masks=wrong_dtype)

    def test_slice_labels_are_binary_without_subregion_interpretation(self) -> None:
        segmentation = np.zeros((2, 2, 2), dtype=np.float32)
        segmentation[0, 0, 0] = 1
        segmentation[1, 1, 1] = 3
        self.assertEqual(
            create_slice_label(
                segmentation, 0, expected_shape=(2, 2, 2)
            ),
            1,
        )
        self.assertEqual(
            create_slice_label(
                segmentation, 1, expected_shape=(2, 2, 2)
            ),
            1,
        )
        self.assertEqual(
            create_slice_label(
                np.zeros_like(segmentation),
                0,
                expected_shape=(2, 2, 2),
            ),
            0,
        )

    def test_segmentation_shape_and_slice_index_are_validated(self) -> None:
        segmentation = np.zeros((2, 2, 2), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "does not match"):
            create_slice_label(
                segmentation, 0, expected_shape=(2, 2, 1)
            )
        with self.assertRaises(IndexError):
            create_slice_label(
                segmentation, 2, expected_shape=(2, 2, 2)
            )

    def test_invalid_segmentation_values_are_rejected(self) -> None:
        invalid_values = (np.nan, 1.5, 4)
        for value in invalid_values:
            with self.subTest(value=value):
                segmentation = np.zeros((2, 2, 2), dtype=np.float64)
                segmentation[0, 0, 0] = value
                with self.assertRaises(ValueError):
                    create_slice_label(
                        segmentation, 0, expected_shape=(2, 2, 2)
                    )

    def test_input_arrays_are_not_modified(self) -> None:
        volumes = make_volumes()
        originals = {key: value.copy() for key, value in volumes.items()}
        segmentation = np.zeros((2, 2, 2), dtype=np.uint8)
        segmentation[0, 0, 0] = 2
        segmentation_original = segmentation.copy()
        masks = {
            modality: np.ones(volume.shape, dtype=bool)
            for modality, volume in volumes.items()
        }
        mask_originals = {key: value.copy() for key, value in masks.items()}

        extract_slice_features(volumes, 0, foreground_masks=masks)
        create_slice_label(
            segmentation, 0, expected_shape=(2, 2, 2)
        )

        for key in volumes:
            np.testing.assert_array_equal(volumes[key], originals[key])
            np.testing.assert_array_equal(masks[key], mask_originals[key])
        np.testing.assert_array_equal(segmentation, segmentation_original)


if __name__ == "__main__":
    unittest.main()
