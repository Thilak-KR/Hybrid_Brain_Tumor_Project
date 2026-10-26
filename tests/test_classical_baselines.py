"""Focused tests for classical baseline feature and split isolation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from backend.dataset_builder import feature_columns
from models.classical.train_baselines import _read_split, run_baselines


FEATURES = tuple(feature_columns())


def make_frame(row_count: int) -> pd.DataFrame:
    data: dict[str, object] = {
        "patient_id": [f"patient-{index}" for index in range(row_count)],
        "slice_index": list(range(row_count)),
        "target": [index % 2 for index in range(row_count)],
    }
    for feature_index, name in enumerate(FEATURES):
        data[name] = [float(index + feature_index) for index in range(row_count)]
    return pd.DataFrame(data)


class ClassicalBaselineTests(unittest.TestCase):
    def test_model_features_exclude_metadata_columns(self) -> None:
        frame = make_frame(4)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic.csv"
            frame.to_csv(path, index=False)
            features, targets, _ = _read_split(path, FEATURES)

        self.assertEqual(tuple(features.columns), FEATURES)
        self.assertNotIn("patient_id", features.columns)
        self.assertNotIn("slice_index", features.columns)
        self.assertNotIn("target", features.columns)
        self.assertEqual(targets.tolist(), [0, 1, 0, 1])

    def test_models_fit_train_rows_only_and_do_not_load_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            (dataset_dir / "manifest.json").write_text(
                json.dumps({"feature_columns": list(FEATURES)}),
                encoding="utf-8",
            )
            make_frame(8).to_csv(dataset_dir / "train.csv", index=False)
            make_frame(4).to_csv(dataset_dir / "validation.csv", index=False)
            output_root = root / "results"

            fit_rows: dict[str, list[int]] = {
                "logistic_regression": [],
                "random_forest": [],
            }
            original_logistic_fit = LogisticRegression.fit
            original_forest_fit = RandomForestClassifier.fit

            def capture_logistic_fit(estimator, x, y, *args, **kwargs):
                fit_rows["logistic_regression"].append(len(x))
                return original_logistic_fit(estimator, x, y, *args, **kwargs)

            def capture_forest_fit(estimator, x, y, *args, **kwargs):
                fit_rows["random_forest"].append(len(x))
                return original_forest_fit(estimator, x, y, *args, **kwargs)

            with (
                patch.object(LogisticRegression, "fit", capture_logistic_fit),
                patch.object(RandomForestClassifier, "fit", capture_forest_fit),
                patch(
                    "models.classical.train_baselines.pd.read_csv",
                    wraps=pd.read_csv,
                ) as read_csv,
            ):
                output_dir = run_baselines(
                    dataset_dir,
                    results_root=output_root,
                    seed=11,
                )

            self.assertEqual(fit_rows["logistic_regression"], [8])
            self.assertEqual(fit_rows["random_forest"], [8])
            loaded_files = [Path(call.args[0]).name for call in read_csv.call_args_list]
            self.assertEqual(loaded_files, ["train.csv", "validation.csv"])
            report = json.loads(
                (output_dir / "validation_results.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(report["data_usage"]["test_split_loaded"])
            self.assertEqual(report["split_summaries"]["train"]["rows"], 8)
            self.assertEqual(
                report["split_summaries"]["validation"]["rows"], 4
            )
            self.assertTrue((output_dir / "logistic_regression.joblib").is_file())
            self.assertTrue((output_dir / "random_forest.joblib").is_file())


if __name__ == "__main__":
    unittest.main()