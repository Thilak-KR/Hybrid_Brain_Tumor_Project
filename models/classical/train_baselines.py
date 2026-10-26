"""Train classical baselines for mask-positive versus zero-mask slices."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.dataset_builder import feature_columns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "results" / "phase2_full_dataset_seed42"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_SEED = 42
METADATA_COLUMNS = ("patient_id", "slice_index")
TARGET_COLUMN = "target"


def _read_feature_columns(dataset_dir: Path) -> tuple[str, ...]:
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    columns = manifest.get("feature_columns")
    if not isinstance(columns, list) or not all(
        isinstance(column, str) and column for column in columns
    ):
        raise ValueError("manifest feature_columns must be a list of names")

    selected = tuple(columns)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("manifest feature_columns must be nonempty and unique")
    if set(selected) & {*METADATA_COLUMNS, TARGET_COLUMN}:
        raise ValueError("manifest feature_columns contains metadata or target columns")
    if selected != tuple(feature_columns()):
        raise ValueError(
            "manifest feature_columns do not match backend.dataset_builder.feature_columns()"
        )
    return selected


def _read_split(path: Path, features: tuple[str, ...]) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    frame = pd.read_csv(path)
    expected_columns = [*METADATA_COLUMNS, TARGET_COLUMN, *features]
    if frame.columns.tolist() != expected_columns:
        raise ValueError(
            f"{path.name} columns do not match the expected ordered schema"
        )
    if frame.empty:
        raise ValueError(f"{path.name} contains no rows")
    if frame.loc[:, METADATA_COLUMNS].isna().any().any():
        raise ValueError(f"{path.name} contains missing patient metadata")

    try:
        feature_frame = frame.loc[:, features].astype(np.float64)
        targets = pd.to_numeric(frame[TARGET_COLUMN], errors="raise")
        target_values = targets.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path.name} has non-numeric model values") from exc
    if not np.isfinite(feature_frame.to_numpy()).all():
        raise ValueError(f"{path.name} contains non-finite feature values")
    if not np.isfinite(target_values).all() or not np.isin(
        target_values, (0, 1)
    ).all():
        raise ValueError(f"{path.name} targets must contain only 0 and 1")

    targets = targets.astype(np.int64)
    class_counts = targets.value_counts().reindex((0, 1), fill_value=0)
    class_total = len(targets)
    distribution = {
        str(label): {
            "count": int(class_counts[label]),
            "percentage": round(100 * int(class_counts[label]) / class_total, 4),
        }
        for label in (0, 1)
    }
    return feature_frame, targets, {
        "rows": len(frame),
        "unique_patients": int(frame["patient_id"].nunique()),
        "class_distribution": distribution,
    }


def _evaluate(model: Any, features: pd.DataFrame, targets: pd.Series) -> dict[str, Any]:
    predictions = model.predict(features)
    roc_auc = None
    if targets.nunique() == 2 and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        positive_column = list(model.classes_).index(1)
        roc_auc = float(roc_auc_score(targets, probabilities[:, positive_column]))

    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "f1_score": float(f1_score(targets, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=(0, 1)
        ).tolist(),
        "confusion_matrix_labels": [0, 1],
        "roc_auc": roc_auc,
    }


def _reserve_output_dir(results_root: Path, seed: int) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    base_name = f"classical_baselines_seed{seed}_{timestamp}"
    for suffix in range(1000):
        name = base_name if suffix == 0 else f"{base_name}_{suffix}"
        output_dir = results_root / name
        try:
            output_dir.mkdir()
        except FileExistsError:
            continue
        return output_dir
    raise FileExistsError("Unable to reserve a unique baseline output directory")


def run_baselines(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    *,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Fit both baselines on train.csv and compare them on validation.csv."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    dataset_path = Path(dataset_dir)
    selected_features = _read_feature_columns(dataset_path)
    x_train, y_train, train_summary = _read_split(
        dataset_path / "train.csv", selected_features
    )
    x_validation, y_validation, validation_summary = _read_split(
        dataset_path / "validation.csv", selected_features
    )
    if set(y_train.unique()) != {0, 1}:
        raise ValueError("train.csv must contain both target classes")

    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=200,
            n_jobs=-1,
            random_state=seed,
        ),
    }
    validation_metrics: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        validation_metrics[name] = _evaluate(
            model, x_validation, y_validation
        )

    selected_model_name = max(
        models,
        key=lambda name: (
            validation_metrics[name]["f1_score"],
            validation_metrics[name]["roc_auc"]
            if validation_metrics[name]["roc_auc"] is not None
            else -1.0,
            validation_metrics[name]["accuracy"],
        ),
    )
    output_dir = _reserve_output_dir(Path(results_root), seed)
    for name, model in models.items():
        joblib.dump(model, output_dir / f"{name}.joblib")

    report = {
        "random_seed": seed,
        "dataset_dir": str(dataset_path.resolve()),
        "feature_columns": list(selected_features),
        "excluded_columns": list(METADATA_COLUMNS),
        "target_column": TARGET_COLUMN,
        "target_definition": (
            "1 means the axial slice contains any nonzero segmentation-mask voxel; "
            "0 means it contains none. This is mask-presence classification, "
            "not tumor-subregion classification or segmentation."
        ),
        "class_weighting": "balanced for both estimators; no rows or labels altered",
        "data_usage": {
            "fit_split": "train.csv",
            "comparison_split": "validation.csv",
            "test_split_loaded": False,
        },
        "split_summaries": {
            "train": train_summary,
            "validation": validation_summary,
        },
        "validation_metrics": validation_metrics,
        "selection": {
            "metric_order": ["f1_score", "roc_auc", "accuracy"],
            "selected_model": selected_model_name,
        },
        "artifacts": {name: f"{name}.joblib" for name in models},
    }
    (output_dir / "validation_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    output_dir = run_baselines(
        args.dataset_dir,
        results_root=args.results_root,
        seed=args.seed,
    )
    report = json.loads(
        (output_dir / "validation_results.json").read_text(encoding="utf-8")
    )
    print(f"Baseline artifacts written to {output_dir}")
    print(f"Selected model: {report['selection']['selected_model']}")
    for name, metrics in report["validation_metrics"].items():
        print(
            f"{name}: accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1_score']:.4f}, "
            f"roc_auc={metrics['roc_auc']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())