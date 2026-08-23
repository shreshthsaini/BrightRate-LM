#!/usr/bin/env python3
"""Compute the BrightVQ 100-split metrics used by the zero-shot study."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.stats import kendalltau, pearsonr, spearmanr

from benchmark_io import METADATA, SPLITS, load_metadata, load_predictions, load_splits


def logistic4(
    x: np.ndarray, high: float, low: float, midpoint: float, scale: float
) -> np.ndarray:
    exponent = np.clip(-(x - midpoint) / scale, -60.0, 60.0)
    return low + (high - low) / (1.0 + np.exp(exponent))


def logistic_map(prediction: np.ndarray, mos: np.ndarray) -> tuple[np.ndarray, bool]:
    center = float(np.median(prediction))
    spread = float(np.std(prediction))
    if spread < 1e-10:
        return np.full_like(mos, float(np.mean(mos))), False
    x = (prediction - center) / spread
    positive = spearmanr(x, mos).statistic >= 0
    initial = [
        float(np.max(mos) if positive else np.min(mos)),
        float(np.min(mos) if positive else np.max(mos)),
        0.0,
        1.0,
    ]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            params, _ = curve_fit(
                logistic4,
                x,
                mos,
                p0=initial,
                bounds=(
                    [-500.0, -500.0, -20.0, 1e-3],
                    [500.0, 500.0, 20.0, 100.0],
                ),
                maxfev=50000,
            )
        mapped = logistic4(x, *params)
        if not np.all(np.isfinite(mapped)):
            raise ValueError("Non-finite logistic mapping")
        return mapped, True
    except (RuntimeError, ValueError, OptimizeWarning):
        slope, intercept = np.polyfit(prediction, mos, 1)
        return slope * prediction + intercept, False


def compute_metrics(
    prediction_path: Path,
    ids: list[str],
    mos_by_id: dict[str, float],
    splits: dict,
) -> dict:
    predictions = load_predictions(prediction_path)
    if set(predictions) != set(ids):
        raise ValueError("Prediction file must contain each BrightVQ video exactly once")
    per_split = []
    fallback_count = 0
    for index, split in enumerate(splits["splits"]):
        test_ids = split["test"]
        prediction = np.asarray([predictions[item] for item in test_ids])
        mos = np.asarray([mos_by_id[item] for item in test_ids])
        mapped, fitted = logistic_map(prediction, mos)
        fallback_count += int(not fitted)
        per_split.append(
            {
                "split": index,
                "n": len(test_ids),
                "srocc": float(spearmanr(prediction, mos).statistic),
                "plcc": float(pearsonr(mapped, mos).statistic),
                "krcc": float(kendalltau(prediction, mos).statistic),
                "rmse": float(np.sqrt(np.mean((mapped - mos) ** 2))),
                "logistic_fit": fitted,
            }
        )
    summary = {}
    for name in ("srocc", "plcc", "krcc", "rmse"):
        values = np.asarray([row[name] for row in per_split])
        summary[name] = {
            "median": float(np.median(values)),
            "std": float(np.std(values, ddof=0)),
        }
    return {
        "prediction_file": str(prediction_path),
        "count": len(ids),
        "logistic": "four-parameter fit on each test split",
        "logistic_fallback_splits": fallback_count,
        "summary": summary,
        "per_split": per_split,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=METADATA)
    parser.add_argument("--splits", type=Path, default=SPLITS)
    args = parser.parse_args()

    rows = load_metadata(args.metadata)
    ids = [row["Video"] for row in rows]
    mos_by_id = {row["Video"]: float(row["mos_j"]) for row in rows}
    result = compute_metrics(
        args.prediction, ids, mos_by_id, load_splits(args.splits)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    medians = {name: values["median"] for name, values in result["summary"].items()}
    print(json.dumps(medians, indent=2))


if __name__ == "__main__":
    main()
