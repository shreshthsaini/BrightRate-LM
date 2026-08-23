#!/usr/bin/env python3
"""Shared paths and resumable I/O for the BrightVQ MLLM benchmark."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

from config import repo_path

METADATA = repo_path("metadata")
SPLITS = repo_path("splits")
RESULTS_DIR = repo_path("adapters").parent / "benchmark-results"
VIDEOS_DIR = repo_path("videos")
FRAMES_ROOT = repo_path("benchmark_frames")
HF_HOME = repo_path("hf_cache")
LEVELS = ("bad", "poor", "fair", "good", "excellent")
LEVEL_VALUES = (1.0, 2.0, 3.0, 4.0, 5.0)


def load_metadata(path: Path = METADATA) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["Video"] for row in rows]
    if len(rows) != 2100 or len(set(ids)) != 2100:
        raise ValueError(f"Expected 2100 unique videos in {path}, found {len(rows)}")
    return rows


def load_splits(path: Path = SPLITS) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if len(data.get("splits", [])) != 100:
        raise ValueError(f"Expected 100 splits in {path}")
    return data


def target_ids(
    target: str, rows: list[dict[str, str]], splits_path: Path = SPLITS
) -> list[str]:
    ordered = [row["Video"] for row in rows]
    if target == "all":
        return ordered
    if target in {"train0", "test0"}:
        side = "train" if target == "train0" else "test"
        selected = set(load_splits(splits_path)["splits"][0][side])
        return [video_id for video_id in ordered if video_id in selected]
    ids_path = Path(target)
    selected = {line.strip() for line in ids_path.read_text().splitlines() if line.strip()}
    unknown = selected.difference(ordered)
    if unknown:
        raise ValueError(f"Unknown IDs in {ids_path}: {sorted(unknown)[:5]}")
    return [video_id for video_id in ordered if video_id in selected]


def frame_paths(variant: str, video_id: str, root: Path = FRAMES_ROOT) -> list[Path]:
    paths = sorted((root / variant / video_id).glob("frame_*.png"))
    if len(paths) != 8:
        raise FileNotFoundError(
            f"Expected 8 cached {variant} frames for {video_id}, found {len(paths)}"
        )
    return paths


def load_predictions(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["Video", "score"]:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")
        rows = list(reader)
    result: dict[str, float] = {}
    for row in rows:
        video_id = row["Video"]
        if video_id in result:
            raise ValueError(f"Duplicate prediction for {video_id} in {path}")
        result[video_id] = float(row["score"])
    return result


def append_predictions(path: Path, rows: Iterable[tuple[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if needs_header:
            writer.writerow(["Video", "score"])
        for video_id, score in rows:
            writer.writerow([video_id, f"{score:.10f}"])
        handle.flush()
        os.fsync(handle.fileno())


def rewrite_predictions(path: Path, ids: list[str], values: dict[str, float]) -> None:
    missing = [video_id for video_id in ids if video_id not in values]
    if missing:
        raise ValueError(f"Cannot finalize {path}: {len(missing)} predictions are missing")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        tmp = Path(handle.name)
        writer = csv.writer(handle)
        writer.writerow(["Video", "score"])
        for video_id in ids:
            writer.writerow([video_id, f"{values[video_id]:.10f}"])
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        tmp = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
