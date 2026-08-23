#!/usr/bin/env python3
"""Shared utilities for BrightRate-LM training and evaluation."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from config import repo_path, value

HF_HOME = repo_path("hf_cache")
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))

import numpy as np
import torch
from PIL import Image
from scipy.stats import kendalltau, pearsonr, spearmanr


PROJECT_ROOT = repo_path("metadata").parents[1]
DEFAULT_CSV = repo_path("metadata")
DEFAULT_SPLITS = repo_path("splits")
DEFAULT_FRAMES = repo_path("sdr_frames")
DEFAULT_MODEL = value("models", "primary")
LEVELS = ("bad", "poor", "fair", "good", "excellent")
LEVEL_TOKEN_TEXT = ("Bad", "Poor", "Fair", "Good", "Excellent")
LEVEL_MOS = np.asarray((0.0, 25.0, 50.0, 75.0, 100.0), dtype=np.float64)
RATING_PROMPT = (
    "These images are eight uniformly sampled frames in temporal order from one HDR "
    "user-generated video. Judge only perceptual visual quality, including compression, "
    "blur, noise, exposure, color, and temporal consistency. Rate the overall quality "
    "with exactly one word: Bad, Poor, Fair, Good, or Excellent."
)
DESCRIPTION_PROMPT = (
    "These images are uniformly sampled frames in temporal order from one HDR "
    "user-generated video. In one or two short sentences, describe the overall visual "
    "quality and the main visible quality problems. Do not mention that you saw frames."
)


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_rows(csv_path: Path | str = DEFAULT_CSV) -> list[dict[str, Any]]:
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["mos_j"] = float(row["mos_j"])
        row["sos_j"] = float(row["sos_j"])
        row["height"] = int(row["height"])
        row["width"] = int(row["width"])
    ids = [row["Video"] for row in rows]
    if len(rows) != 2100 or len(set(ids)) != len(ids):
        raise ValueError(f"Expected 2100 unique BrightVQ rows, found {len(rows)} rows")
    return rows


def rows_by_id(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["Video"]: row for row in rows}


def load_splits(path: Path | str = DEFAULT_SPLITS) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("separator") != "content_name":
        raise ValueError("Expected content_name-separated splits")
    if len(payload.get("splits", [])) < 5:
        raise ValueError("At least five splits are required")
    return payload


def validate_split(
    split: dict[str, list[str]], by_id: dict[str, dict[str, Any]]
) -> None:
    train_ids = list(split["train"])
    test_ids = list(split["test"])
    if set(train_ids) & set(test_ids):
        raise ValueError("Train and test video IDs overlap")
    if set(train_ids) | set(test_ids) != set(by_id):
        raise ValueError("Train and test IDs do not cover BrightVQ exactly")
    train_contents = {by_id[video_id]["content_name"] for video_id in train_ids}
    test_contents = {by_id[video_id]["content_name"] for video_id in test_ids}
    if train_contents & test_contents:
        raise ValueError("Train and test contents overlap")


def development_partition(
    train_ids: Sequence[str],
    by_id: dict[str, dict[str, Any]],
    seed: int,
    val_fraction: float = 0.1,
) -> tuple[list[str], list[str]]:
    contents = sorted({by_id[video_id]["content_name"] for video_id in train_ids})
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(contents, dtype=object)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(contents) * val_fraction)))
    val_contents = set(shuffled[:n_val].tolist())
    fit_ids = [
        video_id
        for video_id in train_ids
        if by_id[video_id]["content_name"] not in val_contents
    ]
    val_ids = [
        video_id
        for video_id in train_ids
        if by_id[video_id]["content_name"] in val_contents
    ]
    if {
        by_id[video_id]["content_name"] for video_id in fit_ids
    } & {by_id[video_id]["content_name"] for video_id in val_ids}:
        raise AssertionError("Development content separation failed")
    return fit_ids, val_ids


def stratified_pilot_ids(
    train_ids: Sequence[str], by_id: dict[str, dict[str, Any]], count: int = 100
) -> list[str]:
    ordered = sorted(train_ids, key=lambda video_id: (by_id[video_id]["mos_j"], video_id))
    positions = np.linspace(0, len(ordered) - 1, num=count, dtype=int)
    selected = [ordered[int(position)] for position in positions]
    if len(set(selected)) != count:
        raise AssertionError("Pilot sampling produced duplicate IDs")
    return selected


def soft_level_target(mos: float, device: torch.device) -> torch.Tensor:
    coordinate = float(np.clip(mos, 0.0, 100.0)) / 25.0
    low = int(math.floor(coordinate))
    high = int(math.ceil(coordinate))
    target = torch.zeros(len(LEVELS), dtype=torch.float32, device=device)
    if low == high:
        target[low] = 1.0
    else:
        target[low] = high - coordinate
        target[high] = coordinate - low
    return target


def level_token_ids(tokenizer: Any) -> list[int]:
    token_ids: list[int] = []
    for level in LEVEL_TOKEN_TEXT:
        encoded = tokenizer.encode(level, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Level {level!r} is not one token: {encoded}")
        token_ids.append(int(encoded[0]))
    if len(set(token_ids)) != len(LEVEL_TOKEN_TEXT):
        raise ValueError(f"Level token IDs are not unique: {token_ids}")
    return token_ids


def frame_paths(
    video_id: str, frames_root: Path | str = DEFAULT_FRAMES, frame_count: int = 8
) -> list[Path]:
    root = Path(frames_root) / video_id
    paths = [root / f"frame_{index:02d}.png" for index in range(frame_count)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing cached frames for {video_id}: {missing[:2]}")
    return paths


def verify_frame_set(paths: Sequence[Path]) -> None:
    for path in paths:
        with Image.open(path) as image:
            image.verify()


def build_messages(paths: Sequence[Path], prompt: str = RATING_PROMPT) -> list[dict[str, Any]]:
    content: list[dict[str, str]] = [
        {"type": "image", "image": str(path)} for path in paths
    ]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def prepare_inputs(
    processor: Any,
    paths: Sequence[Path],
    device: torch.device,
    prompt: str = RATING_PROMPT,
) -> dict[str, torch.Tensor]:
    from qwen_vl_utils import process_vision_info

    messages = build_messages(paths, prompt)
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[rendered],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in inputs.items()}


def prepare_input_batch(
    processor: Any,
    path_batches: Sequence[Sequence[Path]],
    device: torch.device,
    prompt: str = RATING_PROMPT,
) -> dict[str, torch.Tensor]:
    """Prepare a padded batch while preserving each video's image placeholders."""
    from qwen_vl_utils import process_vision_info

    conversations = [build_messages(paths, prompt) for paths in path_batches]
    rendered = [
        processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        for messages in conversations
    ]
    image_inputs, video_inputs = process_vision_info(conversations)
    inputs = processor(
        text=rendered,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in inputs.items()}


def rating_logits(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    token_ids: Sequence[int],
) -> torch.Tensor:
    outputs = model(**inputs, use_cache=False)
    last_index = int(inputs["attention_mask"][0].sum().item()) - 1
    return outputs.logits[0, last_index, list(token_ids)].float()


def batched_rating_logits(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    token_ids: Sequence[int],
) -> torch.Tensor:
    outputs = model(**inputs, use_cache=False)
    positions = torch.arange(
        inputs["attention_mask"].shape[1], device=inputs["attention_mask"].device
    ).unsqueeze(0)
    last_indices = (positions * inputs["attention_mask"].long()).max(dim=1).values
    batch_indices = torch.arange(last_indices.numel(), device=last_indices.device)
    selected = outputs.logits[batch_indices, last_indices]
    return selected[:, list(token_ids)].float()


def score_from_logits(logits: torch.Tensor) -> tuple[float, np.ndarray]:
    probabilities = torch.softmax(logits.float(), dim=-1).detach().cpu().numpy()
    score = float(np.dot(probabilities.astype(np.float64), LEVEL_MOS))
    return score, probabilities


def compute_metrics(mos: Sequence[float], predictions: Sequence[float]) -> dict[str, float]:
    truth = np.asarray(mos, dtype=np.float64)
    pred = np.asarray(predictions, dtype=np.float64)
    if len(truth) < 3 or not np.all(np.isfinite(pred)):
        raise ValueError("Metrics require at least three finite predictions")
    return {
        "srocc": float(spearmanr(truth, pred).statistic),
        "plcc": float(pearsonr(truth, pred).statistic),
        "krcc": float(kendalltau(truth, pred).statistic),
        "rmse": float(np.sqrt(np.mean(np.square(truth - pred)))),
        "n": int(len(truth)),
    }


def atomic_json(path: Path | str, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, destination)


def write_prediction_csv(path: Path | str, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_id",
        "mos",
        "prediction",
        *[f"p_{level}" for level in LEVELS],
        "content_name",
        "bitrate",
        "resolution",
        "orientation",
    ]
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, destination)


def append_run_record(path: Path | str, title: str, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {title}\n\n")
        handle.write("```json\n")
        handle.write(json.dumps(payload, indent=2, sort_keys=True))
        handle.write("\n```\n")


def load_processor(model_name: str, min_pixels: int, max_pixels: int) -> Any:
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        model_name,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )


def load_trainable_model(
    model_name: str,
    rank: int,
    alpha: int,
    dropout: float,
    qlora: bool,
) -> torch.nn.Module:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
        "device_map": {"": 0},
    }
    if qlora:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
    if qlora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    return model


def load_adapter_model(model_name: str, adapter_path: Path | str) -> torch.nn.Module:
    from peft import PeftModel
    from transformers import Qwen2_5_VLForConditionalGeneration

    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False)
    model.eval()
    model.config.use_cache = True
    return model
