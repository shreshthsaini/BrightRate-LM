#!/usr/bin/env python3
"""Run deterministic five-level next-token scoring with Qwen2.5-VL-7B."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoModelForImageTextToText

from benchmark_io import (
    FRAMES_ROOT,
    HF_HOME,
    LEVELS,
    LEVEL_VALUES,
    append_predictions,
    frame_paths,
    load_metadata,
    load_predictions,
    rewrite_predictions,
    target_ids,
    write_json,
)


PROMPT = (
    "Rate the overall perceptual quality of this video. Consider compression artifacts, "
    "blur, noise, temporal artifacts, and overall visual fidelity. Answer with exactly one "
    "lowercase word from: bad, poor, fair, good, excellent."
)
ANSWER_PREFIX = "The quality of the video is"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["direct", "tonemap"], required=True)
    parser.add_argument("--target", default="all", help="all, train0, test0, or an ID file")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path(__file__).parents[1] / "configs/splits.json",
    )
    parser.add_argument("--frames-root", type=Path, default=FRAMES_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-csv", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-pixels", type=int, default=360 * 420)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()



def video_fps(variant: str, video_id: str, frames_root: Path) -> float:
    paths = frame_paths(variant, video_id, frames_root)
    manifest = json.loads((paths[0].parent / "manifest.json").read_text())
    return round(len(paths) / float(manifest["duration_seconds"]), 6)

def make_message(variant: str, video_id: str, frames_root: Path) -> list[dict]:
    paths = frame_paths(variant, video_id, frames_root)
    manifest_path = paths[0].parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    sampled_fps = len(paths) / float(manifest["duration_seconds"])
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": [path.resolve().as_uri() for path in paths],
                    "fps": sampled_fps,
                },
                {"type": "text", "text": PROMPT},
            ],
        },
        {"role": "assistant", "content": ANSWER_PREFIX},
    ]


def token_ids(tokenizer) -> list[int]:
    result = []
    for level in LEVELS:
        ids = tokenizer.encode(" " + level, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Expected one token for {level!r}, found IDs {ids}")
        result.append(ids[0])
    if len(set(result)) != len(result):
        raise ValueError(f"Candidate token IDs are not unique: {result}")
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen2.5-VL inference")
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    rows = load_metadata(args.metadata)
    ids = target_ids(args.target, rows, args.splits)
    if args.limit:
        ids = ids[: args.limit]

    predictions = load_predictions(args.output)
    if args.initial_csv:
        for video_id, score in load_predictions(args.initial_csv).items():
            if video_id in ids:
                predictions.setdefault(video_id, score)
        rewrite_predictions(args.output, [video_id for video_id in ids if video_id in predictions], predictions)
    unknown = set(predictions).difference(ids)
    if unknown:
        raise ValueError(f"Output contains IDs outside target: {sorted(unknown)[:5]}")

    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        args.checkpoint,
        revision=args.revision,
        min_pixels=64 * 28 * 28,
        max_pixels=args.max_pixels,
        use_fast=False,
        cache_dir=str(HF_HOME / "hub"),
    )
    processor.tokenizer.padding_side = "left"
    candidates = token_ids(processor.tokenizer)
    model = AutoModelForImageTextToText.from_pretrained(
        args.checkpoint,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        cache_dir=str(HF_HOME / "hub"),
    ).eval()
    load_seconds = time.perf_counter() - load_started
    commit = getattr(model.config, "_commit_hash", None)
    device = next(model.parameters()).device
    weights = torch.tensor(LEVEL_VALUES, dtype=torch.float32, device=device)
    torch.cuda.reset_peak_memory_stats()

    missing = [video_id for video_id in ids if video_id not in predictions]
    groups: dict[float, list[str]] = {}
    for vid in missing:
        groups.setdefault(video_fps(args.variant, vid, args.frames_root), []).append(vid)
    batches = []
    for fps_key in sorted(groups):
        group = groups[fps_key]
        for k in range(0, len(group), args.batch_size):
            batches.append(group[k : k + args.batch_size])
    inference_seconds = 0.0
    completed_new = 0
    run_started = time.perf_counter()
    for batch_ids in batches:
        messages = [
            make_message(args.variant, video_id, args.frames_root)
            for video_id in batch_ids
        ]
        texts = [
            processor.apply_chat_template(
                message, tokenize=False, continue_final_message=True
            )
            for message in messages
        ]
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        fps_value = video_kwargs.get("fps")
        if isinstance(fps_value, list):
            if len(set(fps_value)) != 1:
                raise ValueError(f"Mixed fps in batch: {fps_value}")
            video_kwargs = dict(video_kwargs)
            video_kwargs["fps"] = float(fps_value[0])
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(device)
        if not torch.all(inputs.attention_mask[:, -1] == 1):
            raise ValueError("The final input position is padded, so answer logits are ambiguous")
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model(**inputs, use_cache=False, return_dict=True)
            candidate_logits = output.logits[:, -1, candidates].float()
            scores_tensor = torch.softmax(candidate_logits, dim=-1) @ weights
        torch.cuda.synchronize()
        inference_seconds += time.perf_counter() - started
        scores = scores_tensor.detach().cpu().tolist()
        if len(scores) != len(batch_ids):
            raise ValueError(f"Qwen returned {len(scores)} scores for {len(batch_ids)} videos")
        new_rows = list(zip(batch_ids, scores))
        append_predictions(args.output, new_rows)
        predictions.update(new_rows)
        completed_new += len(new_rows)
        del output, inputs, candidate_logits, scores_tensor
        if completed_new % 20 == 0 or completed_new == len(missing):
            elapsed = time.perf_counter() - run_started
            print(
                f"qwen {args.variant}: {completed_new}/{len(missing)} new, "
                f"{len(predictions)}/{len(ids)} total, elapsed={elapsed:.1f}s",
                flush=True,
            )

    rewrite_predictions(args.output, ids, predictions)
    sidecar = {
        "model": args.checkpoint.split("/")[-1],
        "checkpoint": args.checkpoint,
        "requested_revision": args.revision,
        "resolved_revision": commit,
        "variant": args.variant,
        "target": args.target,
        "target_count": len(ids),
        "new_predictions": completed_new,
        "initial_csv": str(args.initial_csv) if args.initial_csv else None,
        "prompt": PROMPT,
        "answer_prefix": ANSWER_PREFIX,
        "levels": list(LEVELS),
        "candidate_token_strings": [" " + level for level in LEVELS],
        "level_values": list(LEVEL_VALUES),
        "candidate_token_ids": candidates,
        "method": "softmax expectation from next-token logits",
        "batch_size": args.batch_size,
        "max_pixels_per_frame": args.max_pixels,
        "model_load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "seconds_per_new_video": inference_seconds / completed_new if completed_new else 0.0,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cuda_device": torch.cuda.get_device_name(0),
    }
    write_json(Path(str(args.output) + ".meta.json"), sidecar)


if __name__ == "__main__":
    main()
