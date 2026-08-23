#!/usr/bin/env python3
"""Run deterministic five-level scoring with google/gemma-4-12B (encoder-free MLLM)."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--checkpoint", default="google/gemma-4-12B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-image-side", type=int, default=448)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def load_frames(
    variant: str, video_id: str, max_side: int, frames_root: Path
) -> list[Image.Image]:
    frames = []
    for path in frame_paths(variant, video_id, frames_root):
        image = Image.open(path).convert("RGB")
        longest = max(image.size)
        if longest > max_side:
            scale = max_side / float(longest)
            new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            image = image.resize(new_size, Image.LANCZOS)
        frames.append(image)
    return frames


def make_message(frames: list[Image.Image]) -> list[dict]:
    content = [{"type": "image", "image": frame} for frame in frames]
    content.append({"type": "text", "text": PROMPT})
    return [{"role": "user", "content": content}]


def configure_image_size(processor, max_side: int) -> dict:
    """Cap the processor's own resolution control if it exposes one; report what was found."""
    report: dict = {"requested_max_image_side": max_side, "processor_size_control": None}
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return report
    size = getattr(image_processor, "size", None)
    if isinstance(size, dict):
        if "longest_edge" in size:
            size["longest_edge"] = max_side
            report["processor_size_control"] = {"attribute": "size.longest_edge", "value": max_side}
        else:
            report["processor_size_control"] = {"attribute": "size", "value": dict(size)}
    for attr in ("max_image_size", "longest_edge"):
        value = getattr(image_processor, attr, None)
        if isinstance(value, int) and value > max_side:
            setattr(image_processor, attr, max_side)
            report["processor_size_control"] = {"attribute": attr, "value": max_side}
    return report


def level_token_ids(tokenizer) -> list[list[int]]:
    per_level = []
    for level in LEVELS:
        ids = tokenizer.encode(" " + level, add_special_tokens=False)
        if not ids:
            raise ValueError(f"Tokenizer produced no tokens for {' ' + level!r}")
        per_level.append(ids)
    return per_level


def pick_scoring_path(per_level: list[list[int]]) -> str:
    firsts = [ids[0] for ids in per_level]
    if all(len(ids) == 1 for ids in per_level) and len(set(firsts)) == len(firsts):
        return "fast_single_token"
    return "general_teacher_forced"


def extend_text_inputs(inputs: dict, extra_ids: list[int]) -> dict:
    """Append extra text token ids to every text-shaped tensor in a processor batch."""
    input_ids = inputs["input_ids"]
    batch, length = input_ids.shape
    extra = torch.tensor(extra_ids, dtype=input_ids.dtype, device=input_ids.device)
    extra = extra.unsqueeze(0).expand(batch, -1)
    extended = {}
    for key, value in inputs.items():
        if torch.is_tensor(value) and value.dim() == 2 and value.shape == (batch, length):
            if key == "input_ids":
                pad = extra
            elif key == "attention_mask":
                pad = torch.ones_like(extra)
            else:
                # token_type_ids and similar per-token annotations: extra tokens are text.
                pad = torch.zeros_like(extra).to(value.dtype)
            extended[key] = torch.cat([value, pad.to(value.dtype)], dim=1)
        else:
            extended[key] = value
    return extended


def fast_batch_scores(model, inputs, per_level, weights) -> torch.Tensor:
    candidates = [ids[0] for ids in per_level]
    output = model(**inputs, use_cache=False, return_dict=True)
    candidate_logits = output.logits[:, -1, candidates].float()
    return torch.softmax(candidate_logits, dim=-1) @ weights


def general_batch_scores(model, inputs, per_level, weights) -> torch.Tensor:
    batch, length = inputs["input_ids"].shape
    level_logprobs = []
    for ids in per_level:
        extended = extend_text_inputs(inputs, ids)
        output = model(**extended, use_cache=False, return_dict=True)
        # Position length-1+j predicts appended token j (teacher forcing).
        span = output.logits[:, length - 1 : length - 1 + len(ids), :].float()
        logprobs = torch.log_softmax(span, dim=-1)
        targets = torch.tensor(ids, device=span.device).view(1, -1, 1).expand(batch, -1, 1)
        level_logprobs.append(logprobs.gather(-1, targets).squeeze(-1).sum(dim=-1))
        del output, extended, span, logprobs
    stacked = torch.stack(level_logprobs, dim=-1)
    return torch.softmax(stacked, dim=-1) @ weights


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gemma-4 inference")
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    rows = load_metadata(args.metadata)
    ids = target_ids(args.target, rows, args.splits)
    if args.limit:
        ids = ids[: args.limit]

    predictions = load_predictions(args.output)
    unknown = set(predictions).difference(ids)
    if unknown:
        raise ValueError(f"Output contains IDs outside target: {sorted(unknown)[:5]}")

    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        args.checkpoint,
        revision=args.revision,
        cache_dir=str(HF_HOME / "hub"),
    )
    processor.tokenizer.padding_side = "left"
    image_size_report = configure_image_size(processor, args.max_image_side)
    per_level = level_token_ids(processor.tokenizer)
    scoring_path = pick_scoring_path(per_level)
    model = AutoModelForImageTextToText.from_pretrained(
        args.checkpoint,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=str(HF_HOME / "hub"),
    ).eval()
    load_seconds = time.perf_counter() - load_started
    commit = getattr(model.config, "_commit_hash", None)
    device = next(model.parameters()).device
    weights = torch.tensor(LEVEL_VALUES, dtype=torch.float32, device=device)
    torch.cuda.reset_peak_memory_stats()

    missing = [video_id for video_id in ids if video_id not in predictions]
    inference_seconds = 0.0
    completed_new = 0
    run_started = time.perf_counter()
    for offset in range(0, len(missing), args.batch_size):
        batch_ids = missing[offset : offset + args.batch_size]
        batch_frames = [
            load_frames(args.variant, video_id, args.max_image_side, args.frames_root)
            for video_id in batch_ids
        ]
        messages = [make_message(frames) for frames in batch_frames]
        texts = [
            processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            + ANSWER_PREFIX
            for message in messages
        ]
        inputs = processor(
            text=texts,
            images=batch_frames,
            padding=True,
            return_tensors="pt",
        ).to(device)
        if not torch.all(inputs["attention_mask"][:, -1] == 1):
            raise ValueError("The final input position is padded, so answer logits are ambiguous")
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            if scoring_path == "fast_single_token":
                scores_tensor = fast_batch_scores(model, inputs, per_level, weights)
            else:
                scores_tensor = general_batch_scores(model, inputs, per_level, weights)
        torch.cuda.synchronize()
        inference_seconds += time.perf_counter() - started
        scores = scores_tensor.detach().cpu().tolist()
        if len(scores) != len(batch_ids):
            raise ValueError(f"gemma4 returned {len(scores)} scores for {len(batch_ids)} videos")
        new_rows = list(zip(batch_ids, scores))
        append_predictions(args.output, new_rows)
        predictions.update(new_rows)
        completed_new += len(new_rows)
        del inputs, scores_tensor
        if completed_new % 20 == 0 or completed_new == len(missing):
            elapsed = time.perf_counter() - run_started
            print(
                f"gemma4 {args.variant}: {completed_new}/{len(missing)} new, "
                f"{len(predictions)}/{len(ids)} total, elapsed={elapsed:.1f}s",
                flush=True,
            )

    rewrite_predictions(args.output, ids, predictions)
    sidecar = {
        "model": "gemma4-12b",
        "checkpoint": args.checkpoint,
        "requested_revision": args.revision,
        "resolved_revision": commit,
        "variant": args.variant,
        "target": args.target,
        "target_count": len(ids),
        "new_predictions": completed_new,
        "prompt": PROMPT,
        "answer_prefix": ANSWER_PREFIX,
        "levels": list(LEVELS),
        "candidate_token_strings": [" " + level for level in LEVELS],
        "level_values": list(LEVEL_VALUES),
        "level_token_ids": per_level,
        "scoring_path": scoring_path,
        "method": (
            "softmax expectation from next-token logits"
            if scoring_path == "fast_single_token"
            else "softmax expectation over summed teacher-forced continuation log-probs"
        ),
        "batch_size": args.batch_size,
        "max_image_side": args.max_image_side,
        "image_size_report": image_size_report,
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
