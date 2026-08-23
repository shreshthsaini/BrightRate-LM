#!/usr/bin/env python3
"""Score one HDR video and return a short visual diagnosis."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import build_expstack_cache as exposure_cache
from config import value
from describe_v2 import (
    DESCRIPTION_INSTRUCTION,
    LEVEL_WORDS,
    REASONING_TEMPLATE,
    generate_text,
)
from evaluate_v2 import load_adapter_model
from train_v2 import (
    d5,
    detect_family,
    load_processor,
    model_input_device,
    prepare_sub_batches,
    resolve_level_tokens,
    variant_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--adapter", required=True, help="Local adapter path or HF model ID")
    parser.add_argument("--model", default=value("models", "primary"))
    parser.add_argument("--ffmpeg", default=value("video", "ffmpeg"))
    parser.add_argument("--cache-dir", type=Path, help="Keep rendered frames here")
    parser.add_argument("--device-map", choices=("auto",))
    parser.add_argument("--seed", type=int, default=1200)
    parser.add_argument("--max-new-tokens-description", type=int, default=80)
    parser.add_argument("--max-new-tokens-reasoning", type=int, default=160)
    return parser.parse_args()


def render_video(video: Path, frames_root: Path, ffmpeg: str) -> str:
    """Render the eight-frame, three-exposure interface used during training."""
    exposure_cache.VIDEOS_DIR = video.parent
    exposure_cache.OUT_ROOT = frames_root
    exposure_cache.D5_ROOT = frames_root / "no-reference-cache"
    exposure_cache.FFMPEG = ffmpeg
    video_id, _, _ = exposure_cache.process_video(video.stem)
    return video_id


def main() -> None:
    args = parse_args()
    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    if video.suffix.lower() != ".mp4":
        raise ValueError("The public demo currently accepts MP4 video")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for BrightRate-LM inference")

    temporary = None
    if args.cache_dir:
        frames_root = args.cache_dir.expanduser().resolve()
        frames_root.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="brightrate-lm-")
        frames_root = Path(temporary.name) / "frames8-expstack"
        frames_root.mkdir(parents=True)

    try:
        video_id = render_video(video, frames_root, args.ffmpeg)
        d5.seed_everything(args.seed)
        family = detect_family(args.model)
        variant = "expstack24"
        processor_args = SimpleNamespace(
            min_pixels=4 * 28 * 28,
            max_pixels=int(value("video", "max_edge")) ** 2,
        )
        processor = load_processor(args.model, family, processor_args)
        _, _, token_ids = resolve_level_tokens(processor.tokenizer)
        model = load_adapter_model(
            args.model, args.adapter, device_map=args.device_map
        )
        device = model_input_device(model)

        rating_inputs, _ = prepare_sub_batches(
            family,
            variant,
            processor,
            frames_root,
            [video_id],
            int(value("video", "frame_count")),
            variant_prompt(family, variant),
            device,
        )[0]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = d5.batched_rating_logits(model, rating_inputs, token_ids)
        score, probabilities = d5.score_from_logits(logits[0])
        level = LEVEL_WORDS[int(np.argmax(probabilities))]
        del rating_inputs, logits

        description_inputs, _ = prepare_sub_batches(
            family,
            variant,
            processor,
            frames_root,
            [video_id],
            int(value("video", "frame_count")),
            DESCRIPTION_INSTRUCTION,
            device,
        )[0]
        description = generate_text(
            model,
            processor,
            description_inputs,
            args.max_new_tokens_description,
        )
        del description_inputs

        reasoning_inputs, _ = prepare_sub_batches(
            family,
            variant,
            processor,
            frames_root,
            [video_id],
            int(value("video", "frame_count")),
            REASONING_TEMPLATE.format(level=level),
            device,
        )[0]
        reasoning = generate_text(
            model,
            processor,
            reasoning_inputs,
            args.max_new_tokens_reasoning,
        )
        result = {
            "video": str(video),
            "score": round(score, 2),
            "level": level,
            "description": description,
            "reasoning": reasoning,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
