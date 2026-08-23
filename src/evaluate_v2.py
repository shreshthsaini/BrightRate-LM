#!/usr/bin/env python3
"""Model-agnostic evaluation of a train_v2 LoRA adapter on a BrightVQ test split.

Mirrors analysis/mllm-brightrate/evaluate.py (5-level expectation readout on
every test-side video, predictions CSV plus a metrics json with the same
schema) but generalized exactly like train_v2.py:
  - --model: any supported checkpoint, loaded via
    AutoModelForImageTextToText + AutoProcessor (transformers 5.x).
  - Same V0/V1 --frames-root handling: frames8 caches use the qwen_vl_utils
    video path with the fps-homogeneous scalar-fps fix for the Qwen family and
    8 images per sample for gemma-4; expstack caches feed 24 images per sample
    in temporal-major order f0_em2, f0_e0, f0_ep2, f1_em2, ...
  - Same runtime level-token ladder as training.
  - --adapter loads a PEFT dir; PeftModel.from_pretrained also restores
    modules_to_save extras (for example the gemma-4 patch-embed modules saved
    by --train-patch-embed) from the adapter payload.
  - --device-map auto loads the unquantized bf16 base across visible GPUs and
    sends processor outputs to the first dispatched device.

Input construction, prompts, and token resolution are imported from
train_v2.py so evaluation cannot drift from training.
"""

from __future__ import annotations

import argparse
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from train_v2 import (
    DEFAULT_FRAMES,
    DEFAULT_RUNS,
    HF_HOME,
    SUPPORTED_MODELS,
    d5,
    detect_family,
    detect_variant,
    load_processor,
    model_input_device,
    prepare_sub_batches,
    register_gemma4_e_lora_modules,
    resolve_level_tokens,
    variant_prompt,
)
from native_pq_input import (
    configure_native_pq_model,
    native_pq_config,
    native_pq_mode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=int, required=True)
    parser.add_argument("--adapter", required=True, help="Local adapter path or HF model ID")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=SUPPORTED_MODELS[0])
    parser.add_argument("--csv", type=Path, default=d5.DEFAULT_CSV)
    parser.add_argument("--splits", type=Path, default=d5.DEFAULT_SPLITS)
    parser.add_argument("--frames-root", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--runs-md", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--seed", type=int, default=900)
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--min-pixels", type=int, default=4 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    loading_group = parser.add_mutually_exclusive_group()
    loading_group.add_argument(
        "--qlora",
        action="store_true",
        help="Load the base model in deterministic 4-bit NF4 for a low-memory smoke evaluation.",
    )
    loading_group.add_argument(
        "--device-map",
        choices=("auto",),
        help="Load the unquantized bf16 base across all visible GPUs.",
    )
    parser.add_argument(
        "--max-gpu-memory-gib",
        type=float,
        help="Hard CUDA allocator limit for a shared-GPU smoke evaluation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N test videos (smoke tests only; the "
        "metrics json records the limit).",
    )
    return parser.parse_args()


def load_adapter_model(
    model_name: str,
    adapter_path: str | Path,
    pq_mode: str | None = None,
    qlora: bool = False,
    device_map: str | None = None,
) -> torch.nn.Module:
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

    kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
        "device_map": device_map or {"": 0},
        "cache_dir": str(HF_HOME / "hub"),
    }
    if qlora:
        skip_modules = ["lm_head"]
        if pq_mode is not None:
            skip_modules.extend(["patch_dense", "embedding_projection"])
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=skip_modules,
        )
    base = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
    if pq_mode is not None:
        configure_native_pq_model(base, pq_mode)
    peft_config = PeftConfig.from_pretrained(str(adapter_path))
    register_gemma4_e_lora_modules(peft_config, model_name)
    model = PeftModel.from_pretrained(
        base,
        str(adapter_path),
        is_trainable=False,
        config=peft_config,
    )
    model.eval()
    model.config.use_cache = True
    return model


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 3:
        raise ValueError("limit must be at least 3 (metrics need 3 predictions)")
    if args.max_gpu_memory_gib is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for evaluation")
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocation_limit = args.max_gpu_memory_gib * 2**30
        if allocation_limit <= 0 or allocation_limit >= total_memory:
            raise ValueError(
                "max-gpu-memory-gib must be positive and smaller than total GPU memory"
            )
        torch.cuda.set_per_process_memory_fraction(allocation_limit / total_memory, device=0)
    family = detect_family(args.model)
    variant = detect_variant(args.frames_root)
    pq_mode = native_pq_mode() if variant == "native_pq8" else None
    if variant == "native_pq8" and family != "gemma4":
        raise ValueError("The native PQ input path is specific to Gemma-4")
    prompt = variant_prompt(family, variant)
    d5.seed_everything(args.seed)
    rows = d5.load_rows(args.csv)
    by_id = d5.rows_by_id(rows)
    payload = d5.load_splits(args.splits)
    split = payload["splits"][args.split]
    d5.validate_split(split, by_id)
    test_ids = list(split["test"])
    if args.limit:
        test_ids = test_ids[: args.limit]
    processor = load_processor(args.model, family, args)
    token_label, token_strings, token_ids = resolve_level_tokens(processor.tokenizer)
    model = load_adapter_model(
        args.model,
        args.adapter,
        pq_mode=pq_mode,
        qlora=args.qlora,
        device_map=args.device_map,
    )
    device = model_input_device(model)
    started = time.time()
    prediction_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for video_id in tqdm(test_ids, desc=f"evaluate split {args.split}"):
            sub_batches = prepare_sub_batches(
                family, variant, processor, args.frames_root, [video_id],
                args.frame_count, prompt, device,
            )
            inputs, _ = sub_batches[0]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = d5.batched_rating_logits(model, inputs, token_ids)
            score, probabilities = d5.score_from_logits(logits[0])
            metadata = by_id[video_id]
            prediction_rows.append(
                {
                    "video_id": video_id,
                    "mos": metadata["mos_j"],
                    "prediction": score,
                    **{
                        f"p_{level}": float(probability)
                        for level, probability in zip(d5.LEVELS, probabilities)
                    },
                    "content_name": metadata["content_name"],
                    "bitrate": metadata["bitrate"],
                    "resolution": metadata["resolution"],
                    "orientation": metadata["orientation"],
                }
            )
            del inputs, logits
    metrics = d5.compute_metrics(
        [row["mos"] for row in prediction_rows],
        [row["prediction"] for row in prediction_rows],
    )
    d5.write_prediction_csv(args.output, prediction_rows)
    result = {
        "split": args.split,
        "adapter": str(args.adapter),
        "output": str(args.output),
        "model": args.model,
        "model_family": family,
        "frames_root": str(args.frames_root),
        "input_variant": args.frames_root.name,
        "frame_count": args.frame_count,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "qlora": args.qlora,
        "seed": args.seed,
        "limit": args.limit,
        "prompt": prompt,
        "level_token_strategy": token_label,
        "level_token_ids": dict(zip(token_strings, token_ids)),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "metrics": metrics,
        "wall_seconds": time.time() - started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    if args.device_map is not None:
        result["device_map"] = args.device_map
    if variant == "native_pq8":
        result["native_pq"] = native_pq_config(pq_mode)
    d5.atomic_json(args.output.with_suffix(".metrics.json"), result)
    d5.append_run_record(args.runs_md, f"EVAL split {args.split}", result)
    print(result)


if __name__ == "__main__":
    main()
