#!/usr/bin/env python3
"""Paper-ready qualitative examples for a train_v2 adapter: score + text.

Mirrors analysis/mllm-brightrate/describe.py but generalized like
evaluate_v2.py, and extended to the full output interface of the final model:
for each selected test-side video it emits a JSON record with
  video_id, mos_gt, mos_predicted, predicted_level, description, reasoning
where mos_predicted uses the exact level-token expectation readout of
evaluate_v2.py (same prompt, same token ladder, so the number matches the eval
pipeline), description is a 1-2 sentence account of the concrete quality
problems, and reasoning is a 2-3 sentence justification of the predicted
level. Description and reasoning come from two separate greedy
(deterministic) generation prompts to the same adapter-loaded model, both
conditioned on the frames; the reasoning prompt is additionally given the
predicted quality level word.

Selection is stratified over the test side of the given split across bitrate
rungs (0.2M, 0.5M, 1M, 2M, 3M, ref) and MOS terciles (tercile edges are the
1/3 and 2/3 quantiles of test-side MOS): cells are visited round-robin in
sorted (bitrate, tercile) order, one seeded-shuffled candidate per visit,
skipping already-used content names; any shortfall is filled from the
remaining unused-content videos in seeded order.

Outputs: a quals JSON at --output plus one contact-sheet PNG per video (the 8
sampled frames in a row, small) in --output-dir. For expstack caches the
contact sheet uses the mid-exposure (e0) rendition of each frame.

Shared helpers are imported from train_v2.py / evaluate_v2.py so prompts,
V0/V1 frames-root handling, the Qwen fps fix, the level-token ladder, and
adapter loading (including modules_to_save extras) cannot drift.
"""

from __future__ import annotations

import argparse
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from train_v2 import (
    DEFAULT_FRAMES,
    DEFAULT_RUNS,
    EXPOSURE_TAGS,
    SUPPORTED_MODELS,
    d5,
    detect_family,
    detect_variant,
    load_processor,
    model_input_device,
    prepare_sub_batches,
    resolve_level_tokens,
    variant_prompt,
)
from evaluate_v2 import load_adapter_model

LEVEL_WORDS = ("Bad", "Poor", "Fair", "Good", "Excellent")
BITRATE_ORDER = ("0.2M", "0.5M", "1M", "2M", "3M", "ref")

DESCRIPTION_INSTRUCTION = (
    "In one or two short sentences, describe the overall visual quality of "
    "this video and name the concrete quality problems you can see, such as "
    "blockiness, banding, blur, noise, blown highlights, crushed shadows, "
    "color shifts, or motion artifacts. Comment only on visual quality and "
    "never mention audio. Do not mention that you saw frames."
)
REASONING_TEMPLATE = (
    "The overall quality of this video was rated {level}. In two or three "
    "sentences, explain why a {level} rating is justified: point to the "
    "specific quality problems or strengths that are visible, and say how "
    "severe they are and how much of the video they affect. Comment only on "
    "visual quality and never mention audio. Do not mention that you saw frames."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=int, required=True)
    parser.add_argument("--adapter", required=True, help="Local adapter path or HF model ID")
    parser.add_argument("--output", type=Path, required=True, help="Quals JSON path.")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Contact-sheet PNG directory."
    )
    parser.add_argument("--model", default=SUPPORTED_MODELS[0])
    parser.add_argument("--csv", type=Path, default=d5.DEFAULT_CSV)
    parser.add_argument("--splits", type=Path, default=d5.DEFAULT_SPLITS)
    parser.add_argument("--frames-root", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--runs-md", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--seed", type=int, default=1200)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument(
        "--limit", type=int, help="Process only the first N selected videos."
    )
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--min-pixels", type=int, default=4 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    parser.add_argument(
        "--device-map",
        choices=("auto",),
        help="Load the unquantized bf16 base across all visible GPUs.",
    )
    parser.add_argument("--max-new-tokens-description", type=int, default=80)
    parser.add_argument("--max-new-tokens-reasoning", type=int, default=160)
    parser.add_argument("--sheet-height", type=int, default=128)
    return parser.parse_args()


def clean_text(text: str) -> str:
    """Sanitize generated text and trim a token-cap cut to a full sentence."""
    text = text.replace("\u2014", ",").replace("\u00a7", "").strip()
    last_end = max(text.rfind(mark) for mark in (".", "!", "?"))
    if last_end > 0:
        text = text[: last_end + 1]
    return text.strip()


def stratified_selection(
    test_ids: Sequence[str],
    by_id: dict[str, dict[str, Any]],
    count: int,
    seed: int,
) -> tuple[list[str], dict[str, Any]]:
    """Round-robin over (bitrate rung x MOS tercile) cells, seeded."""
    mos_values = np.asarray([by_id[vid]["mos_j"] for vid in test_ids], dtype=np.float64)
    edges = np.quantile(mos_values, [1.0 / 3.0, 2.0 / 3.0])

    def tercile(mos: float) -> int:
        return int(np.searchsorted(edges, mos, side="right"))

    rng = np.random.default_rng(seed)
    cells: dict[tuple[int, int], list[str]] = {}
    for video_id in sorted(test_ids):
        meta = by_id[video_id]
        bitrate_rank = BITRATE_ORDER.index(meta["bitrate"])
        cells.setdefault((bitrate_rank, tercile(meta["mos_j"])), []).append(video_id)
    for members in cells.values():
        rng.shuffle(members)
    selected: list[str] = []
    used_contents: set[str] = set()
    cell_keys = sorted(cells)
    cursors = {key: 0 for key in cell_keys}
    progressed = True
    while len(selected) < count and progressed:
        progressed = False
        for key in cell_keys:
            if len(selected) == count:
                break
            members = cells[key]
            while cursors[key] < len(members):
                candidate = members[cursors[key]]
                cursors[key] += 1
                if by_id[candidate]["content_name"] in used_contents:
                    continue
                selected.append(candidate)
                used_contents.add(by_id[candidate]["content_name"])
                progressed = True
                break
    if len(selected) < count:
        remainder = [
            vid
            for vid in sorted(test_ids)
            if vid not in selected and by_id[vid]["content_name"] not in used_contents
        ]
        rng.shuffle(remainder)
        for video_id in remainder:
            if len(selected) == count:
                break
            selected.append(video_id)
            used_contents.add(by_id[video_id]["content_name"])
    if len(selected) < count:
        raise RuntimeError(f"Could only select {len(selected)} of {count} examples")
    doc = {
        "strategy": (
            "round-robin over bitrate-rung x MOS-tercile cells, seeded shuffle "
            "within cells, unique content names, shortfall filled from unused "
            "contents in seeded order"
        ),
        "bitrate_order": list(BITRATE_ORDER),
        "mos_tercile_edges": [float(edge) for edge in edges],
        "seed": seed,
    }
    return selected, doc


def sheet_frame_paths(variant: str, frames_root: Path, video_id: str, frame_count: int) -> list[Path]:
    root = Path(frames_root) / video_id
    if variant == "expstack24":
        return [root / f"f{index}_e0.png" for index in range(frame_count)]
    return [root / f"frame_{index:02d}.png" for index in range(frame_count)]


def write_contact_sheet(paths: Sequence[Path], destination: Path, height: int) -> None:
    if destination.exists():
        raise FileExistsError(f"Contact sheet already exists: {destination}")
    thumbnails = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        scale = height / float(image.height)
        thumbnails.append(
            image.resize((max(1, round(image.width * scale)), height), Image.LANCZOS)
        )
    total_width = sum(thumb.width for thumb in thumbnails)
    sheet = Image.new("RGB", (total_width, height), (0, 0, 0))
    offset = 0
    for thumb in thumbnails:
        sheet.paste(thumb, (offset, 0))
        offset += thumb.width
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def generate_text(
    model: torch.nn.Module,
    processor: Any,
    inputs: dict[str, Any],
    max_new_tokens: int,
) -> str:
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
    continuation = generated[:, inputs["input_ids"].shape[1] :]
    text = processor.batch_decode(
        continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return clean_text(text)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Quals output already exists: {args.output}")
    started = time.time()
    family = detect_family(args.model)
    variant = detect_variant(args.frames_root)
    rating_prompt = variant_prompt(family, variant)
    d5.seed_everything(args.seed)
    rows = d5.load_rows(args.csv)
    by_id = d5.rows_by_id(rows)
    payload = d5.load_splits(args.splits)
    split = payload["splits"][args.split]
    d5.validate_split(split, by_id)
    selected, selection_doc = stratified_selection(
        split["test"], by_id, args.count, args.seed
    )
    if args.limit:
        selected = selected[: args.limit]
    processor = load_processor(args.model, family, args)
    token_label, token_strings, token_ids = resolve_level_tokens(processor.tokenizer)
    model = load_adapter_model(
        args.model, args.adapter, device_map=args.device_map
    )
    device = model_input_device(model)

    records: list[dict[str, Any]] = []
    for video_id in tqdm(selected, desc=f"describe split {args.split}"):
        # 1) Score with the exact evaluate_v2 readout (same prompt and tokens).
        inputs, _ = prepare_sub_batches(
            family, variant, processor, args.frames_root, [video_id],
            args.frame_count, rating_prompt, device,
        )[0]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = d5.batched_rating_logits(model, inputs, token_ids)
        score, probabilities = d5.score_from_logits(logits[0])
        level_word = LEVEL_WORDS[int(np.argmax(probabilities))]
        del inputs, logits

        # 2) Description: greedy generation from a separate prompt.
        description_inputs, _ = prepare_sub_batches(
            family, variant, processor, args.frames_root, [video_id],
            args.frame_count, DESCRIPTION_INSTRUCTION, device,
        )[0]
        description = generate_text(
            model, processor, description_inputs, args.max_new_tokens_description
        )
        del description_inputs

        # 3) Reasoning: greedy generation conditioned on the predicted level.
        reasoning_prompt = REASONING_TEMPLATE.format(level=level_word)
        reasoning_inputs, _ = prepare_sub_batches(
            family, variant, processor, args.frames_root, [video_id],
            args.frame_count, reasoning_prompt, device,
        )[0]
        reasoning = generate_text(
            model, processor, reasoning_inputs, args.max_new_tokens_reasoning
        )
        del reasoning_inputs

        sheet_path = args.output_dir / f"{video_id}.png"
        write_contact_sheet(
            sheet_frame_paths(variant, args.frames_root, video_id, args.frame_count),
            sheet_path,
            args.sheet_height,
        )
        metadata = by_id[video_id]
        record = {
            "video_id": video_id,
            "mos_gt": float(metadata["mos_j"]),
            "mos_predicted": score,
            "predicted_level": level_word,
            "description": description,
            "reasoning": reasoning,
            "content_name": metadata["content_name"],
            "bitrate": metadata["bitrate"],
            "resolution": metadata["resolution"],
            "orientation": metadata["orientation"],
            "contact_sheet": str(sheet_path),
        }
        records.append(record)
        print(
            f"{video_id}: pred {score:.1f} ({level_word}) | {description}",
            flush=True,
        )

    result = {
        "split": args.split,
        "adapter": str(args.adapter),
        "model": args.model,
        "model_family": family,
        "frames_root": str(args.frames_root),
        "input_variant": args.frames_root.name,
        "frame_count": args.frame_count,
        "count_requested": args.count,
        "count_generated": len(records),
        "limit": args.limit,
        "seed": args.seed,
        "selection": selection_doc,
        "rating_prompt": rating_prompt,
        "description_prompt": DESCRIPTION_INSTRUCTION,
        "reasoning_prompt_template": REASONING_TEMPLATE,
        "level_token_strategy": token_label,
        "level_token_ids": dict(zip(token_strings, token_ids)),
        "max_new_tokens_description": args.max_new_tokens_description,
        "max_new_tokens_reasoning": args.max_new_tokens_reasoning,
        "decoding": "greedy (do_sample=False, num_beams=1)",
        "output_dir": str(args.output_dir),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "wall_seconds": time.time() - started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    if args.device_map is not None:
        result["device_map"] = args.device_map
    d5.atomic_json(args.output, result)
    d5.append_run_record(
        args.runs_md,
        f"QUALITATIVE split {args.split}",
        {key: value for key, value in result.items() if key != "records"},
    )
    print(f"Wrote {len(records)} records to {args.output}", flush=True)


if __name__ == "__main__":
    main()
