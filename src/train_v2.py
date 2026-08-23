#!/usr/bin/env python3
"""Model-agnostic LoRA SFT trainer for the BrightRate journal extension (D9).

Extends analysis/mllm-brightrate/train.py (the proven D5 Qwen2.5-VL trainer)
with the same CLI and semantics plus:
  - --model: Qwen/Qwen2.5-VL-7B-Instruct, Qwen/Qwen3-VL-{2B,4B,8B}-Instruct,
    google/gemma-4-12B-it, google/gemma-4-{E2B,E4B}-it, or the large
    google/gemma-4-{26B-A4B,31B}-it checkpoints, loaded via
    AutoModelForImageTextToText and AutoProcessor (transformers 5.x).
  - --frames-root selects the input variant cache:
      frames8-d5-tonemap448 (V0): 8 tonemapped PNGs per video. Qwen family
      consumes them through the qwen_vl_utils VIDEO path with the
      transformers-5.x fps fix (fps-homogeneous processor calls, scalar fps).
      Gemma-4 consumes them as 8 images per sample (infer_gemma4 convention).
      frames8-d9-expstack (V1): 24 PNGs per video f<k>_e{m2,0,p2}, fed to all
      models as 24 images in temporal-major order
      f0_em2, f0_e0, f0_ep2, f1_em2, ...
  - Per-optimizer-step tracking in run_dir/steps.jsonl and an optional quick
    SROCC curve in run_dir/eval_curve.jsonl (--eval-every-updates).
  - LoRA target modules resolved per architecture and recorded in config.json.
  - --train-patch-embed: for gemma-4, additionally trains the encoder-free
    multimodal input embedding/projection (vision_embedder.patch_dense and
    embed_vision.embedding_projection) via PEFT modules_to_save.
  - --device-map auto: loads an unquantized bf16 base across the visible GPUs.
    Processor outputs start on the first dispatched device and Accelerate hooks
    move activations between the remaining devices.
  - frames8-d9-pq16 (V2): loads true 16-bit PQ code values through
    native_pq_input.py, exactly inverts the image processor rescale, and
    bypasses Gemma-4's pre-dense patch LayerNorm so patch_dense receives raw
    PQ patches. V0 and V1 do not take this path.
  - BRIGHTRATE_NATIVE_PQ_MODE=v2b: uses the same PQ16 float32 source but keeps
    Gemma-4's standard processor arithmetic and learned pre-dense LayerNorm.
    The unset/default mode remains V2.

Determinism, resumability stance, and FileExistsError behavior are identical
to train.py: a run directory, epoch adapter, or durable adapter that already
exists raises FileExistsError; there is no mid-run resume.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from config import repo_path, value

HF_HOME = repo_path("hf_cache")
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from native_pq_input import (
    configure_native_pq_model,
    is_native_pq_root,
    native_pq_config,
    native_pq_mode,
    prepare_native_pq_batch,
)


import common as d5

PROJECT_ROOT = d5.PROJECT_ROOT
DEFAULT_FRAMES = repo_path("sdr_frames")
DEFAULT_CHECKPOINTS = repo_path("checkpoints")
DEFAULT_ADAPTERS = repo_path("adapters")
DEFAULT_RUNS = repo_path("runs")
DEFAULT_MODEL = value("models", "primary")

SUPPORTED_MODELS = (
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "google/gemma-4-12B-it",
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
    "google/gemma-4-26B-A4B-it",
    "google/gemma-4-31B-it",
)
QWEN_FAMILIES = {"qwen2_5_vl", "qwen3_vl"}
EXPOSURE_TAGS = ("m2", "0", "p2")

# Attention + MLP projection Linears. These leaf names exist only in the
# language decoder stacks of Qwen3-VL (vision uses qkv/proj/linear_fc*) and
# gemma-4 unified (vision_embedder uses patch_dense), and reproduce the exact
# proven D5 behavior for Qwen2.5-VL. The concrete matched module list is
# expanded at load time and written into config.json.
LORA_TARGET_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

PROMPT_V0_VIDEO = (
    "This is one HDR user-generated video, shown as eight uniformly sampled "
    "frames in temporal order. Judge only perceptual visual quality, including "
    "compression, blur, noise, exposure, color, and temporal consistency. Rate "
    "the overall quality with exactly one word: Bad, Poor, Fair, Good, or "
    "Excellent."
)
PROMPT_V0_IMAGES = d5.RATING_PROMPT
PROMPT_V1_IMAGES = (
    "These images show eight uniformly sampled frames in temporal order from "
    "one HDR user-generated video. Each frame appears three times in a row, "
    "rendered at -2, 0, and +2 stops of exposure, so both dark and bright "
    "regions are visible. Judge only perceptual visual quality, including "
    "compression, blur, noise, exposure, color, and temporal consistency. Rate "
    "the overall quality with exactly one word: Bad, Poor, Fair, Good, or "
    "Excellent."
)

V1_ORDER_NOTE = (
    "24 images per sample in temporal-major order: "
    "f0_em2, f0_e0, f0_ep2, f1_em2, f1_e0, f1_ep2, ..., f7_em2, f7_e0, f7_ep2 "
    "(exposures -2, 0, +2 stops nested inside each of the 8 sampled frames)"
)

CANDIDATE_TOKEN_SETS = (
    ("title_no_space", ("Bad", "Poor", "Fair", "Good", "Excellent")),
    ("title_leading_space", (" Bad", " Poor", " Fair", " Good", " Excellent")),
    ("lower_no_space", ("bad", "poor", "fair", "good", "excellent")),
    ("lower_leading_space", (" bad", " poor", " fair", " good", " excellent")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pilot", "development", "final"), required=True)
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--csv", type=Path, default=d5.DEFAULT_CSV)
    parser.add_argument("--splits", type=Path, default=d5.DEFAULT_SPLITS)
    parser.add_argument("--frames-root", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--adapter-root", type=Path, default=DEFAULT_ADAPTERS)
    parser.add_argument("--runs-md", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--seed", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=value("training", "epochs"))
    parser.add_argument(
        "--schedule-epochs",
        type=int,
        help="Cosine schedule horizon. Defaults to the number of training epochs.",
    )
    parser.add_argument("--pilot-count", type=int, default=100)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--learning-rate", type=float, default=value("training", "learning_rate")
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=value("training", "gradient_accumulation"),
    )
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=value("training", "lora_rank"))
    parser.add_argument("--alpha", type=int, default=value("training", "lora_alpha"))
    parser.add_argument(
        "--dropout", type=float, default=value("training", "lora_dropout")
    )
    parser.add_argument("--frame-count", type=int, default=value("video", "frame_count"))
    parser.add_argument("--min-pixels", type=int, default=4 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    loading_group = parser.add_mutually_exclusive_group()
    loading_group.add_argument("--qlora", action="store_true")
    loading_group.add_argument(
        "--device-map",
        choices=("auto",),
        help="Load the unquantized bf16 base across all visible GPUs.",
    )
    parser.add_argument(
        "--max-gpu-memory-gib",
        type=float,
        help="Hard CUDA allocator limit for shared-GPU pilots. Omit for normal spool runs.",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--eval-every-updates",
        type=int,
        help="Quick SROCC on a fixed dev subset every N optimizer steps, "
        "appended to run_dir/eval_curve.jsonl.",
    )
    parser.add_argument(
        "--dev-subset-size",
        type=int,
        default=64,
        help="Size of the fixed dev subset used by --eval-every-updates.",
    )
    parser.add_argument(
        "--train-patch-embed",
        action="store_true",
        help="Additionally train the multimodal input embedding/projection "
        "modules (gemma-4 encoder-free patch embedding; Qwen visual patch "
        "embed). Default off.",
    )
    return parser.parse_args()


def detect_family(model_name: str) -> str:
    lowered = model_name.lower()
    if "qwen2.5-vl" in lowered or "qwen2_5" in lowered:
        return "qwen2_5_vl"
    if "qwen3-vl" in lowered:
        return "qwen3_vl"
    if "gemma-4" in lowered or "gemma4" in lowered:
        return "gemma4"
    raise ValueError(
        f"Unsupported model {model_name!r}; expected one of {SUPPORTED_MODELS}"
    )


def is_gemma4_e_series(model_name: str) -> bool:
    """Return whether this is a Gemma-4 E-series checkpoint.

    The E2B and E4B checkpoints use Gemma4ClippableLinear in multimodal
    projection stacks. The 12B unified checkpoint does not need this extension.
    """
    checkpoint_name = model_name.rsplit("/", 1)[-1].lower()
    return re.fullmatch(r"gemma-4-e\d+b(?:-.+)?", checkpoint_name) is not None


def needs_gemma4_clippable_lora(model_name: str) -> bool:
    """Return whether PEFT must adapt Gemma4ClippableLinear wrappers."""
    checkpoint_name = model_name.rsplit("/", 1)[-1].lower()
    return (
        re.fullmatch(
            r"gemma-4-(?:e\d+b|26b-a4b|31b)(?:-.+)?", checkpoint_name
        )
        is not None
    )


def register_gemma4_e_lora_modules(lora_config: Any, model_name: str) -> bool:
    """Register PEFT support for Gemma-4's clippable Linear wrapper.

    PEFT's custom-module mapping is attached to a LoraConfig at runtime and is
    not serialized. Call this for both adapter creation and adapter loading.
    The custom layer adapts the wrapped nn.Linear while preserving the
    original input and output clipping around the base-plus-LoRA computation.
    The legacy function name is retained because existing evaluation code
    imports it. Large dense and MoE Gemma-4 checkpoints use the same wrapper as
    the E-series checkpoints and therefore need the same registration.
    """
    if not needs_gemma4_clippable_lora(model_name):
        return False

    from peft.tuners.lora.layer import Linear as PeftLoraLinear
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear

    register = getattr(lora_config, "_register_custom_module", None)
    if not callable(register):
        raise RuntimeError(
            "Installed PEFT does not expose LoraConfig._register_custom_module; "
            "Gemma-4 clippable LoRA requires PEFT custom-module support"
        )

    class Gemma4ClippableLoraLinear(PeftLoraLinear):
        """LoRA for the inner Linear with Gemma-4 clipping kept intact."""

        def __init__(
            self,
            base_layer: Gemma4ClippableLinear,
            adapter_name: str,
            config: Any,
            **kwargs: Any,
        ) -> None:
            if not isinstance(base_layer.linear, torch.nn.Linear):
                raise TypeError(
                    "Gemma4ClippableLinear.linear must be torch.nn.Linear, got "
                    f"{type(base_layer.linear).__name__}"
                )
            use_clipping = bool(base_layer.use_clipped_linears)
            clip_buffers = {
                name: getattr(base_layer, name).detach().clone()
                for name in ("input_min", "input_max", "output_min", "output_max")
                if use_clipping
            }
            clip_buffer_persistence = {
                name: name not in base_layer._non_persistent_buffers_set
                for name in clip_buffers
            }
            super().__init__(
                base_layer.linear,
                adapter_name,
                config=config,
                **kwargs,
            )
            self.use_clipped_linears = use_clipping
            for name, value in clip_buffers.items():
                self.register_buffer(
                    name,
                    value,
                    persistent=clip_buffer_persistence[name],
                )

        def forward(
            self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any
        ) -> torch.Tensor:
            if self.use_clipped_linears:
                hidden_states = torch.clamp(
                    hidden_states, self.input_min, self.input_max
                )
            hidden_states = super().forward(hidden_states, *args, **kwargs)
            if self.use_clipped_linears:
                hidden_states = torch.clamp(
                    hidden_states, self.output_min, self.output_max
                )
            return hidden_states

    register({Gemma4ClippableLinear: Gemma4ClippableLoraLinear})
    return True


def detect_variant(frames_root: Path) -> str:
    if is_native_pq_root(frames_root):
        return "native_pq8"
    return "expstack24" if "expstack" in frames_root.name else "frames8"


def sample_frame_paths(
    variant: str, frames_root: Path, video_id: str, frame_count: int
) -> list[Path]:
    root = Path(frames_root) / video_id
    if variant == "expstack24":
        paths = [
            root / f"f{index}_e{tag}.png"
            for index in range(frame_count)
            for tag in EXPOSURE_TAGS
        ]
    else:
        paths = [root / f"frame_{index:02d}.png" for index in range(frame_count)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing cached frames for {video_id}: {missing[:2]}")
    return paths


_FPS_CACHE: dict[tuple[str, str], float] = {}


def sampled_fps(frames_root: Path, video_id: str, frame_count: int) -> float:
    """Effective fps of the uniform 8-frame sample, from the cache manifest."""
    key = (str(frames_root), video_id)
    if key not in _FPS_CACHE:
        manifest_path = Path(frames_root) / video_id / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _FPS_CACHE[key] = round(frame_count / float(manifest["duration_seconds"]), 6)
    return _FPS_CACHE[key]


def variant_prompt(family: str, variant: str) -> str:
    if variant == "expstack24":
        return PROMPT_V1_IMAGES
    if family in QWEN_FAMILIES:
        return PROMPT_V0_VIDEO
    return PROMPT_V0_IMAGES


def resolve_level_tokens(tokenizer: Any) -> tuple[str, list[str], list[int]]:
    """Pick the first candidate spelling where all 5 levels are single tokens."""
    for label, strings in CANDIDATE_TOKEN_SETS:
        token_ids: list[int] = []
        usable = True
        for text in strings:
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if len(encoded) != 1:
                usable = False
                break
            token_ids.append(int(encoded[0]))
        if usable and len(set(token_ids)) == len(token_ids):
            return label, list(strings), token_ids
    raise ValueError("No single-token level spelling found for this tokenizer")


def load_processor(model_name: str, family: str, args: argparse.Namespace) -> Any:
    from transformers import AutoProcessor

    kwargs: dict[str, Any] = {"cache_dir": str(HF_HOME / "hub")}
    if family in QWEN_FAMILIES:
        # Proven D9 zero-shot conventions for transformers 5.x Qwen processors.
        kwargs.update(
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            use_fast=False,
        )
    processor = AutoProcessor.from_pretrained(model_name, **kwargs)
    processor.tokenizer.padding_side = "right"
    return processor


def resolve_lora_modules(model: torch.nn.Module) -> list[str]:
    resolved = [
        name
        for name, module in model.named_modules()
        if name.split(".")[-1] in LORA_TARGET_SUFFIXES
        and "Linear" in type(module).__name__
    ]
    if not resolved:
        raise ValueError("No LoRA target modules resolved for this architecture")
    leaves = {name.split(".")[-1] for name in resolved}
    missing = set(LORA_TARGET_SUFFIXES) - leaves
    if missing:
        raise ValueError(f"LoRA suffixes missing in this architecture: {sorted(missing)}")
    return sorted(resolved)


def resolve_patch_embed_modules(model: torch.nn.Module, family: str) -> list[str]:
    """Full module names of the multimodal input embedding/projection stack.

    gemma-4 (encoder-free): the vision patch embedding Linear (patch_dense)
    and the vision-to-LM embedding projection, found under the embed_vision /
    vision_embedder branch so the audio projection is excluded. Qwen family:
    the visual patch_embed module.
    """
    if family == "gemma4":
        resolved = sorted(
            name
            for name, module in model.named_modules()
            if ("embed_vision" in name or "vision_embedder" in name)
            and name.split(".")[-1] in ("patch_dense", "embedding_projection")
            and "Linear" in type(module).__name__
        )
    else:
        resolved = sorted(
            name
            for name, _ in model.named_modules()
            if name.endswith("visual.patch_embed")
        )
    if not resolved:
        raise ValueError(f"No patch-embed modules found for family {family!r}")
    return resolved


def load_trainable_model(
    model_name: str, family: str, args: argparse.Namespace
) -> tuple[torch.nn.Module, list[str], list[str], list[str]]:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

    kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
        "device_map": args.device_map or {"": 0},
        "cache_dir": str(HF_HOME / "hub"),
    }
    if args.qlora:
        quantization_kwargs: dict[str, Any] = {}
        if family == "gemma4" and args.train_patch_embed:
            # modules_to_save must remain floating point so PEFT can train the
            # full native patch projection and vision-to-LM projection. The
            # frozen language-model Linears still use NF4.
            quantization_kwargs["llm_int8_skip_modules"] = [
                "lm_head",
                "patch_dense",
                "embedding_projection",
            ]
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            **quantization_kwargs,
        )
    model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
    native_pq_bypassed_modules: list[str] = []
    if is_native_pq_root(args.frames_root):
        native_pq_bypassed_modules = configure_native_pq_model(
            model, native_pq_mode()
        )
    if args.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    resolved_modules = resolve_lora_modules(model)
    patch_embed_modules: list[str] = []
    if args.train_patch_embed:
        patch_embed_modules = resolve_patch_embed_modules(model, family)
        for module_name in patch_embed_modules:
            module = model.get_submodule(module_name)
            non_floating = [
                name
                for name, parameter in module.named_parameters()
                if not parameter.dtype.is_floating_point
            ]
            if non_floating:
                raise TypeError(
                    f"Trainable patch module {module_name} was quantized: {non_floating}"
                )
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(LORA_TARGET_SUFFIXES),
        modules_to_save=patch_embed_modules or None,
    )
    register_gemma4_e_lora_modules(lora_config, model_name)
    model = get_peft_model(model, lora_config)
    return model, resolved_modules, patch_embed_modules, native_pq_bypassed_modules


def model_input_device(model: torch.nn.Module) -> torch.device:
    """Return the first CUDA device in a Transformers dispatch map.

    PEFT forwards most base-model attributes, but the explicit walk keeps this
    helper usable with both a bare Transformers model and a wrapped PeftModel.
    The single-GPU fallback exactly matches the pre-sharding behavior.
    """
    candidates = [
        model,
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
    ]
    for candidate in candidates:
        device_map = getattr(candidate, "hf_device_map", None)
        if not isinstance(device_map, dict):
            continue
        for mapped_device in device_map.values():
            if isinstance(mapped_device, int):
                device = torch.device(f"cuda:{mapped_device}")
            else:
                try:
                    device = torch.device(mapped_device)
                except (TypeError, RuntimeError):
                    continue
            if device.type == "cuda":
                return device
    return torch.device("cuda:0")


def build_qwen_video_message(
    paths: Sequence[Path], fps: float, prompt: str
) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": [path.resolve().as_uri() for path in paths],
                    "fps": fps,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def build_image_message(paths: Sequence[Path], prompt: str) -> list[dict[str, Any]]:
    content: list[dict[str, str]] = [
        {"type": "image", "image": str(path)} for path in paths
    ]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _to_device(inputs: Any, device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def prepare_qwen_video_batch(
    processor: Any,
    path_batches: Sequence[Sequence[Path]],
    fps_values: Sequence[float],
    prompt: str,
    device: torch.device,
) -> dict[str, Any]:
    from qwen_vl_utils import process_vision_info

    if len(set(fps_values)) != 1:
        raise ValueError(f"fps-heterogeneous Qwen video batch: {fps_values}")
    messages = [
        build_qwen_video_message(paths, fps, prompt)
        for paths, fps in zip(path_batches, fps_values)
    ]
    texts = [
        processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        for message in messages
    ]
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )
    # transformers-5.x fps fix: the processor requires one scalar fps per call,
    # so batches are fps-homogeneous and the list is collapsed to a scalar.
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
    )
    return _to_device(inputs, device)


def prepare_qwen_image_batch(
    processor: Any,
    path_batches: Sequence[Sequence[Path]],
    prompt: str,
    device: torch.device,
) -> dict[str, Any]:
    from qwen_vl_utils import process_vision_info

    conversations = [build_image_message(paths, prompt) for paths in path_batches]
    texts = [
        processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        for message in conversations
    ]
    image_inputs, video_inputs = process_vision_info(conversations)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return _to_device(inputs, device)


def prepare_gemma_image_batch(
    processor: Any,
    path_batches: Sequence[Sequence[Path]],
    prompt: str,
    device: torch.device,
) -> dict[str, Any]:
    frame_batches = [
        [Image.open(path).convert("RGB") for path in paths] for paths in path_batches
    ]
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": frame} for frame in frames
                ]
                + [{"type": "text", "text": prompt}],
            }
        ]
        for frames in frame_batches
    ]
    texts = [
        processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        for message in messages
    ]
    inputs = processor(
        text=texts,
        images=frame_batches,
        padding=True,
        return_tensors="pt",
    )
    return _to_device(inputs, device)


def prepare_sub_batches(
    family: str,
    variant: str,
    processor: Any,
    frames_root: Path,
    video_ids: Sequence[str],
    frame_count: int,
    prompt: str,
    device: torch.device,
) -> list[tuple[dict[str, Any], list[str]]]:
    """Build processor batches for one microbatch.

    Qwen + frames8 uses the video path; a microbatch is split into
    fps-homogeneous sub-forwards (transformers-5.x fps fix). All other
    combinations produce exactly one sub-batch of image inputs.
    """
    if family in QWEN_FAMILIES and variant == "frames8":
        groups: dict[float, list[str]] = {}
        for video_id in video_ids:
            fps = sampled_fps(frames_root, video_id, frame_count)
            groups.setdefault(fps, []).append(video_id)
        sub_batches = []
        for fps in sorted(groups):
            group = groups[fps]
            paths = [
                sample_frame_paths(variant, frames_root, video_id, frame_count)
                for video_id in group
            ]
            inputs = prepare_qwen_video_batch(
                processor, paths, [fps] * len(group), prompt, device
            )
            sub_batches.append((inputs, group))
        return sub_batches
    paths = [
        sample_frame_paths(variant, frames_root, video_id, frame_count)
        for video_id in video_ids
    ]
    if variant == "native_pq8":
        inputs = prepare_native_pq_batch(processor, paths, prompt, device)
    elif family in QWEN_FAMILIES:
        inputs = prepare_qwen_image_batch(processor, paths, prompt, device)
    else:
        inputs = prepare_gemma_image_batch(processor, paths, prompt, device)
    return [(inputs, list(video_ids))]


def evaluate_ids(
    model: torch.nn.Module,
    processor: Any,
    family: str,
    variant: str,
    ids: Sequence[str],
    by_id: dict[str, dict[str, Any]],
    token_ids: Sequence[int],
    frames_root: Path,
    frame_count: int,
    prompt: str,
    label: str,
) -> dict[str, float]:
    model.eval()
    truth: list[float] = []
    predictions: list[float] = []
    device = model_input_device(model)
    with torch.inference_mode():
        for video_id in tqdm(ids, desc=label, leave=False):
            sub_batches = prepare_sub_batches(
                family, variant, processor, frames_root, [video_id], frame_count,
                prompt, device,
            )
            inputs, _ = sub_batches[0]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = d5.batched_rating_logits(model, inputs, token_ids)
            score, _ = d5.score_from_logits(logits[0])
            truth.append(float(by_id[video_id]["mos_j"]))
            predictions.append(score)
            del inputs, logits
    model.train()
    return d5.compute_metrics(truth, predictions)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def save_epoch_adapter(model: torch.nn.Module, directory: Path) -> None:
    if directory.exists():
        raise FileExistsError(f"Checkpoint already exists: {directory}")
    directory.mkdir(parents=True)
    model.save_pretrained(directory, safe_serialization=True)


def copy_durable_adapter(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Durable adapter already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)


def main() -> None:
    from transformers import get_cosine_schedule_with_warmup

    args = parse_args()
    if args.micro_batch < 1 or args.grad_accum < 1:
        raise ValueError("micro-batch and grad-accum must both be positive")
    schedule_epochs = args.schedule_epochs or args.epochs
    if schedule_epochs < args.epochs:
        raise ValueError("schedule-epochs cannot be smaller than epochs")
    if args.split < 0 or args.split > 4:
        raise ValueError("D9 only evaluates splits 0 through 4")
    if args.eval_every_updates is not None and args.eval_every_updates < 1:
        raise ValueError("eval-every-updates must be positive")
    if args.max_gpu_memory_gib is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for training")
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
    if variant == "native_pq8" and not args.train_patch_embed:
        raise ValueError("The native PQ training path requires --train-patch-embed")
    prompt = variant_prompt(family, variant)
    images_per_sample = (
        args.frame_count * len(EXPOSURE_TAGS)
        if variant == "expstack24"
        else args.frame_count
    )
    d5.seed_everything(args.seed)
    rows = d5.load_rows(args.csv)
    by_id = d5.rows_by_id(rows)
    split_payload = d5.load_splits(args.splits)
    split = split_payload["splits"][args.split]
    d5.validate_split(split, by_id)

    if args.mode == "pilot":
        train_ids = d5.stratified_pilot_ids(split["train"], by_id, args.pilot_count)
        selection_ids = train_ids
        selection_label = "pilot_train"
    elif args.mode == "development":
        train_ids, selection_ids = d5.development_partition(
            split["train"], by_id, args.seed, args.val_fraction
        )
        selection_label = "development"
    else:
        train_ids = list(split["train"])
        selection_ids = []
        selection_label = "fixed_epoch"

    uses_video_path = family in QWEN_FAMILIES and variant == "frames8"
    for video_id in set(train_ids) | set(selection_ids):
        sample_frame_paths(variant, args.frames_root, video_id, args.frame_count)
        if uses_video_path:
            sampled_fps(args.frames_root, video_id, args.frame_count)

    dev_ids: list[str] = []
    if args.eval_every_updates:
        dev_pool = selection_ids if selection_ids else train_ids
        dev_ids = d5.stratified_pilot_ids(
            dev_pool, by_id, min(args.dev_subset_size, len(dev_pool))
        )

    run_dir = args.checkpoint_root / args.run_name
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    steps_path = run_dir / "steps.jsonl"
    eval_curve_path = run_dir / "eval_curve.jsonl"
    started_wall = time.time()
    started_iso = datetime.now(timezone.utc).isoformat()
    input_layout = (
        V1_ORDER_NOTE
        if variant == "expstack24"
        else (
            (
                "8 truecolor 16-bit PNGs decoded to float32 PQ in [0,1], "
                "passed unchanged through Gemma-4's standard processor "
                "arithmetic and learned pre-patch_dense LayerNorm"
                if pq_mode == "v2b"
                else
                "8 truecolor 16-bit PNGs decoded to float32 PQ in [0,1], "
                "explicit bilinear Gemma-4 grid resize, processor rescale "
                "exactly inverted, and pre-patch_dense LayerNorm bypassed"
            )
            if variant == "native_pq8"
            else (
            "8 frames as one video via qwen_vl_utils, fps from manifest "
            "duration; fps-homogeneous processor calls with scalar fps "
            "(transformers-5.x fix)"
            if uses_video_path
            else "8 frames as 8 images per sample in temporal order"
            )
        )
    )
    config_args = vars(args).copy()
    if args.device_map is None:
        config_args.pop("device_map")
    config = {
        **config_args,
        "trainer": "train_v2",
        "csv": str(args.csv),
        "splits": str(args.splits),
        "frames_root": str(args.frames_root),
        "checkpoint_root": str(args.checkpoint_root),
        "adapter_root": str(args.adapter_root),
        "runs_md": str(args.runs_md),
        "run_dir": str(run_dir),
        "model_family": family,
        "model_supported": args.model in SUPPORTED_MODELS,
        "input_variant": args.frames_root.name,
        "input_layout": input_layout,
        "images_per_sample": images_per_sample,
        "prompt": prompt,
        "train_count": len(train_ids),
        "selection_count": len(selection_ids),
        "selection_label": selection_label,
        "dev_subset_count": len(dev_ids),
        "started_utc": started_iso,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "levels": list(d5.LEVELS),
        "soft_label_knots_mos": [0, 25, 50, 75, 100],
        "effective_batch_size": args.micro_batch * args.grad_accum,
        "schedule_epochs": schedule_epochs,
    }
    if variant == "native_pq8":
        config["native_pq"] = native_pq_config(pq_mode)
    if args.qlora and family == "gemma4" and args.train_patch_embed:
        config["qlora_modules_not_quantized"] = [
            "lm_head",
            "patch_dense",
            "embedding_projection",
        ]
    d5.atomic_json(run_dir / "config.json", config)
    d5.atomic_json(run_dir / "train_ids.json", train_ids)
    d5.atomic_json(run_dir / "selection_ids.json", selection_ids)
    if dev_ids:
        d5.atomic_json(run_dir / "dev_ids.json", dev_ids)
    d5.append_run_record(args.runs_md, f"START {args.run_name}", config)

    import transformers

    processor = load_processor(args.model, family, args)
    token_label, token_strings, token_ids = resolve_level_tokens(processor.tokenizer)
    config["transformers"] = transformers.__version__
    config["level_token_strategy"] = token_label
    config["level_token_strings"] = token_strings
    config["level_token_ids"] = dict(zip(token_strings, token_ids))
    d5.atomic_json(run_dir / "config.json", config)
    model, resolved_modules, patch_embed_modules, native_pq_bypassed_modules = load_trainable_model(
        args.model, family, args
    )
    model.print_trainable_parameters()
    config["lora_target_suffixes"] = list(LORA_TARGET_SUFFIXES)
    config["lora_resolved_modules"] = resolved_modules
    config["gemma4_e_custom_lora"] = is_gemma4_e_series(args.model)
    if needs_gemma4_clippable_lora(args.model) and not is_gemma4_e_series(
        args.model
    ):
        config["gemma4_clippable_custom_lora"] = True
    config["patch_embed_trainable_modules"] = patch_embed_modules
    config["native_pq_bypassed_modules"] = native_pq_bypassed_modules
    d5.atomic_json(run_dir / "config.json", config)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    batches_per_epoch = math.ceil(len(train_ids) / args.micro_batch)
    updates_per_epoch = math.ceil(batches_per_epoch / args.grad_accum)
    total_updates = updates_per_epoch * schedule_epochs
    warmup_steps = int(round(total_updates * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_updates
    )
    device = model_input_device(model)

    history: list[dict[str, Any]] = []
    best_metric = -float("inf")
    best_epoch = args.epochs if args.mode == "final" else -1
    best_path: Path | None = None
    global_update = 0
    samples_seen_total = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.monotonic()
        order = list(train_ids)
        random_generator = np.random.default_rng(args.seed + epoch)
        random_generator.shuffle(order)
        running_loss = 0.0
        optimizer_steps = 0
        model.train()
        batches = [
            order[start : start + args.micro_batch]
            for start in range(0, len(order), args.micro_batch)
        ]
        seen_samples = 0
        step_loss_sum = 0.0
        step_sample_count = 0
        for batch_index, video_ids in enumerate(
            tqdm(batches, desc=f"train epoch {epoch}/{args.epochs}"), start=1
        ):
            sub_batches = prepare_sub_batches(
                family, variant, processor, args.frames_root, video_ids,
                args.frame_count, prompt, device,
            )
            microbatch_size = len(video_ids)
            for inputs, sub_ids in sub_batches:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = d5.batched_rating_logits(model, inputs, token_ids)
                    target = torch.stack(
                        [
                            d5.soft_level_target(by_id[video_id]["mos_j"], logits.device)
                            for video_id in sub_ids
                        ]
                    )
                    per_sample_loss = -(
                        target * torch.log_softmax(logits, dim=-1)
                    ).sum(dim=-1)
                    # sum/(microbatch*grad_accum) equals train.py's
                    # mean/grad_accum once all fps sub-forwards accumulate.
                    scaled_loss = per_sample_loss.sum() / (
                        microbatch_size * args.grad_accum
                    )
                scaled_loss.backward()
                loss_sum = float(per_sample_loss.detach().sum().cpu())
                running_loss += loss_sum
                step_loss_sum += loss_sum
                del inputs, logits, target, per_sample_loss, scaled_loss
            seen_samples += microbatch_size
            step_sample_count += microbatch_size
            samples_seen_total += microbatch_size
            should_step = batch_index % args.grad_accum == 0 or batch_index == len(batches)
            if should_step:
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                global_update += 1
                step_record = {
                    "global_update": global_update,
                    "epoch": epoch,
                    "samples_seen": samples_seen_total,
                    "instant_loss": step_loss_sum / step_sample_count,
                    "running_mean_loss": running_loss / seen_samples,
                    "lr": scheduler.get_last_lr()[0],
                    "max_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "wall_seconds_since_start": time.time() - started_wall,
                }
                append_jsonl(steps_path, step_record)
                step_loss_sum = 0.0
                step_sample_count = 0
                if global_update % args.log_every == 0:
                    print(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "global_update": global_update,
                                "loss": running_loss / seen_samples,
                                "lr": scheduler.get_last_lr()[0],
                                "max_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                if (
                    args.eval_every_updates
                    and global_update % args.eval_every_updates == 0
                ):
                    dev_metrics = evaluate_ids(
                        model, processor, family, variant, dev_ids, by_id,
                        token_ids, args.frames_root, args.frame_count, prompt,
                        f"dev eval @ update {global_update}",
                    )
                    append_jsonl(
                        eval_curve_path,
                        {
                            "global_update": global_update,
                            "epoch": epoch,
                            "srocc": dev_metrics["srocc"],
                            "plcc": dev_metrics["plcc"],
                            "rmse": dev_metrics["rmse"],
                            "n": dev_metrics["n"],
                            "wall_seconds_since_start": time.time() - started_wall,
                        },
                    )

        epoch_path = run_dir / f"epoch_{epoch:02d}"
        save_epoch_adapter(model, epoch_path)
        record: dict[str, Any] = {
            "epoch": epoch,
            "mean_train_loss": running_loss / len(order),
            "optimizer_steps": optimizer_steps,
            "wall_seconds": time.monotonic() - epoch_started,
            "adapter": str(epoch_path),
        }
        if selection_ids:
            selection_metrics = evaluate_ids(
                model, processor, family, variant, selection_ids, by_id,
                token_ids, args.frames_root, args.frame_count, prompt,
                f"{selection_label} epoch {epoch}",
            )
            record["selection_metrics"] = selection_metrics
            selection_metric = selection_metrics["srocc"]
        else:
            selection_metric = float(epoch == args.epochs)
        history.append(record)
        d5.atomic_json(run_dir / "history.json", history)
        if selection_metric > best_metric:
            best_metric = selection_metric
            best_epoch = epoch
            best_path = epoch_path
        print(json.dumps(record, sort_keys=True), flush=True)

    if best_path is None:
        raise AssertionError("No adapter was selected")
    (run_dir / "best_adapter.txt").write_text(str(best_path) + "\n", encoding="utf-8")
    durable_path: Path | None = None
    if args.mode in {"development", "final"}:
        durable_path = args.adapter_root / args.run_name
        copy_durable_adapter(best_path, durable_path)
        d5.atomic_json(
            durable_path / "d9_provenance.json",
            {
                "run_name": args.run_name,
                "mode": args.mode,
                "split": args.split,
                "seed": args.seed,
                "selected_epoch": best_epoch,
                "source_checkpoint": str(best_path),
                "base_model": args.model,
                "model_family": family,
                "frames_root": str(args.frames_root),
                "input_variant": args.frames_root.name,
                "input_layout": input_layout,
                "train_patch_embed": args.train_patch_embed,
            },
        )

    completed = {
        "run_name": args.run_name,
        "mode": args.mode,
        "split": args.split,
        "seed": args.seed,
        "model": args.model,
        "model_family": family,
        "input_variant": args.frames_root.name,
        "best_epoch": best_epoch,
        "best_selection_srocc": best_metric if selection_ids else None,
        "best_checkpoint": str(best_path),
        "durable_adapter": str(durable_path) if durable_path else None,
        "history": history,
        "wall_seconds": time.time() - started_wall,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "max_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    d5.atomic_json(run_dir / "completed.json", completed)
    d5.append_run_record(args.runs_md, f"DONE {args.run_name}", completed)


if __name__ == "__main__":
    main()
