"""Native-PQ input construction for the Gemma-4 V2 and V2B paths.

The cache stores 16-bit BT.2020 RGB PQ code values. This module decodes those
values without Pillow and converts them to float32 in [0, 1]. The environment
variable ``BRIGHTRATE_NATIVE_PQ_MODE`` selects how those floats enter Gemma-4:

``v2`` (the default) exactly inverts the processor rescale and bypasses the
learned LayerNorm before ``patch_dense``. ``v2b`` supplies the same floats to
the processor without adjustment and retains the standard processor arithmetic
and learned LayerNorm.

The unified Gemma-4 checkpoint also has a learned LayerNorm immediately before
``patch_dense``. A LayerNorm cannot be algebraically inverted for arbitrary
patches because its outputs are constrained to normalized per-patch moments.
For V2 only, ``configure_native_pq_model`` replaces that one LayerNorm with an
identity. This makes the input seen by ``patch_dense`` the raw PQ patch tensor.
V2B validates that the module is still the checkpoint LayerNorm. V0 and V1
never call this native-PQ module and retain the original architecture.
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torchvision.io import ImageReadMode, decode_image


PQ16_ROOT_NAME = "frames8-d9-pq16"
NATIVE_PQ_MODE_ENV = "BRIGHTRATE_NATIVE_PQ_MODE"
NATIVE_PQ_MODES = ("v2", "v2b")
EXPECTED_FRAME_COUNT = 8
EXPECTED_MAX_EDGE = 448
EXPECTED_PATCH_SIZE = 16
EXPECTED_POOLING_KERNEL_SIZE = 3
EXPECTED_MAX_SOFT_TOKENS = 280
EXPECTED_MODEL_PATCH_SIZE = 48
EXPECTED_PATCH_DIM = 6912
PROCESSOR_RESAMPLE = Image.Resampling.BILINEAR


def is_native_pq_root(frames_root: Path | str) -> bool:
    """Return whether a frame root selects the explicit V2 cache."""
    return Path(frames_root).name == PQ16_ROOT_NAME


def native_pq_mode(mode: str | None = None) -> str:
    """Resolve the native-PQ mode, preserving V2 as the default."""
    resolved = (mode or os.environ.get(NATIVE_PQ_MODE_ENV, "v2")).strip().lower()
    if resolved not in NATIVE_PQ_MODES:
        raise ValueError(
            f"{NATIVE_PQ_MODE_ENV} must be one of {NATIVE_PQ_MODES}, got {resolved!r}"
        )
    return resolved


def _assert_processor_contract(processor: Any) -> None:
    image_processor = processor.image_processor
    observed = {
        "patch_size": image_processor.patch_size,
        "pooling_kernel_size": image_processor.pooling_kernel_size,
        "max_soft_tokens": image_processor.max_soft_tokens,
        "do_rescale": image_processor.do_rescale,
        "do_normalize": image_processor.do_normalize,
        "image_mean": tuple(image_processor.image_mean),
        "image_std": tuple(image_processor.image_std),
    }
    expected = {
        "patch_size": EXPECTED_PATCH_SIZE,
        "pooling_kernel_size": EXPECTED_POOLING_KERNEL_SIZE,
        "max_soft_tokens": EXPECTED_MAX_SOFT_TOKENS,
        "do_rescale": True,
        "do_normalize": False,
        "image_mean": (0.0, 0.0, 0.0),
        "image_std": (1.0, 1.0, 1.0),
    }
    if observed != expected:
        raise ValueError(f"Gemma-4 image processor contract changed: {observed}")
    factor = float(image_processor.rescale_factor)
    if not math.isclose(factor, 1.0 / 255.0, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(f"Expected Gemma-4 rescale factor 1/255, got {factor}")


def expected_processor_size(
    height: int,
    width: int,
    patch_size: int = EXPECTED_PATCH_SIZE,
    max_soft_tokens: int = EXPECTED_MAX_SOFT_TOKENS,
    pooling_kernel_size: int = EXPECTED_POOLING_KERNEL_SIZE,
) -> tuple[int, int]:
    """Reproduce Gemma-4's aspect-ratio-preserving target size arithmetic."""
    max_patches = max_soft_tokens * pooling_kernel_size**2
    target_pixels = max_patches * patch_size**2
    factor = math.sqrt(target_pixels / (height * width))
    side_multiple = pooling_kernel_size * patch_size
    target_height = int(math.floor(factor * height / side_multiple)) * side_multiple
    target_width = int(math.floor(factor * width / side_multiple)) * side_multiple
    if target_height <= 0 or target_width <= 0:
        raise ValueError(f"Invalid Gemma-4 resize target for {height}x{width}")
    if target_height * target_width > target_pixels:
        raise ValueError(
            f"Gemma-4 resize target {target_height}x{target_width} exceeds patch budget"
        )
    return target_height, target_width


@lru_cache(maxsize=4096)
def _read_manifest(directory: str) -> dict[str, Any]:
    path = Path(directory) / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "variant": "native_pq16",
        "bit_depth": 16,
        "color_primaries": "BT.2020",
        "tone_mapping": "none",
        "gamut_conversion": "none",
        "resized_max_edge": EXPECTED_MAX_EDGE,
    }
    for key, value in required.items():
        if payload.get(key) != value:
            raise ValueError(f"Unexpected {key} in {path}: {payload.get(key)!r}")
    if len(payload.get("frame_indices", [])) != EXPECTED_FRAME_COUNT:
        raise ValueError(f"Expected eight frame indices in {path}")
    return payload


def load_pq16_frame(path: Path) -> torch.Tensor:
    """Load one truecolor 16-bit PNG as CHW float32 PQ values in [0, 1]."""
    manifest = _read_manifest(str(path.parent))
    encoded = decode_image(str(path), mode=ImageReadMode.RGB)
    if encoded.dtype != torch.uint16:
        raise TypeError(f"Expected uint16 PNG decode for {path}, got {encoded.dtype}")
    if encoded.ndim != 3 or encoded.shape[0] != 3:
        raise ValueError(f"Expected CHW RGB for {path}, got {tuple(encoded.shape)}")
    expected_width, expected_height = (int(value) for value in manifest["output_size"])
    if tuple(encoded.shape[1:]) != (expected_height, expected_width):
        raise ValueError(
            f"Shape mismatch for {path}: {tuple(encoded.shape[1:])} "
            f"versus manifest {(expected_height, expected_width)}"
        )
    if max(expected_height, expected_width) != EXPECTED_MAX_EDGE:
        raise ValueError(f"Expected a {EXPECTED_MAX_EDGE}px longer side for {path}")
    pq = encoded.to(torch.float32).mul_(1.0 / 65535.0)
    if not torch.isfinite(pq).all():
        raise ValueError(f"Non-finite PQ values in {path}")
    pq_min, pq_max = float(pq.min()), float(pq.max())
    if pq_min < 0.0 or pq_max > 1.0:
        raise ValueError(f"PQ values outside [0,1] in {path}: [{pq_min}, {pq_max}]")
    return pq


def _image_message(paths: Sequence[Path], prompt: str) -> list[dict[str, Any]]:
    content: list[dict[str, str]] = [
        {"type": "image", "image": str(path)} for path in paths
    ]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def prepare_native_pq_batch(
    processor: Any,
    path_batches: Sequence[Sequence[Path]],
    prompt: str,
    device: torch.device,
    mode: str | None = None,
) -> dict[str, Any]:
    """Construct a Gemma-4 batch from float32 PQ values without uint8 conversion.

    V2 passes ``x_arg = x_pq / rescale_factor`` and explicitly uses bilinear
    resize. V2B passes ``x_pq`` unchanged and supplies no image-processing
    overrides, so the checkpoint's normal resize, rescale, normalization, and
    RGB-conversion settings remain active. Target shapes, dtypes, ranges,
    position grids, and padding are checked after processing.
    """
    resolved_mode = native_pq_mode(mode)
    _assert_processor_contract(processor)
    if not path_batches or any(len(paths) != EXPECTED_FRAME_COUNT for paths in path_batches):
        raise ValueError("Native PQ batches require exactly eight frames per sample")

    pq_batches = [[load_pq16_frame(path) for path in paths] for paths in path_batches]
    for frames in pq_batches:
        for frame in frames:
            if frame.dtype != torch.float32:
                raise TypeError(f"Expected float32 native PQ source, got {frame.dtype}")
            if float(frame.min()) < 0.0 or float(frame.max()) > 1.0:
                raise ValueError("Native PQ source must remain in [0,1]")

    factor = float(processor.image_processor.rescale_factor)
    processor_batches = (
        [[frame / factor for frame in frames] for frames in pq_batches]
        if resolved_mode == "v2"
        else pq_batches
    )
    messages = [_image_message(paths, prompt) for paths in path_batches]
    texts = [
        processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        for message in messages
    ]
    processor_kwargs: dict[str, Any] = {}
    if resolved_mode == "v2":
        processor_kwargs = {
            "do_convert_rgb": False,
            "do_resize": True,
            "resample": PROCESSOR_RESAMPLE,
            "do_rescale": True,
            "rescale_factor": factor,
            "do_normalize": False,
        }
    inputs = processor(
        text=texts,
        images=processor_batches,
        padding=True,
        return_tensors="pt",
        **processor_kwargs,
    )
    _assert_processed_batch(inputs, pq_batches, processor, resolved_mode)
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def _assert_processed_batch(
    inputs: Any,
    pq_batches: list[list[torch.Tensor]],
    processor: Any,
    mode: str,
) -> None:
    pixel_values = inputs["pixel_values"]
    position_ids = inputs["image_position_ids"]
    flat_frames = [frame for frames in pq_batches for frame in frames]
    expected_images = len(flat_frames)
    expected_shape = (expected_images, EXPECTED_MAX_SOFT_TOKENS, EXPECTED_PATCH_DIM)
    if tuple(pixel_values.shape) != expected_shape:
        raise ValueError(f"Unexpected native PQ pixel_values shape: {tuple(pixel_values.shape)}")
    if tuple(position_ids.shape) != (expected_images, EXPECTED_MAX_SOFT_TOKENS, 2):
        raise ValueError(f"Unexpected native PQ position-id shape: {tuple(position_ids.shape)}")
    if pixel_values.dtype != torch.float32:
        raise TypeError(f"Expected float32 processor output, got {pixel_values.dtype}")
    if not torch.isfinite(pixel_values).all():
        raise ValueError("Non-finite native PQ processor output")

    epsilon = 2e-6
    for index, source in enumerate(flat_frames):
        height, width = source.shape[-2:]
        target_height, target_width = expected_processor_size(height, width)
        expected_grid = (target_height // EXPECTED_MODEL_PATCH_SIZE, target_width // EXPECTED_MODEL_PATCH_SIZE)
        valid = (position_ids[index] != -1).all(dim=-1)
        expected_valid = expected_grid[0] * expected_grid[1]
        if int(valid.sum()) != expected_valid:
            raise ValueError(
                f"Position grid mismatch for image {index}: {int(valid.sum())} versus {expected_valid}"
            )
        valid_positions = position_ids[index, valid]
        observed_grid = (
            int(valid_positions[:, 1].max()) + 1,
            int(valid_positions[:, 0].max()) + 1,
        )
        if observed_grid != expected_grid:
            raise ValueError(
                f"Position extent mismatch for image {index}: {observed_grid} versus {expected_grid}"
            )
        patches = pixel_values[index, valid]
        source_min, source_max = float(source.min()), float(source.max())
        patch_min, patch_max = float(patches.min()), float(patches.max())
        if mode == "v2":
            if patch_min < max(0.0, source_min - epsilon) or patch_max > min(1.0, source_max + epsilon):
                raise ValueError(
                    f"Native PQ resize escaped source range for image {index}: "
                    f"source=[{source_min}, {source_max}], patches=[{patch_min}, {patch_max}]"
                )
            if patch_min < -epsilon or patch_max > 1.0 + epsilon:
                raise ValueError(f"Native PQ patches outside [0,1]: [{patch_min}, {patch_max}]")
        else:
            # The standard bicubic resize can overshoot a bounded float source.
            # This conservative bound still proves the configured 1/255 rescale
            # ran and catches an accidental V2 inverse-rescale path.
            standard_bound = 2.0 * float(processor.image_processor.rescale_factor)
            if patch_min < -standard_bound or patch_max > standard_bound:
                raise ValueError(
                    f"V2B standard processor output outside expected rescaled range: "
                    f"source=[{source_min}, {source_max}], patches=[{patch_min}, {patch_max}]"
                )
        if torch.count_nonzero(pixel_values[index, ~valid]).item() != 0:
            raise ValueError(f"Nonzero padded PQ patches for image {index}")


def configure_native_pq_model(
    model: torch.nn.Module, mode: str | None = None
) -> list[str]:
    """Apply V2's bypass or validate V2B's standard learned LayerNorm."""
    resolved_mode = native_pq_mode(mode)
    name, module = find_patch_layernorm(model)
    if resolved_mode == "v2b":
        if not isinstance(module, torch.nn.LayerNorm):
            raise TypeError(
                f"V2B requires the standard LayerNorm at {name}, got {type(module).__name__}"
            )
        normalized_shape = tuple(int(value) for value in module.normalized_shape)
        if normalized_shape != (EXPECTED_PATCH_DIM,):
            raise ValueError(f"Unexpected {name} normalized shape: {normalized_shape}")
        return []
    if isinstance(module, torch.nn.Identity):
        return [name]
    if not isinstance(module, torch.nn.LayerNorm):
        raise TypeError(f"Expected LayerNorm at {name}, got {type(module).__name__}")
    normalized_shape = tuple(int(value) for value in module.normalized_shape)
    if normalized_shape != (EXPECTED_PATCH_DIM,):
        raise ValueError(f"Unexpected {name} normalized shape: {normalized_shape}")
    parent_name, attribute = name.rsplit(".", 1)
    parent = model.get_submodule(parent_name)
    setattr(parent, attribute, torch.nn.Identity())
    return [name]


def find_patch_layernorm(model: torch.nn.Module) -> tuple[str, torch.nn.Module]:
    """Find the active Gemma-4 pre-patch projection normalization module."""
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if name.endswith("patch_ln1")
        and ("vision_embedder" in name or "embed_vision" in name)
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one Gemma-4 vision patch_ln1 module, found "
            f"{[name for name, _ in candidates]}"
        )
    return candidates[0]


def find_patch_dense(model: torch.nn.Module) -> tuple[str, torch.nn.Module]:
    """Find the active 6912 to 3840 native patch projection."""
    candidates = []
    for name, module in model.named_modules():
        if not name.endswith("patch_dense"):
            continue
        if not isinstance(module, torch.nn.Linear):
            continue
        if module.in_features == EXPECTED_PATCH_DIM and module.out_features == 3840:
            candidates.append((name, module))
    if len(candidates) != 1:
        raise ValueError(f"Expected one active patch_dense, found {[name for name, _ in candidates]}")
    return candidates[0]


def native_pq_config(mode: str | None = None) -> dict[str, Any]:
    """Return serializable native-PQ input-path metadata for run records."""
    resolved_mode = native_pq_mode(mode)
    common = {
        "mode": resolved_mode,
        "mode_env": NATIVE_PQ_MODE_ENV,
        "storage": "16-bit RGB PNG",
        "code_space": "SMPTE ST 2084 PQ in BT.2020 primaries",
        "cache_quantization_divisor": 65535,
        "processor_clipping": "none",
        "patch_size": EXPECTED_MODEL_PATCH_SIZE,
        "patch_dim": EXPECTED_PATCH_DIM,
        "max_soft_tokens_per_image": EXPECTED_MAX_SOFT_TOKENS,
    }
    if resolved_mode == "v2":
        common.update(
            {
                "processor_rescale_inverse": "x_arg = x_pq / (1/255) = 255*x_pq",
                "processor_resize": "explicit bilinear to asserted Gemma-4 target grid",
                "processor_normalization": "none after exact rescale inversion",
                "model_pre_patch_dense_layernorm": "bypassed because exact inversion is infeasible",
            }
        )
    else:
        common.update(
            {
                "processor_rescale_inverse": "none; float32 x_pq in [0,1] is supplied unchanged",
                "processor_resize": "checkpoint default bicubic",
                "processor_normalization": "checkpoint standard arithmetic, including 1/255 rescale",
                "model_pre_patch_dense_layernorm": "standard learned LayerNorm retained",
            }
        )
    return common


__all__ = [
    "configure_native_pq_model",
    "expected_processor_size",
    "find_patch_dense",
    "find_patch_layernorm",
    "is_native_pq_root",
    "load_pq16_frame",
    "native_pq_config",
    "native_pq_mode",
    "prepare_native_pq_batch",
]
