#!/usr/bin/env python3
"""Build the V1 multi-exposure stack frame cache for BrightVQ HDR10 videos.

For each video, extract the same 8 uniformly sampled frames as the existing
frames8-d5-tonemap448 cache (identical indices, taken from the d5 manifests,
with the d5 probe + interval-center rule as fallback), then render each frame
at three exposures (-2, 0, +2 stops) with scientifically explicit HDR math:

  1. Decode: ffmpeg select + zscale (lanczos resize to 448 px longer side,
     done on PQ-encoded values) to planar float32 RGB (gbrpf32le rawvideo).
     Input is tagged explicitly as BT.2020 ncl / SMPTE ST 2084 / limited range.
  2. PQ to linear light via the exact SMPTE ST 2084 EOTF (m1, m2, c1, c2, c3),
     normalized so 1.0 equals 10000 nits.
  3. Exposure: linear light is anchored so that at s=0 stops the 203 nit
     HDR reference white (ITU-R BT.2408) maps to 0.18 scene linear
     (mid-gray); shifting by s stops multiplies by 2**s.
  4. Tone curve: Hable filmic (Uncharted 2 operator, A=0.15 B=0.50 C=0.10
     D=0.20 E=0.02 F=0.30), exposure_bias=2.0, normalized by the curve value
     at white_point=11.2, applied per channel in BT.2020 linear.
  5. BT.2020 -> BT.709 linear gamut matrix, clip to [0,1], sRGB OETF,
     quantize to 8-bit PNG.

Output layout (matches frames8-d5-tonemap448 conventions):
  <out_root>/<video_hash>/f<k>_e{m2,0,p2}.png  for k in 0..7
  <out_root>/<video_hash>/manifest.json

Resumable: a video is skipped when its dir holds all 24 PNGs + manifest.
Per-video failures append to <out_root>/errors.log and never crash the run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

from config import repo_path, value

FFMPEG = str(value("video", "ffmpeg"))
VIDEOS_DIR = repo_path("videos")
OUT_ROOT = repo_path("multiexposure_frames")
D5_ROOT = repo_path("sdr_frames")

MAX_EDGE = int(value("video", "max_edge"))
N_FRAMES = int(value("video", "frame_count"))
EXPOSURES = [(-2, "m2"), (0, "0"), (2, "p2")]
SAMPLING_NOTE = "centers of eight equal intervals over estimated frame count"

# SMPTE ST 2084 (PQ) constants.
PQ_M1 = 2610.0 / 16384.0            # 0.1593017578125
PQ_M2 = 2523.0 / 4096.0 * 128.0     # 78.84375
PQ_C1 = 3424.0 / 4096.0             # 0.8359375
PQ_C2 = 2413.0 / 4096.0 * 32.0      # 18.8515625
PQ_C3 = 2392.0 / 4096.0 * 32.0      # 18.6875
PQ_PEAK_NITS = 10000.0

REFERENCE_WHITE_NITS = 203.0        # ITU-R BT.2408 HDR reference white
MID_GRAY = 0.18
HABLE_EXPOSURE_BIAS = 2.0
HABLE_WHITE_POINT = 11.2

# Linear BT.2020 -> linear BT.709 primaries conversion (D65).
M_2020_TO_709 = np.array(
    [
        [1.6604910, -0.5876411, -0.0728499],
        [-0.1245505, 1.1328999, -0.0083494],
        [-0.0181508, -0.1005789, 1.1187297],
    ],
    dtype=np.float64,
)

DURATION_RE = re.compile(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)")
FPS_RE = re.compile(r"(?:,| )\s*(\d+(?:\.\d+)?) fps(?:,| )")
DIMS_RE = re.compile(r"\b(\d{3,5})x(\d{2,5})\b")


def pq_eotf(encoded: np.ndarray) -> np.ndarray:
    """SMPTE ST 2084 EOTF: PQ code value in [0,1] to linear light, 1.0 = 10000 nits."""
    e = np.clip(encoded.astype(np.float64), 0.0, 1.0)
    ep = np.power(e, 1.0 / PQ_M2)
    num = np.maximum(ep - PQ_C1, 0.0)
    den = PQ_C2 - PQ_C3 * ep
    return np.power(num / den, 1.0 / PQ_M1)


def hable_curve(x: np.ndarray | float):
    """Uncharted 2 filmic curve (John Hable), unnormalized."""
    a, b, c, d, e, f = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
    return ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f


def hable_tonemap(linear: np.ndarray) -> np.ndarray:
    curr = hable_curve(HABLE_EXPOSURE_BIAS * linear)
    white = hable_curve(HABLE_EXPOSURE_BIAS * HABLE_WHITE_POINT)
    return curr / white


def srgb_oetf(linear: np.ndarray) -> np.ndarray:
    lin = np.clip(linear, 0.0, 1.0)
    low = 12.92 * lin
    high = 1.055 * np.power(np.maximum(lin, 1e-12), 1.0 / 2.4) - 0.055
    return np.where(lin <= 0.0031308, low, high)


def probe(video: Path) -> tuple[float, float, int, int]:
    """Return (duration_seconds, fps, width, height). Same parsing as extract_frames.py."""
    result = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", str(video)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    match = DURATION_RE.search(result.stderr)
    if not match:
        raise RuntimeError(f"Could not parse duration for {video}: {result.stderr[-800:]}")
    hours, minutes, seconds = match.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    video_lines = [line for line in result.stderr.splitlines() if " Video: " in line]
    if not video_lines:
        raise RuntimeError(f"No video stream line for {video}")
    fps_match = FPS_RE.search(video_lines[0])
    if not fps_match:
        raise RuntimeError(f"Could not parse fps for {video}: {video_lines[0]}")
    fps = float(fps_match.group(1))
    dims_match = DIMS_RE.search(video_lines[0])
    if not dims_match:
        raise RuntimeError(f"Could not parse dimensions for {video}: {video_lines[0]}")
    width, height = int(dims_match.group(1)), int(dims_match.group(2))
    if not math.isfinite(duration) or duration <= 0 or fps <= 0:
        raise ValueError(f"Invalid probe values for {video}: duration={duration}, fps={fps}")
    return duration, fps, width, height


def frame_indices(duration: float, fps: float, count: int = N_FRAMES) -> list[int]:
    """Identical interval-center rule to mllm-baselines/scripts/extract_frames.py."""
    estimated_frames = max(count, int(math.floor(duration * fps)))
    indices = [
        int(math.floor((2 * index + 1) * estimated_frames / (2 * count)))
        for index in range(count)
    ]
    indices = [min(value, estimated_frames - 1) for value in indices]
    if len(set(indices)) != count:
        raise ValueError(f"Non-unique indices from duration={duration}, fps={fps}")
    return indices


def target_size(width: int, height: int) -> tuple[int, int]:
    if width >= height:
        return MAX_EDGE, max(1, round(height * MAX_EDGE / width))
    return max(1, round(width * MAX_EDGE / height)), MAX_EDGE


def expected_files() -> list[str]:
    return [f"f{k}_e{tag}.png" for k in range(N_FRAMES) for _, tag in EXPOSURES]


def cache_is_complete(directory: Path) -> bool:
    if not (directory / "manifest.json").is_file():
        return False
    return all((directory / name).is_file() for name in expected_files())


def decode_frames(video: Path, indices: list[int], out_w: int, out_h: int, raw_path: Path) -> np.ndarray:
    """Decode selected frames to float32 RGB (N,H,W,3), PQ-encoded, via zscale."""
    select = "+".join(f"eq(n\\,{index})" for index in indices)
    # Resize happens on PQ-encoded values with lanczos; input colorimetry is
    # tagged explicitly so untagged streams still decode as HDR10.
    graph = (
        f"select={select},"
        f"zscale=w={out_w}:h={out_h}:filter=lanczos:"
        "min=2020_ncl:pin=2020:tin=smpte2084:rin=limited,"
        "format=gbrpf32le"
    )
    command = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-threads", "2",
        "-i", str(video),
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", graph,
        "-fps_mode", "vfr",
        "-frames:v", str(len(indices)),
        "-f", "rawvideo", "-pix_fmt", "gbrpf32le",
        "-y", str(raw_path),
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    frame_bytes = out_w * out_h * 3 * 4
    size = raw_path.stat().st_size if raw_path.is_file() else 0
    if result.returncode != 0 or size != frame_bytes * len(indices):
        raise RuntimeError(
            f"ffmpeg decode failed: rc={result.returncode}, "
            f"bytes={size}/{frame_bytes * len(indices)}, stderr={result.stderr[-1500:]}"
        )
    data = np.fromfile(raw_path, dtype=np.float32).reshape(len(indices), 3, out_h, out_w)
    # gbrp plane order is G, B, R.
    return np.stack([data[:, 2], data[:, 0], data[:, 1]], axis=-1)


def render_exposures(pq_rgb: np.ndarray) -> dict[str, np.ndarray]:
    """PQ-encoded RGB (H,W,3) to 8-bit sRGB renditions per exposure tag."""
    linear = pq_eotf(pq_rgb)                       # 1.0 = 10000 nits
    nits = linear * PQ_PEAK_NITS
    out = {}
    for stops, tag in EXPOSURES:
        scene = nits * (MID_GRAY * (2.0 ** stops) / REFERENCE_WHITE_NITS)
        toned = hable_tonemap(scene)               # per channel, BT.2020 linear
        rgb709 = np.einsum("ij,hwj->hwi", M_2020_TO_709, toned)
        srgb = srgb_oetf(np.clip(rgb709, 0.0, 1.0))
        out[tag] = np.round(srgb * 255.0).astype(np.uint8)
    return out


def process_video(video_id: str) -> tuple[str, str, float]:
    start = time.perf_counter()
    out_dir = OUT_ROOT / video_id
    if cache_is_complete(out_dir):
        return video_id, "cached", time.perf_counter() - start

    video = VIDEOS_DIR / f"{video_id}.mp4"
    if not video.is_file():
        raise FileNotFoundError(video)

    duration, fps, width, height = probe(video)

    # Reuse the d5 cache manifest so frame indices are identical by
    # construction; fall back to the identical probe + center rule.
    d5_manifest = D5_ROOT / video_id / "manifest.json"
    indices_source = "recomputed (d5 rule)"
    if d5_manifest.is_file():
        d5 = json.loads(d5_manifest.read_text())
        indices = [int(v) for v in d5["frame_indices"]]
        duration = float(d5["duration_seconds"])
        fps = float(d5["fps"])
        indices_source = "frames8-d5-tonemap448 manifest"
    else:
        indices = frame_indices(duration, fps)

    out_w, out_h = target_size(width, height)

    tmp = out_dir.with_name(f"{video_id}.tmp.{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    raw_path = tmp / "decode.gbrpf32le"
    try:
        frames = decode_frames(video, indices, out_w, out_h, raw_path)
        raw_path.unlink()
        for k in range(N_FRAMES):
            renders = render_exposures(frames[k])
            for _, tag in EXPOSURES:
                Image.fromarray(renders[tag], mode="RGB").save(
                    tmp / f"f{k}_e{tag}.png", compress_level=3
                )
        manifest = {
            "video": video_id,
            "source": str(video),
            "source_size": video.stat().st_size,
            "variant": "expstack",
            "duration_seconds": duration,
            "fps": fps,
            "frame_indices": indices,
            "frame_indices_source": indices_source,
            "sampling": SAMPLING_NOTE,
            "exposure_stops": [stops for stops, _ in EXPOSURES],
            "exposure_tags": {str(stops): tag for stops, tag in EXPOSURES},
            "source_dimensions": [width, height],
            "output_size": [out_w, out_h],
            "resized_max_edge": MAX_EDGE,
            "resize": "ffmpeg zscale lanczos on PQ-encoded values",
            "decode": "zscale min=2020_ncl pin=2020 tin=smpte2084 rin=limited -> gbrpf32le float",
            "eotf": "SMPTE ST 2084 (PQ), normalized 1.0 = 10000 nits",
            "exposure_anchor": "203 nits (BT.2408 reference white) -> 0.18 scene linear at 0 stops",
            "tone_curve": (
                "Hable filmic (Uncharted 2; A=0.15 B=0.50 C=0.10 D=0.20 E=0.02 F=0.30), "
                "exposure_bias=2.0, white_point=11.2, per channel in BT.2020 linear"
            ),
            "gamut": "linear BT.2020 -> BT.709 matrix, clip to [0,1]",
            "oetf": "sRGB (IEC 61966-2-1), quantized to 8-bit",
            "ffmpeg": FFMPEG,
        }
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if out_dir.exists():
            shutil.rmtree(out_dir)
        os.replace(tmp, out_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return video_id, "built", time.perf_counter() - start


def log_error(message: str) -> None:
    with open(OUT_ROOT / "errors.log", "a") as handle:
        handle.write(message + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", type=Path, default=VIDEOS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUT_ROOT)
    parser.add_argument(
        "--reference-cache",
        type=Path,
        default=D5_ROOT,
        help="Optional SDR cache whose manifests define matching frame indices.",
    )
    parser.add_argument("--ffmpeg", default=FFMPEG)
    parser.add_argument("--video", type=Path, help="Render one video file.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids", nargs="*", help="explicit video ids (hashes) to process")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    global VIDEOS_DIR, OUT_ROOT, D5_ROOT, FFMPEG
    args = parse_args()
    VIDEOS_DIR = args.video.parent.resolve() if args.video else args.videos_dir.resolve()
    OUT_ROOT = args.output_dir.resolve()
    D5_ROOT = args.reference_cache.resolve()
    FFMPEG = args.ffmpeg
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.video:
        if args.ids:
            raise ValueError("--video and --ids cannot be used together")
        ids = [args.video.stem]
    elif args.ids:
        ids = list(args.ids)
    else:
        ids = sorted(path.stem for path in VIDEOS_DIR.glob("*.mp4"))
    if args.limit:
        ids = ids[: args.limit]

    started = time.perf_counter()
    counts = {"cached": 0, "built": 0, "failed": 0}
    print(f"expstack cache build: {len(ids)} videos, {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_video, vid): vid for vid in ids}
        for done, future in enumerate(as_completed(futures), start=1):
            video_id = futures[future]
            try:
                _, status, seconds = future.result()
                counts[status] += 1
            except Exception as error:
                counts["failed"] += 1
                log_error(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {video_id} {type(error).__name__}: {error}")
            if done % args.progress_every == 0 or done == len(ids):
                elapsed = time.perf_counter() - started
                rate = done / elapsed if elapsed > 0 else 0.0
                remaining = (len(ids) - done) / rate if rate > 0 else float("inf")
                print(
                    f"{done}/{len(ids)} cached={counts['cached']} built={counts['built']} "
                    f"failed={counts['failed']} elapsed={elapsed:.0f}s eta={remaining:.0f}s",
                    flush=True,
                )
    print(
        f"done: cached={counts['cached']} built={counts['built']} failed={counts['failed']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
