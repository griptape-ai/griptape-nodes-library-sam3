"""Reversible LTX canvas preparation. No model, engine or GPU imports."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CanvasPlan:
    original_width: int
    original_height: int
    width: int
    height: int
    content_width: int
    content_height: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def widget_dict(self) -> dict[str, int]:
        return {
            "width": self.width,
            "height": self.height,
            "contentWidth": self.content_width,
            "contentHeight": self.content_height,
            "padLeft": self.pad_left,
            "padTop": self.pad_top,
            "padRight": self.pad_right,
            "padBottom": self.pad_bottom,
        }


def plan_canvas(
    original_width: int, original_height: int, canvas_width: int = 1920, canvas_height: int = 1088
) -> CanvasPlan:
    """Fit, never crop/stretch. Center padding with any odd extra pixel right/bottom."""
    values = (original_width, original_height, canvas_width, canvas_height)
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in values):
        raise ValueError("Source and canvas dimensions must be positive integers.")
    if canvas_width % 32 or canvas_height % 32:
        raise ValueError("LTX canvas width and height must be divisible by 32.")
    if canvas_width > 1920 or canvas_height > 1088:
        raise ValueError("This Alpha-Gen profile supports a canvas up to 1920 × 1088.")
    scale = min(canvas_width / original_width, canvas_height / original_height)
    w = min(canvas_width, max(1, math.floor(original_width * scale + 0.5)))
    h = min(canvas_height, max(1, math.floor(original_height * scale + 0.5)))
    left, top = (canvas_width - w) // 2, (canvas_height - h) // 2
    return CanvasPlan(
        original_width,
        original_height,
        canvas_width,
        canvas_height,
        w,
        h,
        left,
        top,
        canvas_width - w - left,
        canvas_height - h - top,
    )


def _resize(array: Any, size: tuple[int, int], *, hard: bool = False) -> Any:
    import numpy as np
    from PIL import Image

    a = np.asarray(array)
    if (a.shape[1], a.shape[0]) == size:
        return a.copy()
    method = Image.Resampling.NEAREST if hard else Image.Resampling.BILINEAR
    if a.dtype == np.uint8:
        return np.asarray(Image.fromarray(a).resize(size, method)).copy()
    if a.ndim == 3:
        return np.stack([_resize(a[..., k], size, hard=hard) for k in range(a.shape[2])], -1)
    # Pillow F mode avoids quantizing floating point alpha through an 8-bit image.
    return np.asarray(Image.fromarray(a.astype(np.float32)).resize(size, method)).astype(a.dtype)


def plan_custom_resize(original_width: int, original_height: int, target_width: int, target_height: int) -> CanvasPlan:
    """Fit the original into user bounds, then add model padding; never enlarge it."""
    values = (original_width, original_height, target_width, target_height)
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in values):
        raise ValueError("Source and custom resize dimensions must be positive integers.")
    if not 32 <= target_width <= 1920 or not 32 <= target_height <= 1088:
        raise ValueError("Custom resize bounds must be 32–1920 wide and 32–1088 high.")
    scale = min(target_width / original_width, target_height / original_height, 1.0)
    w = min(target_width, max(1, math.floor(original_width * scale + 0.5)))
    h = min(target_height, max(1, math.floor(original_height * scale + 0.5)))
    cw, ch = ((w + 31) // 32) * 32, ((h + 31) // 32) * 32
    left, top = (cw - w) // 2, (ch - h) // 2
    return CanvasPlan(original_width, original_height, cw, ch, w, h, left, top, cw - w - left, ch - h - top)


def _pad(array: Any, plan: CanvasPlan, *, hard: bool = False) -> Any:
    import numpy as np

    a = np.asarray(array)
    if a.shape[:2] != (plan.original_height, plan.original_width):
        raise ValueError("Frame dimensions do not match the source canvas plan.")
    out = np.zeros((plan.height, plan.width) + a.shape[2:], dtype=a.dtype)
    out[plan.pad_top : plan.pad_top + plan.content_height, plan.pad_left : plan.pad_left + plan.content_width] = (
        _resize(a, (plan.content_width, plan.content_height), hard=hard)
    )
    return out


def pad_rgb(frame: Any, plan: CanvasPlan) -> Any:
    return _pad(frame, plan)


def pad_mask(mask: Any, plan: CanvasPlan, *, hard: bool = False) -> Any:
    """Black padding means no subject; RGB and mask use the exact same rectangle."""
    return _pad(mask, plan, hard=hard)


def restore_mask(mask: Any, plan: CanvasPlan, *, output_size: str = "original") -> Any:
    import numpy as np

    a = np.asarray(mask)
    if a.shape[:2] != (plan.height, plan.width):
        raise ValueError("Rendered alpha dimensions do not match the LTX canvas.")
    cropped = a[plan.pad_top : plan.pad_top + plan.content_height, plan.pad_left : plan.pad_left + plan.content_width]
    if output_size not in ("original", "processing"):
        raise ValueError("Output size must be original or processing.")
    restored = (
        _resize(cropped, (plan.original_width, plan.original_height)) if output_size == "original" else cropped.copy()
    )
    return np.clip(restored, 0, 1) if np.issubdtype(restored.dtype, np.floating) else restored


@dataclass(frozen=True)
class Chunk:
    start: int
    valid_frames: int
    model_frames: int


def plan_chunks(num_frames: int, chunk_size: int = 49, overlap: int = 9) -> list[Chunk]:
    """LTX's temporal lattice is 8n+1. Repeat end frames only inside each model call."""
    if num_frames < 1 or chunk_size < 9 or chunk_size > 145 or (chunk_size - 1) % 8:
        raise ValueError("Use at least one source frame and an LTX chunk of 9–145 frames (8n+1).")
    if overlap < 1 or overlap >= chunk_size or (chunk_size - overlap) % 8:
        raise ValueError("Chunk overlap must leave a positive step divisible by 8 (49/9 is recommended).")
    result = []
    start = 0
    while True:
        valid = min(chunk_size, num_frames - start)
        result.append(Chunk(start, valid, chunk_size))
        if start + valid >= num_frames:
            break
        start += chunk_size - overlap
    return result


def find_ffmpeg(explicit: str | None = None, *, probe: bool = False) -> str:
    name = "ffprobe" if probe else "ffmpeg"
    candidates = [explicit] if explicit else []
    candidates += [os.environ.get(name.upper() + "_BINARY"), shutil.which(name)]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates.append(
            str(
                Path(local)
                / "Griptape Nodes"
                / "xdg_data_home"
                / "griptape_nodes"
                / "ffmpeg"
                / "bin"
                / "win32"
                / (name + ".exe")
            )
        )
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    # The engine depends on static-ffmpeg, so its binaries resolve in both the engine and the worker.
    try:
        import static_ffmpeg

        ffmpeg_path, ffprobe_path = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
        return ffprobe_path if probe else ffmpeg_path
    except Exception:
        pass
    if not probe:
        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
    raise RuntimeError(f"{name} is unavailable. Install the Griptape FFmpeg dependency or set {name.upper()}_BINARY.")


def probe_media(path: str) -> dict[str, Any]:
    from PIL import Image

    try:
        with Image.open(path) as im:
            return {"width": im.width, "height": im.height, "fps": 24.0, "num_frames": 1, "kind": "image"}
    except (OSError, ValueError):
        pass
    command = [
        find_ffmpeg(probe=True),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_read_frames,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
    )
    if result.returncode:
        raise RuntimeError("Could not inspect source video: " + result.stderr[-1000:])
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError("Source has no video stream.")
    stream = streams[0]
    n, d = str(stream.get("avg_frame_rate", "0/1")).split("/")
    fps = float(n) / float(d) if float(d) else 0
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("The source video has no usable frame rate.")
    count = stream.get("nb_read_frames", stream.get("nb_frames", "0"))
    if count in (None, "N/A", "0"):
        raise ValueError("The source video has no usable frame count.")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "num_frames": int(count),
        "kind": "video",
    }


def prepare_video(source: str, output_path: str, plan: CanvasPlan) -> str:
    """Lossless FFV1 intermediate; no crop, frame-rate conversion or audio synthesis."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    filters = (
        f"scale={plan.content_width}:{plan.content_height}:flags=lanczos,"
        f"pad={plan.width}:{plan.height}:{plan.pad_left}:{plan.pad_top}:color=black,setsar=1"
    )
    command = [
        find_ffmpeg(),
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-i",
        source,
        "-map",
        "0:v:0",
        "-vf",
        filters,
        "-an",
        "-fps_mode",
        "passthrough",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-pix_fmt",
        "bgr0",
        output_path,
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
    )
    if result.returncode:
        raise RuntimeError("LTX canvas preparation failed: " + result.stderr[-1500:])
    return output_path
