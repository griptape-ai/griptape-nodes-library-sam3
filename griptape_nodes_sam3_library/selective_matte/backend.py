"""Native Meta SAM 3.1 adapter, independent of any ComfyUI integration.

API verified against facebookresearch/sam3's model_builder.py,
sam3_base_predictor.py and sam3.1_video_predictor_example.ipynb (2026-10-14).
Heavy dependencies and model weights are loaded only when segment() is called.
The current native multiplex predictor requires CUDA with bfloat16 support.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import json
import math
import os
import shutil
import threading
import uuid
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

MODEL_ID = "facebook/sam3.1"
CHECKPOINT_NAME = "sam3.1_multiplex.pt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
NATIVE_MODULES = ("torch", "torchvision", "sam3", "triton", "pycocotools", "einops", "timm", "pkg_resources", "psutil")


class SAM31Error(RuntimeError):
    """An actionable configuration, media, or native inference failure."""


class SegmentationCancelled(SAM31Error):
    """The caller cancelled between native inference operations."""


def _check_cancel(cancel: Any) -> None:
    if cancel is not None and (cancel.is_set() if hasattr(cancel, "is_set") else cancel()):
        raise SegmentationCancelled("SAM 3.1 segmentation cancelled.")


def _report(progress: Callable | None, fraction: float, message: str) -> None:
    if progress is not None:
        progress(max(0.0, min(1.0, fraction)), message)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not a boolean.")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if parsed != value:
        raise ValueError(f"{name} must be an integer.")
    return parsed


def normalize_points(points: Any, num_frames: int, selected_frame: int = 0) -> list[dict]:
    """Validate points without importing torch; coordinates stay normalized.

    Accepts the widget's frame/objectId keys and Python frame_index/object_id.
    Exclude clicks are negative prompts for their object, not separate objects.
    """
    if isinstance(points, str):
        points = json.loads(points)
    if isinstance(points, dict):
        points = points.get("points", [])
    if not isinstance(points, (list, tuple)) or not points:
        raise ValueError("Add an include click on the subject before running SAM 3.1.")
    normalized = []
    for item in points:
        if not isinstance(item, dict):
            raise ValueError("Each click must be an object with x, y, label and frame.")
        frame = _integer(item.get("frame_index", item.get("frame", selected_frame)), "Click frame")
        object_id = _integer(item.get("object_id", item.get("objectId", 1)), "Object ID")
        if not 0 <= frame < num_frames:
            raise ValueError(f"Click frame {frame} is outside 0–{num_frames - 1}.")
        if object_id < 1:
            raise ValueError("Object IDs must be positive integers.")
        try:
            x, y = float(item["x"]), float(item["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Every click needs normalized x and y coordinates.") from exc
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y)):
            raise ValueError("Click x and y must be finite normalized coordinates from 0 to 1.")
        label = item.get("label", 1)
        if isinstance(label, str):
            label = {"include": 1, "exclude": 0, "positive": 1, "negative": 0}.get(label.lower(), label)
        if label not in (0, 1):
            raise ValueError("Click label must be include/exclude or 1/0.")
        normalized.append({"frame_index": frame, "object_id": object_id, "x": x, "y": y, "label": int(label)})
    grouped = defaultdict(list)
    for point in normalized:
        grouped[point["object_id"]].append(point)
    if len(grouped) > 16:
        raise ValueError("This SAM 3.1 adapter supports up to 16 selected objects per run.")
    for object_id, group in grouped.items():
        earliest = min(p["frame_index"] for p in group)
        if not any(p["label"] == 1 and p["frame_index"] == earliest for p in group):
            raise ValueError(f"Object {object_id} needs an include click on its first prompted frame ({earliest}).")
    return normalized


def _media_modules():
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise SAM31Error(
            "Install this library's media dependencies (numpy, Pillow, opencv-python-headless) in the node engine environment."
        ) from exc
    try:
        import cv2
    except ImportError:
        cv2 = None
    return cv2, np, Image


def _require_cv2(cv2):
    if cv2 is None:
        raise SAM31Error("Video input requires opencv-python-headless in the SAM 3.1 Python environment.")


def inspect_media(source_path: str | os.PathLike, cancel=None) -> dict:
    """Return decoded frame count and dimensions, plus the source's reported FPS."""
    cv2, _, Image = _media_modules()
    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise SAM31Error(f"Source file does not exist: {source}")
    if source.suffix.lower() in IMAGE_SUFFIXES:
        with Image.open(source) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise SAM31Error("Animated images are unsupported; supply a video or a still image.")
            width, height = image.size
        return {"width": width, "height": height, "fps": 1.0, "num_frames": 1, "is_image": True}
    _require_cv2(cv2)
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise SAM31Error(f"Cannot decode video: {source.name}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise SAM31Error("The video has no usable frame rate. Convert it to a constant-frame-rate video first.")
        count = 0
        width = height = 0
        while True:
            _check_cancel(cancel)
            ok, frame = capture.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if count and (h, w) != (height, width):
                raise SAM31Error("The video's decoded frame dimensions change within the clip.")
            height, width = h, w
            count += 1
        if count == 0:
            raise SAM31Error("The video has no decodable frames.")
        return {"width": width, "height": height, "fps": fps, "num_frames": count, "is_image": False}
    finally:
        capture.release()


def extract_frame(source_path, frame_index: int, destination, *, is_image=False, cancel=None) -> str:
    """Write an exact decoded RGB PNG; sequential decoding avoids seek ambiguity."""
    cv2, _, Image = _media_modules()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if is_image:
        with Image.open(source_path) as image:
            image.convert("RGB").save(destination)
    else:
        _require_cv2(cv2)
        capture = cv2.VideoCapture(str(source_path))
        try:
            for _ in range(frame_index + 1):
                _check_cancel(cancel)
                ok, frame = capture.read()
                if not ok:
                    raise SAM31Error(f"Cannot decode source frame {frame_index}.")
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(destination)
        finally:
            capture.release()
    return str(destination.resolve())


def _stage_video_frames(source_path, destination, media, cancel=None, progress=None) -> Path:
    """Native image-folder loading correctly normalizes RGB; current CV2 path does not.

    Preserve the exact decoded source pixels, order, dimensions and frame count.
    The caller deletes these temporary PNGs once synchronous init_state returns.
    """
    cv2, _, Image = _media_modules()
    _require_cv2(cv2)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    capture = None
    try:
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            raise SAM31Error("Cannot decode source frames for native SAM 3.1.")
        count = 0
        while True:
            _check_cancel(cancel)
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != (media["height"], media["width"]):
                raise SAM31Error("A source frame changed dimensions while preparing SAM 3.1.")
            if count >= media["num_frames"]:
                raise SAM31Error("The source frame count changed while preparing SAM 3.1.")
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(
                destination / f"{count:06d}.png", compress_level=1
            )
            count += 1
            _report(
                progress,
                0.08 + 0.05 * count / media["num_frames"],
                f"Prepared {count} of {media['num_frames']} original frames",
            )
        if count != media["num_frames"]:
            raise SAM31Error(
                f"Source decoded {count} frames; expected {media['num_frames']}. Refusing a shifted mask sequence."
            )
        _check_cancel(cancel)
        return destination
    except BaseException:
        # This function created the directory exclusively; never remove an
        # existing user-provided directory on a collision or failed mkdir.
        shutil.rmtree(destination, ignore_errors=True)
        raise
    finally:
        if capture is not None:
            capture.release()


def _release_native_autocast(predictor) -> None:
    """Exit the two constructor-owned contexts in reverse construction order."""
    tracker = getattr(getattr(predictor, "model", None), "tracker", None)
    seen = set()
    for owner in (predictor, tracker):
        native_context = getattr(owner, "bf16_context", None)
        if native_context is not None and id(native_context) not in seen:
            native_context.__exit__(None, None, None)
            seen.add(id(native_context))


def _allow_native_sdpa_fallback(decoder_module, backend_enum) -> None:
    """Let native non-FA3 memory attention use available PyTorch CUDA kernels.

    The pinned decoder forces FLASH_ATTENTION even on the official Windows
    Torch wheel, which has no compiled FlashAttention. Change only the decoder's
    imported context helper, retaining native attention and Torch flag cleanup.
    """
    original = decoder_module.sdpa_kernel
    if getattr(original, "_sam31_allows_fallback", False):
        return

    @functools.wraps(original)
    def compatible(backends, *args, **kwargs):
        requested = backends if isinstance(backends, list) else [backends]
        if requested == [backend_enum.FLASH_ATTENTION]:
            backends = [backend_enum.FLASH_ATTENTION, backend_enum.EFFICIENT_ATTENTION, backend_enum.MATH]
        return original(backends, *args, **kwargs)

    compatible._sam31_allows_fallback = True
    decoder_module.sdpa_kernel = compatible


def _adapt_native_init_state(predictor) -> None:
    """Bridge a specific upstream base/multiplex signature mismatch.

    The shared start_session currently forwards offload_state_to_cpu=False,
    although multiplex init_state does not accept it. Do not enable or emulate
    state offloading; merely discard that unsupported false default.
    """
    original = predictor.model.init_state
    signature = inspect.signature(original)
    if "offload_state_to_cpu" in signature.parameters or any(
        p.kind == p.VAR_KEYWORD for p in signature.parameters.values()
    ):
        return

    @functools.wraps(original)
    def compatible(*args, offload_state_to_cpu=False, **kwargs):
        if offload_state_to_cpu:
            raise SAM31Error("This native SAM 3.1 build does not support state offloading.")
        return original(*args, **kwargs)

    predictor.model.init_state = compatible


def _adapt_native_first_point_output(predictor) -> None:
    """Keep a freshly computed click mask when native frame cache is empty.

    The pinned multiplex _build_sam2_output returns early for an uncached frame
    before merging refined_obj_id_to_mask. Seed only that empty cache entry;
    native inference and all subsequent native merging/filtering remain intact.
    """
    original = getattr(predictor.model, "_build_sam2_output", None)
    if original is None:
        return
    signature = inspect.signature(original)
    if "refined_obj_id_to_mask" not in signature.parameters:
        return

    @functools.wraps(original)
    def compatible(inference_state, frame_idx, refined_obj_id_to_mask=None):
        if refined_obj_id_to_mask:
            inference_state.setdefault("cached_frame_outputs", {}).setdefault(frame_idx, {})
        return original(inference_state, frame_idx, refined_obj_id_to_mask)

    predictor.model._build_sam2_output = compatible


def _validate_local_weights(torch, predictor, checkpoint: str) -> None:
    """Reject wrong-generation or incomplete checkpoints despite upstream strict=False."""
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if isinstance(state, dict) and isinstance(state.get("model"), dict):
            state = state["model"]
        mapped = {}
        for key, value in state.items():
            if key.startswith("sam3_model."):
                key = "detector." + key[len("sam3_model.") :]
            elif key.startswith("sam2_predictor."):
                key = "tracker." + key[len("sam2_predictor.") :]
            mapped[key] = value
        wrong = [
            name
            for name, param in predictor.model.named_parameters()
            if name not in mapped or tuple(mapped[name].shape) != tuple(param.shape)
        ]
        if wrong:
            raise SAM31Error(
                f"Checkpoint is not a complete SAM 3.1 multiplex checkpoint ({len(wrong)} missing or mismatched parameters). Use {CHECKPOINT_NAME} from {MODEL_ID}."
            )
    except SAM31Error:
        raise
    except Exception as exc:
        raise SAM31Error(
            f"Cannot validate the local checkpoint. Use the official {CHECKPOINT_NAME} PyTorch checkpoint; safetensors and older SAM checkpoints are unsupported."
        ) from exc


def runtime_info() -> dict:
    """Read availability without importing torch/SAM, touching CUDA, or downloading."""
    import importlib.util
    import sys

    available = {}
    for name in NATIVE_MODULES:
        try:
            available[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            available[name] = False
    return {
        "python_executable": sys.executable,
        "python_version": list(sys.version_info[:3]),
        "platform": sys.platform,
        "modules": available,
        "missing_modules": [name for name, present in available.items() if not present],
        "cuda_checked": False,
        "model_loaded": False,
    }


def _check_native_dependencies() -> None:
    missing = runtime_info()["missing_modules"]
    if missing:
        extra = (
            (
                " Meta imports Triton unconditionally, even with FlashAttention 3 and compilation disabled. "
                "On Windows use a separately configured compatible Triton environment; see the README."
            )
            if "triton" in missing
            else ""
        )
        if "pkg_resources" in missing:
            extra += (
                " Meta's pinned source needs pkg_resources; install setuptools==81.0.0 in the dedicated SAM runtime."
            )
        raise SAM31Error(
            "The SAM 3.1 Python environment is missing: "
            + ", ".join(missing)
            + ". Install the native SAM 3.1 requirements in the selected SAM Python environment."
            + extra
        )


def _build_predictor(checkpoint_path=None, device="cuda"):
    import sys

    if device not in ("cuda", "cuda:0", "auto"):
        raise SAM31Error(
            "Native SAM 3.1 multiplex inference requires CUDA GPU 0 with bfloat16 support. CPU and MPS are unsupported by this upstream predictor."
        )
    if sys.version_info < (3, 12):  # noqa: UP036
        raise SAM31Error(
            "Meta's native SAM 3.1 requires Python 3.12 or newer. Select a compatible SAM Python executable."
        )
    _check_native_dependencies()
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise SAM31Error(
            "CUDA-enabled PyTorch could not load. Install PyTorch 2.7 or newer with its matching torchvision and CUDA runtime in the selected SAM Python environment."
        ) from exc
    if not torch.cuda.is_available():
        raise SAM31Error(
            "CUDA is unavailable. Install CUDA-enabled PyTorch and a compatible NVIDIA driver; SAM 3.1 has no CPU fallback in this adapter."
        )
    with torch.cuda.device(0):
        if not torch.cuda.is_bf16_supported():
            raise SAM31Error(
                "CUDA GPU 0 does not support the bfloat16 inference required by Meta's SAM 3.1 multiplex predictor."
            )
    try:
        from sam3.model import decoder as sam_decoder
        from sam3.model_builder import build_sam3_predictor
        from torch.nn.attention import SDPBackend

        _allow_native_sdpa_fallback(sam_decoder, SDPBackend)
    except (ImportError, OSError) as exc:
        raise SAM31Error(
            "The native SAM 3.1 package or one of its compiled dependencies is unavailable. Install the March 2026-or-newer facebookresearch/sam3 source in this Python environment (Python 3.12+, PyTorch 2.7+, CUDA 12.6+). See the library README."
        ) from exc
    checkpoint = None
    if checkpoint_path:
        checkpoint = str(Path(checkpoint_path).expanduser().resolve())
        if not Path(checkpoint).is_file():
            raise SAM31Error(f"SAM 3.1 checkpoint does not exist: {checkpoint}")
        if Path(checkpoint).suffix.lower() != ".pt":
            raise SAM31Error(f"Use the native {CHECKPOINT_NAME} checkpoint, not a SAM3/SAM2 or safetensors checkpoint.")
    predictor = None
    try:
        # Explicit version, no older-model fallback. Optional FA3/compile are
        # disabled for portability; no model download happens before this call.
        with torch.cuda.device(0):
            predictor = build_sam3_predictor(
                version="sam3.1",
                checkpoint_path=checkpoint,
                use_fa3=False,
                compile=False,
                warm_up=False,
                async_loading_frames=False,
            )
        # Upstream enters a thread-local autocast context in its constructor.
        # Restore the caller's context and scope it around our inference calls.
        _release_native_autocast(predictor)
        _adapt_native_init_state(predictor)
        _adapt_native_first_point_output(predictor)
        if checkpoint:
            _validate_local_weights(torch, predictor, checkpoint)
        return predictor
    except SAM31Error:
        if predictor is not None:
            predictor.shutdown()
        raise
    except Exception as exc:
        if predictor is not None:
            predictor.shutdown()
        if not checkpoint:
            raise SAM31Error(
                f"Could not load SAM 3.1. Accept access at https://huggingface.co/{MODEL_ID}, authenticate with `hf auth login` in the engine environment, or supply a local {CHECKPOINT_NAME}. Native error: {exc}"
            ) from exc
        raise SAM31Error(
            f"SAM 3.1 could not load the local checkpoint. Verify native dependencies and use the complete {CHECKPOINT_NAME}. Native error: {exc}"
        ) from exc


def _native_context():
    import torch

    stack = contextlib.ExitStack()
    stack.enter_context(torch.cuda.device(0))
    stack.enter_context(torch.inference_mode())
    stack.enter_context(torch.autocast(device_type="cuda", dtype=torch.bfloat16))
    return stack


class SAM31Backend:
    """One serialized model and one cached source session; injectable for CPU tests."""

    def __init__(self, predictor_factory=None, inference_context=None):
        self._factory = predictor_factory or _build_predictor
        self._context = inference_context or (_native_context if predictor_factory is None else contextlib.nullcontext)
        self._lock = threading.RLock()
        self._predictor = None
        self._model_key = None
        self._session_id = None
        self._session_key = None
        self._media_cache = {}

    def _close_session(self):
        if self._predictor is not None and self._session_id is not None:
            try:
                self._predictor.handle_request({"type": "close_session", "session_id": self._session_id})
            finally:
                self._session_id = self._session_key = None

    def close_session(self):
        with self._lock:
            self._close_session()

    def unload_model(self):
        with self._lock:
            self._close_session()
            if self._predictor is not None:
                self._predictor.shutdown()
            self._predictor = self._model_key = None
            self._media_cache.clear()
            # Returning allocated model tensors is not enough when a separate
            # model needs the device: also release our CUDA allocator pool.
            if self._factory is _build_predictor:
                import gc
                import sys

                gc.collect()
                torch = sys.modules.get("torch")
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _model(self, checkpoint_path, device):
        checkpoint = str(Path(checkpoint_path).expanduser().resolve()) if checkpoint_path else None
        fingerprint = (
            (checkpoint, Path(checkpoint).stat().st_mtime_ns, Path(checkpoint).stat().st_size)
            if checkpoint and Path(checkpoint).is_file()
            else (checkpoint,)
        )
        key = (fingerprint, device)
        if key != self._model_key:
            self.unload_model()
            self._predictor = self._factory(checkpoint, device)
            self._model_key = key
        return self._predictor

    def _session(self, predictor, key, resource):
        if self._session_key == key:
            predictor.handle_request({"type": "reset_session", "session_id": self._session_id})
        else:
            self._close_session()
            result = predictor.handle_request(
                {"type": "start_session", "resource_path": str(resource), "offload_video_to_cpu": True}
            )
            self._session_id = result["session_id"]
            self._session_key = key
        return self._session_id

    def segment(
        self,
        source_path,
        points,
        checkpoint_path=None,
        mode="preview",
        selected_frame=0,
        output_dir=None,
        progress=None,
        cancel=None,
        device="cuda",
        make_preview_video=True,
    ) -> dict:
        """Segment original-resolution media; all frame indices are zero based.

        Preview computes only the selected still frame. Track applies all clicks
        and propagates in both directions, preserving every source frame index.
        PNGs are authoritative; optional browser video is only a viewing proxy.
        """
        if mode not in ("preview", "track"):
            raise ValueError("mode must be 'preview' or 'track'.")
        if output_dir is None:
            raise ValueError("output_dir is required so segmentation artifacts have an explicit destination.")
        with self._lock:
            _check_cancel(cancel)
            source = Path(source_path).expanduser().resolve()
            if not source.is_file():
                raise SAM31Error(f"Source file does not exist: {source}")
            source_key = (str(source), source.stat().st_mtime_ns, source.stat().st_size)
            _report(progress, 0.01, "Inspecting source frames")
            if source_key not in self._media_cache:
                self._media_cache = {source_key: inspect_media(source, cancel)}
            media = dict(self._media_cache[source_key])
            selected_frame = _integer(selected_frame, "Selected frame")
            if not 0 <= selected_frame < media["num_frames"]:
                raise ValueError(f"Selected frame must be between 0 and {media['num_frames'] - 1}.")
            prompts = normalize_points(points, media["num_frames"], selected_frame)
            if mode == "preview":
                prompts = [p for p in prompts if p["frame_index"] == selected_frame]
                if not prompts:
                    raise ValueError("Add an include click on the selected frame to preview it.")
                normalize_points(prompts, media["num_frames"], selected_frame)
            output_root = Path(output_dir).expanduser().resolve()
            run = output_root / ("sam31_" + uuid.uuid4().hex)
            sequence = run / "masks"
            sequence.mkdir(parents=True)
            warnings = []
            try:
                preview_source = extract_frame(
                    source, selected_frame, run / "source_preview.png", is_image=media["is_image"], cancel=cancel
                )
                _report(progress, 0.08, "Loading native SAM 3.1")
                predictor = self._model(checkpoint_path, device)
                _check_cancel(cancel)
                with self._context():
                    resource = preview_source if mode == "preview" or media["is_image"] else str(source)
                    session_key = (
                        (source_key, "still", selected_frame)
                        if mode == "preview" or media["is_image"]
                        else (source_key, "video")
                    )
                    staged_frames = None
                    try:
                        if mode == "track" and not media["is_image"] and self._session_key != session_key:
                            staged_frames = _stage_video_frames(source, run / "source_frames", media, cancel, progress)
                            resource = str(staged_frames)
                        session_id = self._session(predictor, session_key, resource)
                    finally:
                        # async_loading_frames=False guarantees native frames
                        # are resident before this scratch directory is removed.
                        if staged_frames is not None:
                            shutil.rmtree(staged_frames, ignore_errors=True)
                    # Every execution starts with the complete current click set;
                    # deleting an old UI click therefore cannot leave a stale mask.
                    grouped = defaultdict(list)
                    for p in prompts:
                        native_frame = 0 if mode == "preview" else p["frame_index"]
                        grouped[(native_frame, p["object_id"])].append(p)
                    selected_ids = sorted({p["object_id"] for p in prompts})
                    initial = {}
                    for (frame, object_id), group in sorted(grouped.items()):
                        _check_cancel(cancel)
                        response = predictor.handle_request(
                            {
                                "type": "add_prompt",
                                "session_id": session_id,
                                "frame_index": frame,
                                "points": [[p["x"], p["y"]] for p in group],
                                "point_labels": [p["label"] for p in group],
                                "obj_id": object_id,
                                "rel_coordinates": True,
                                "clear_old_points": True,
                            }
                        )
                        initial[int(response["frame_index"])] = response["outputs"]
                    _report(progress, 0.15, "SAM 3.1 clicks applied")
                    written = set()

                    def save_response(frame, outputs):
                        if mode == "preview":
                            if frame != 0:
                                raise SAM31Error(f"SAM 3.1 returned unexpected still-frame index {frame}.")
                            frame = selected_frame
                        if not 0 <= frame < media["num_frames"]:
                            raise SAM31Error(f"SAM 3.1 returned an invalid frame index: {frame}.")
                        mask = _union_mask(outputs, selected_ids, media["width"], media["height"])
                        _, _, Image = _media_modules()
                        Image.fromarray(mask).save(sequence / f"mask_{frame:06d}.png")
                        written.add(frame)

                    if mode == "preview" or media["num_frames"] == 1:
                        for frame, outputs in initial.items():
                            save_response(frame, outputs)
                    else:
                        start_frame = min(p["frame_index"] for p in prompts)
                        direction = "both"
                        if start_frame == media["num_frames"] - 1:
                            # Native action history treats an edge-started pass
                            # as complete. A forward pass from the last frame
                            # would make its backward pass fetch uncached data.
                            # Preserve the native seed and propagate backward.
                            for frame, outputs in initial.items():
                                save_response(frame, outputs)
                            direction = "backward"
                        stream = predictor.handle_stream_request(
                            {
                                "type": "propagate_in_video",
                                "session_id": session_id,
                                "propagation_direction": direction,
                                "start_frame_index": start_frame,
                            }
                        )
                        try:
                            for response in stream:
                                _check_cancel(cancel)
                                save_response(int(response["frame_index"]), response["outputs"])
                                _report(
                                    progress,
                                    0.15 + 0.75 * len(written) / media["num_frames"],
                                    f"Tracked {len(written)} of {media['num_frames']} frames",
                                )
                        finally:
                            if hasattr(stream, "close"):
                                stream.close()
                        missing = set(range(media["num_frames"])) - written
                        if missing:
                            raise SAM31Error(
                                f"SAM 3.1 omitted {len(missing)} source frames (first missing: {min(missing)}). No completed mask sequence was produced."
                            )
                _check_cancel(cancel)
                preview_mask = sequence / f"mask_{selected_frame:06d}.png"
                video_path = None
                if make_preview_video and mode == "track" and media["num_frames"] > 1:
                    _report(progress, 0.92, "Encoding optional mask preview")
                    video_path, warning = _write_preview_video(sequence, run / "mask_preview.mp4", media, cancel)
                    if warning:
                        warnings.append(warning)
                result = dict(
                    preview_mask_path=str(preview_mask),
                    mask_video_path=video_path,
                    mask_sequence_dir=str(sequence),
                    source_preview_path=preview_source,
                    manifest_path=str(run / "manifest.json"),
                    **media,
                    mode=mode,
                    selected_frame=selected_frame,
                    frame_indices=sorted(written),
                    object_ids=selected_ids,
                    model_info={
                        "name": "SAM 3.1",
                        "repo_id": MODEL_ID,
                        "checkpoint": str(checkpoint_path) if checkpoint_path else CHECKPOINT_NAME,
                        "backend": "native-multiplex",
                        "device": device,
                    },
                    warnings=warnings,
                )
                manifest = dict(
                    result,
                    source_path=str(source),
                    points=prompts,
                    mask_pattern="masks/mask_%06d.png",
                    frame_index_base=0,
                    mask_values={"background": 0, "foreground": 255},
                    complete=True,
                    mask_video_is_lossy=video_path is not None,
                    mask_video_padding={"right": 0, "bottom": 0} if video_path else None,
                )
                Path(result["manifest_path"]).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                _report(progress, 1.0, "SAM 3.1 masks ready")
                return result
            except BaseException:
                # Do not reuse partially mutated native state after a failure.
                if self._predictor is not None and self._session_id is not None:
                    with contextlib.suppress(Exception):
                        self._predictor.handle_request({"type": "cancel_propagation", "session_id": self._session_id})
                    with contextlib.suppress(Exception):
                        self._close_session()
                # Only remove this invocation's unique output directory.
                if run.resolve().parent == output_root and run.name.startswith("sam31_"):
                    shutil.rmtree(run, ignore_errors=True)
                raise


def _union_mask(outputs, selected_ids, width, height):
    _, np, _ = _media_modules()

    def array(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    if "out_obj_ids" not in outputs or "out_binary_masks" not in outputs:
        raise SAM31Error("Unexpected native SAM 3.1 output; expected out_obj_ids and out_binary_masks.")
    ids = array(outputs["out_obj_ids"]).reshape(-1)
    masks = array(outputs["out_binary_masks"])
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.shape != (len(ids), height, width):
        raise SAM31Error(
            f"Native mask dimensions {masks.shape} do not match source {(len(ids), height, width)}; refusing a misaligned mask."
        )
    selected = masks[np.isin(ids, selected_ids)]
    union = np.any(selected > 0, axis=0) if len(selected) else np.zeros((height, width), dtype=bool)
    return union.astype(np.uint8) * 255


def _write_preview_video(sequence, destination, media, cancel, *, frame_pattern="mask_{index:06d}.png", crop_box=None):
    """Exact-size browser proxy. Odd dimensions use VP9, even dimensions H.264."""
    odd_dimensions = bool(media["width"] % 2 or media["height"] % 2)
    codec = "libvpx-vp9" if odd_dimensions else "libx264"
    destination = Path(destination).with_suffix(".webm" if odd_dimensions else ".mp4")
    try:
        from fractions import Fraction

        import av

        _, np, Image = _media_modules()
        rate = Fraction(str(media["fps"])).limit_denominator(100000)
        with av.open(str(destination), mode="w") as container:
            stream = container.add_stream(codec, rate=rate)
            stream.width = media["width"]
            stream.height = media["height"]
            stream.pix_fmt = "yuv420p"
            stream.options = (
                {"crf": "18", "b": "0", "deadline": "good", "cpu-used": "4"}
                if odd_dimensions
                else {"crf": "18", "preset": "fast"}
            )
            for index in range(media["num_frames"]):
                _check_cancel(cancel)
                with Image.open(sequence / frame_pattern.format(index=index)) as image:
                    rgb = image.convert("RGB")
                    if crop_box is not None:
                        rgb = rgb.crop(crop_box)
                    mask = np.asarray(rgb)
                if mask.shape[:2] != (stream.height, stream.width):
                    raise SAM31Error(
                        f"Preview frame {index} dimensions {mask.shape[:2]} differ from the source; refusing a misaligned proxy."
                    )
                frame = av.VideoFrame.from_ndarray(mask, format="rgb24")
                frame.pts = index
                frame.time_base = 1 / rate
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        return str(destination), None
    except SegmentationCancelled:
        raise
    except Exception as exc:
        destination.unlink(missing_ok=True)
        return (
            None,
            f"Exact-size preview unavailable ({type(exc).__name__}); install PyAV with {codec} support or use the lossless PNG mask sequence.",
        )


_BACKEND = SAM31Backend()


def segment(
    source_path,
    points,
    checkpoint_path=None,
    mode="preview",
    selected_frame=0,
    output_dir=None,
    progress=None,
    cancel=None,
    device="cuda",
    make_preview_video=True,
):
    return _BACKEND.segment(
        source_path,
        points,
        checkpoint_path,
        mode,
        selected_frame,
        output_dir,
        progress,
        cancel,
        device,
        make_preview_video,
    )


def close_session():
    _BACKEND.close_session()


def unload_model():
    _BACKEND.unload_model()


# A persistent worker keeps native SAM dependencies and CUDA state out of the
# Griptape engine interpreter. It is opt-in and never creates an environment.
_WORKERS = {}
_WORKER_LOCK = threading.RLock()


class _Worker:
    def __init__(self, executable):
        import queue
        import subprocess

        self.messages = queue.Queue()
        self.process = subprocess.Popen(
            [str(executable), "-u", str(Path(__file__).resolve()), "--worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

        def read_messages():
            try:
                for line in self.process.stdout:
                    if line.startswith("__SAM31__"):
                        try:
                            self.messages.put(json.loads(line[len("__SAM31__") :]))
                        except json.JSONDecodeError:
                            pass
            finally:
                self.messages.put({"type": "exited"})

        self.reader = threading.Thread(target=read_messages, name="sam31-worker-output", daemon=True)
        self.reader.start()

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except Exception:
                self.process.kill()
                self.process.wait(timeout=3)
        for handle in (self.process.stdin, self.process.stdout):
            if handle is not None:
                handle.close()


def close_worker(python_executable=None):
    """Stop only worker processes launched by this adapter; retain saved masks."""
    with _WORKER_LOCK:
        keys = [str(Path(python_executable).expanduser().resolve())] if python_executable else list(_WORKERS)
        for key in keys:
            worker = _WORKERS.pop(key, None)
            if worker is not None:
                worker.close()


def segment_via_worker(
    python_executable,
    source_path,
    points,
    checkpoint_path=None,
    mode="preview",
    selected_frame=0,
    output_dir=None,
    progress=None,
    cancel=None,
    device="cuda",
    make_preview_video=True,
):
    """The segment() contract in an explicit Python environment, cached per executable.

    Cancellation terminates only our worker, discards its partial invocation
    directory, and causes the next operation to start a fresh model process.
    """
    import queue

    executable = Path(python_executable).expanduser().resolve()
    if not executable.is_file():
        raise SAM31Error(f"SAM Python executable does not exist: {executable}")
    if output_dir is None:
        raise ValueError("output_dir is required.")
    output_root = Path(output_dir).expanduser().resolve()
    with _WORKER_LOCK:
        _check_cancel(cancel)
        key = str(executable)
        worker = _WORKERS.get(key)
        if worker is None or worker.process.poll() is not None:
            close_worker(key)
            worker = _WORKERS[key] = _Worker(executable)
        invocation = output_root / ("sam31_worker_" + uuid.uuid4().hex)
        invocation.mkdir(parents=True)
        request = {
            "source_path": str(Path(source_path).expanduser().resolve()),
            "points": points,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "mode": mode,
            "selected_frame": selected_frame,
            "output_dir": str(invocation),
            "device": device,
            "make_preview_video": make_preview_video,
        }
        try:
            worker.process.stdin.write(json.dumps(request) + "\n")
            worker.process.stdin.flush()
            while True:
                _check_cancel(cancel)
                try:
                    message = worker.messages.get(timeout=0.1)
                except queue.Empty:
                    continue
                if message.get("type") == "progress":
                    _report(progress, message["fraction"], message["message"])
                elif message.get("type") == "result":
                    return message["result"]
                elif message.get("type") == "error":
                    raise SAM31Error(message["message"])
                elif message.get("type") == "exited":
                    raise SAM31Error(
                        "The SAM 3.1 worker exited before returning masks. Check the configured Python environment, native CUDA dependencies, and available memory."
                    )
        except BaseException:
            close_worker(key)
            if invocation.resolve().parent == output_root and invocation.name.startswith("sam31_worker_"):
                shutil.rmtree(invocation, ignore_errors=True)
            raise


def _worker_main():
    import sys

    channel = sys.stdout

    def emit(message):
        channel.write("__SAM31__" + json.dumps(message) + "\n")
        channel.flush()

    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                # Upstream console output cannot corrupt our JSON protocol.
                with contextlib.redirect_stdout(sys.stderr):
                    result = segment(
                        **request,
                        progress=lambda fraction, message: emit(
                            {"type": "progress", "fraction": fraction, "message": message}
                        ),
                    )
                emit({"type": "result", "result": result})
            except Exception as exc:
                emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
    finally:
        unload_model()


import atexit  # noqa: E402

atexit.register(close_worker)

if __name__ == "__main__":
    import sys

    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("Use backend.py --worker with one segment request JSON object per stdin line.")
    _worker_main()
