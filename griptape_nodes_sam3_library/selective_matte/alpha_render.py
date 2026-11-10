"""Selective-alpha rendering from SAM selections, ViTMatte edges and an optional whole-frame alpha.

RGB and masks travel through one reversible canvas transform. The whole-frame
alpha can come from any matting model; selection happens in the final blend.
"""

from __future__ import annotations

import gc
import json
import shutil
import uuid
from pathlib import Path

from .preprocessing import pad_mask, pad_rgb, plan_canvas, plan_custom_resize, restore_mask


def _cancel(cancel):
    if cancel and (cancel() if callable(cancel) else cancel.is_set()):
        raise InterruptedError("Alpha rendering cancelled.")


def _frames(path):
    import cv2
    import numpy as np
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as im:
            yield np.asarray(im.convert("RGB")).copy()
        return
    except (UnidentifiedImageError, OSError):
        pass
    reader = cv2.VideoCapture(str(path))
    if not reader.isOpened():
        raise ValueError(f"Could not open {Path(path).name} for alpha rendering.")
    try:
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        reader.release()


class ViTMatteRefiner:
    """Small lazy model, released before the node returns."""

    def __init__(self, model_id="hustvl/vitmatte-base-composition-1k", erode=10, dilate=20):
        import torch
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = VitMatteImageProcessor.from_pretrained(model_id)
        self.model = VitMatteForImageMatting.from_pretrained(model_id).to(self.device).eval()
        self.erode, self.dilate = erode, dilate

    def __call__(self, rgb, selection):
        import cv2
        import numpy as np
        from PIL import Image

        foreground = (selection > 0.5).astype(np.uint8)
        if not foreground.any():
            return np.zeros(foreground.shape, np.float32)
        kernel = lambda radius: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)  # noqa: E731
        inner = cv2.erode(foreground, kernel(self.erode))
        outer = cv2.dilate(foreground, kernel(self.dilate))
        trimap = np.zeros(foreground.shape, np.uint8)
        trimap[outer > 0], trimap[inner > 0] = 128, 255
        ys, xs = np.where(outer)
        x0, x1 = max(0, int(xs.min()) - 32), min(rgb.shape[1], int(xs.max()) + 33)
        y0, y1 = max(0, int(ys.min()) - 32), min(rgb.shape[0], int(ys.max()) + 33)
        inputs = self.processor(
            images=Image.fromarray(rgb[y0:y1, x0:x1]),
            trimaps=Image.fromarray(trimap[y0:y1, x0:x1]),
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.inference_mode():
            prediction = self.model(**inputs).alphas[0, 0, : y1 - y0, : x1 - x0].float().cpu().numpy()
        result = np.zeros(foreground.shape, np.float32)
        result[y0:y1, x0:x1] = prediction
        result[trimap == 0], result[trimap == 255] = 0, 1
        return np.clip(result, 0, 1)

    def close(self):
        self.model = None
        self.processor = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def blend_selected(full_alpha, selection, refined, *, reach=48, contact_radius=16):
    """Keep whole-frame alpha detail within an expanded selection; use refined contact edges.

    A hard SAM multiplication clips flyaway hair. Instead a soft outer envelope
    leaves room for that detail, while high-opacity competing foreground identifies
    boundaries where a whole-frame alpha cannot distinguish the selected object.
    """
    import cv2
    import numpy as np

    s = (selection > 0.5).astype(np.uint8)
    if not s.any():
        return np.zeros(s.shape, np.float32)
    alpha = np.clip(full_alpha.astype(np.float32), 0, 1)
    outside_distance = cv2.distanceTransform(1 - s, cv2.DIST_L2, 5)
    envelope = np.clip((reach - outside_distance) / max(1, reach * 0.35), 0, 1)
    candidate = alpha * envelope
    # A whole-frame opaque region immediately outside the selection is ambiguous:
    # it may be an adjacent subject. Consult the trimap-derived matte there, not
    # a wide dilated SAM gate (which would retain a strip of the neighbour).
    # Soft strands remain eligible outside the SAM outline.
    other = ((alpha > 0.95) & (outside_distance > 1)).astype(np.uint8)
    if other.any():
        distance_to_other = cv2.distanceTransform(1 - other, cv2.DIST_L2, 5)
        contact = np.clip(1 - distance_to_other / max(1, contact_radius * 2), 0, 1)
        candidate = candidate * (1 - contact) + refined * contact
    # Fall back per connected selected component when the whole-frame alpha omits an object.
    count, labels = cv2.connectedComponents(s)
    for label in range(1, count):
        component = labels == label
        if float(alpha[component].mean()) < 0.35:
            expanded = cv2.dilate(component.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            candidate[expanded] = refined[expanded]
    return np.clip(candidate, 0, 1)


def _as_alpha(frame, width, height):
    """Whole-frame alpha from a decoded RGB frame, at the source's dimensions."""
    import numpy as np

    from .preprocessing import _resize

    a = np.asarray(frame)
    a = a.astype(np.float32) / np.iinfo(a.dtype).max if np.issubdtype(a.dtype, np.integer) else a.astype(np.float32)
    if a.ndim == 3:
        a = a[..., :3].mean(-1)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("The connected Alpha video has an invalid frame.")
    if abs(a.shape[1] / a.shape[0] - width / height) > 0.01 * width / height:
        raise ValueError(
            f"The Alpha video is {a.shape[1]}×{a.shape[0]}, which doesn't match the Source's {width}×{height} framing."
        )
    return np.clip(_resize(a, (width, height)), 0, 1)


def render_alpha(
    source_path,
    mask_sequence_dir,
    output_dir,
    alpha_path=None,
    canvas_width=1920,
    canvas_height=1088,
    progress=None,
    cancel=None,
    selected_frame=0,
    _refiner=None,
    output_size="processing",
    resize_width=None,
    resize_height=None,
):
    root = Path(output_dir).resolve()
    run = root / ("alpha_" + uuid.uuid4().hex[:12])
    try:
        return _render_alpha_impl(
            source_path,
            mask_sequence_dir,
            alpha_path,
            canvas_width,
            canvas_height,
            progress,
            cancel,
            selected_frame,
            _refiner,
            _run_dir=run,
            output_size=output_size,
            resize_width=resize_width,
            resize_height=resize_height,
        )
    except BaseException:
        # This invocation created this unique directory. Never delete a source,
        # supplied output root, previous render or a path outside that root.
        if run.resolve().parent == root and run.name.startswith("alpha_") and run.is_dir():
            shutil.rmtree(run)
        raise


def _render_alpha_impl(
    source_path,
    mask_sequence_dir,
    alpha_path=None,
    canvas_width=1920,
    canvas_height=1088,
    progress=None,
    cancel=None,
    selected_frame=0,
    _refiner=None,
    _run_dir=None,
    output_size="processing",
    resize_width=None,
    resize_height=None,
):
    """Track first, then render matching Plate/Mask at the unpadded processing size.

    Without ``alpha_path`` the matte is the ViTMatte-refined selection. With it,
    each frame of that whole-frame alpha is blended into the selection.
    The review MP4 is a proxy. The 16-bit PNG sequence is the authoritative alpha.
    No source images or flow files are overwritten.
    """
    import numpy as np
    from PIL import Image

    from .backend import _write_preview_video, unload_model
    from .preprocessing import probe_media

    if output_size not in ("original", "processing"):
        raise ValueError("Output size must be original or processing.")
    media = probe_media(str(source_path))
    n = media["num_frames"]
    if alpha_path is not None:
        alpha_frames = probe_media(str(alpha_path))["num_frames"]
        if alpha_frames != n:
            raise ValueError(f"The Alpha video has {alpha_frames} frames; the Source has {n}.")
    if resize_width is not None or resize_height is not None:
        plan = plan_custom_resize(media["width"], media["height"], resize_width, resize_height)
    else:
        plan = plan_canvas(media["width"], media["height"], canvas_width, canvas_height)
    output_media = dict(media)
    if output_size == "processing":
        output_media.update(width=plan.content_width, height=plan.content_height)
    masks = Path(mask_sequence_dir)
    for frame in range(n):
        if not (masks / f"mask_{frame:06d}.png").is_file():
            raise ValueError(f"Selection mask for source frame {frame} is missing. Track the complete clip first.")
    _cancel(cancel)
    run = Path(_run_dir)
    prepared, alpha_dir = run / "prepared", run / "alpha"
    for folder in (prepared, alpha_dir):
        folder.mkdir(parents=True, exist_ok=False)
    notify = lambda message, fraction: progress(fraction, message) if progress else None  # noqa: E731
    unload_model()  # SAM and ViTMatte must not occupy GPU memory concurrently.
    refiner = _refiner or ViTMatteRefiner()
    whole = _frames(str(alpha_path)) if alpha_path is not None else None
    try:
        count = 0
        for frame, rgb in enumerate(_frames(str(source_path))):
            _cancel(cancel)
            if frame >= n:
                raise ValueError("Decoded source frame count changed during rendering.")
            with Image.open(masks / f"mask_{frame:06d}.png") as im:
                selection = np.asarray(im.convert("L"), dtype=np.float32) / 255
            if selection.shape != rgb.shape[:2]:
                raise ValueError("SAM mask and original plate dimensions do not match.")
            # Run the matting model at the processing scale.
            rgb_canvas, selection_canvas = pad_rgb(rgb, plan), pad_mask(selection, plan)
            refined = np.asarray(refiner(rgb_canvas, selection_canvas), dtype=np.float32)
            if refined.shape != (plan.height, plan.width) or not np.isfinite(refined).all():
                raise ValueError("ViTMatte returned an invalid or misaligned alpha frame.")
            refined = np.clip(refined, 0, 1)
            if whole is not None:
                whole_frame = next(whole, None)
                if whole_frame is None:
                    raise ValueError("The Alpha video ended before the Source.")
                full = pad_mask(_as_alpha(whole_frame, plan.original_width, plan.original_height), plan)
                selected = blend_selected(full, selection_canvas, refined)
            else:
                selected = refined
            Image.fromarray(rgb_canvas).save(prepared / f"rgb_{frame:06d}.png")
            original = restore_mask(selected, plan, output_size=output_size)
            Image.fromarray((original * 65535).round().astype(np.uint16)).save(alpha_dir / f"alpha_{frame:06d}.png")
            count += 1
            notify(f"Rendering alpha {frame + 1}/{n}", 0.9 * count / n)
        if count != n:
            raise ValueError(f"Source decoded {count} frames; expected {n}. No partial matte was published.")
    finally:
        if whole is not None:
            whole.close()
        if hasattr(refiner, "close"):
            refiner.close()
        del refiner
        gc.collect()
    # Encode a review proxy without confusing it with the 16-bit final deliverable.
    proxy_dir = run / "review_masks"
    proxy_dir.mkdir()
    for frame in range(n):
        with Image.open(alpha_dir / f"alpha_{frame:06d}.png") as im:
            a = np.asarray(im, np.float32) / 65535
        Image.fromarray((a * 255).round().astype(np.uint8)).save(proxy_dir / f"mask_{frame:06d}.png")
    video, warning = _write_preview_video(proxy_dir, run / "alpha_review.mp4", output_media, cancel)
    warnings = [warning] if warning else []
    plate_video = None
    if output_size == "processing":
        notify("Resizing Plate to match Mask", 0.97)
        plate_video, plate_warning = _write_preview_video(
            prepared,
            run / "plate_review.mp4",
            output_media,
            cancel,
            frame_pattern="rgb_{index:06d}.png",
            crop_box=(
                plan.pad_left,
                plan.pad_top,
                plan.pad_left + plan.content_width,
                plan.pad_top + plan.content_height,
            ),
        )
        if plate_warning:
            warnings.append("Resized Plate: " + plate_warning)
    selected_frame = min(n - 1, max(0, int(selected_frame)))
    result = dict(
        alpha_video_path=video,
        plate_video_path=plate_video,
        output_size=output_size,
        source_width=media["width"],
        source_height=media["height"],
        alpha_sequence_dir=str(alpha_dir),
        preview_mask_path=str(alpha_dir / f"alpha_{selected_frame:06d}.png"),
        selected_frame=selected_frame,
        canvas_plan=plan.to_dict(),
        warnings=warnings,
        **output_media,
    )
    (run / "manifest.json").write_text(
        json.dumps(
            dict(
                result,
                source_path=str(source_path),
                whole_frame_alpha_path=str(alpha_path) if alpha_path is not None else None,
                alpha_pattern="alpha/alpha_%06d.png",
                alpha_bit_depth=16,
                frame_index_base=0,
                model="SAM 3.1 + ViTMatte" + (" + connected whole-frame alpha" if alpha_path is not None else ""),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    notify("Selective alpha complete", 1.0)
    return result
