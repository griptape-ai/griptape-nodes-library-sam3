"""Independent selective-alpha orchestration for the existing GT LTX pipeline.

RGB and masks travel through one reversible canvas transform. SAM is never
misrepresented as an Alpha-Gen input mask: selection happens in the final blend,
or by removing competing foreground from a second RGB reference pass.
"""

from __future__ import annotations

import gc
import json
import shutil
import uuid
from pathlib import Path

from .preprocessing import pad_mask, pad_rgb, plan_canvas, plan_chunks, plan_custom_resize, restore_mask


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
        raise ValueError("Could not open the source clip for Alpha-Gen.")
    try:
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        reader.release()


class ViTMatteRefiner:
    """Small lazy model with an explicit release before the large LTX pass."""

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
    """Keep LTX detail within an expanded selection; use refined contact edges.

    A hard SAM multiplication clips flyaway hair. Instead a soft outer envelope
    leaves room for LTX detail, while high-opacity competing foreground identifies
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
    # Soft LTX strands remain eligible outside the SAM outline.
    other = ((alpha > 0.95) & (outside_distance > 1)).astype(np.uint8)
    if other.any():
        distance_to_other = cv2.distanceTransform(1 - other, cv2.DIST_L2, 5)
        contact = np.clip(1 - distance_to_other / max(1, contact_radius * 2), 0, 1)
        candidate = candidate * (1 - contact) + refined * contact
    # Fall back per connected selected component when Alpha-Gen omits an object.
    count, labels = cv2.connectedComponents(s)
    for label in range(1, count):
        component = labels == label
        if float(alpha[component].mean()) < 0.35:
            expanded = cv2.dilate(component.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            candidate[expanded] = refined[expanded]
    return np.clip(candidate, 0, 1)


def _isolated_rgb(rgb, full_alpha, selection):
    """Remove competing opaque foreground using surrounding RGB background."""
    import cv2
    import numpy as np

    target = (selection > 0.5).astype(np.uint8)
    safe_target = cv2.dilate(target, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    remove = ((full_alpha > 0.7) & (safe_target == 0)).astype(np.uint8)
    remove = cv2.dilate(remove, np.ones((7, 7), np.uint8)) * (1 - safe_target)
    if not remove.any():
        return rgb
    return cv2.inpaint(rgb, remove * 255, 5, cv2.INPAINT_TELEA)


def _chunk_weights(chunk, index, chunks, overlap):
    import numpy as np

    weight = np.ones(chunk.valid_frames, np.float32)
    if index:
        n = min(overlap, chunk.valid_frames)
        weight[:n] *= np.linspace(1 / (n + 1), n / (n + 1), n)
    if index < len(chunks) - 1:
        n = min(overlap, chunk.valid_frames)
        weight[-n:] *= np.linspace(n / (n + 1), 1 / (n + 1), n)
    return weight


def _as_alpha(frame, width, height):
    import numpy as np

    a = np.asarray(frame)
    if a.shape[:2] != (height, width):
        raise ValueError(f"LTX decoded {a.shape[:2]}, expected {(height, width)}; refusing misaligned alpha.")
    if np.issubdtype(a.dtype, np.integer):
        a = a.astype(np.float32) / np.iinfo(a.dtype).max
    else:
        a = a.astype(np.float32)
    if a.ndim == 3:
        a = a[..., :3].mean(-1)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("LTX produced an invalid alpha frame.")
    return np.clip(a, 0, 1)


def render_alpha(
    pipeline,
    source_path,
    mask_sequence_dir,
    output_dir,
    canvas_width=1920,
    canvas_height=1088,
    chunk_size=49,
    overlap=9,
    seed=42,
    progress=None,
    cancel=None,
    node_name=None,
    quality="fast",
    selected_frame=0,
    _runner=None,
    _refiner=None,
    output_size="processing",
    resize_width=None,
    resize_height=None,
):
    root = Path(output_dir).resolve()
    run = root / ("alpha_" + uuid.uuid4().hex[:12])
    try:
        return _render_alpha_impl(
            pipeline,
            source_path,
            mask_sequence_dir,
            output_dir,
            canvas_width,
            canvas_height,
            chunk_size,
            overlap,
            seed,
            progress,
            cancel,
            node_name,
            quality,
            selected_frame,
            _runner,
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
    pipeline,
    source_path,
    mask_sequence_dir,
    output_dir,
    canvas_width=1920,
    canvas_height=1088,
    chunk_size=49,
    overlap=9,
    seed=42,
    progress=None,
    cancel=None,
    node_name=None,
    quality="fast",
    selected_frame=0,
    _runner=None,
    _refiner=None,
    _run_dir=None,
    output_size="processing",
    resize_width=None,
    resize_height=None,
):
    """Track first, then render matching Plate/Mask at the unpadded processing size.

    The review MP4 is a proxy. The 16-bit PNG sequence is the authoritative alpha.
    No source images or flow files are overwritten.
    """
    import numpy as np
    from PIL import Image

    from .backend import _write_preview_video, unload_model
    from .preprocessing import probe_media

    if quality not in ("fast", "isolated"):
        raise ValueError("Alpha quality must be 'fast' or 'isolated'.")
    if output_size not in ("original", "processing"):
        raise ValueError("Output size must be original or processing.")
    if pipeline is None:
        raise ValueError("Connect the LoRA Pipeline output from your Alpha-Gen workflow to Pipeline.")
    media = probe_media(str(source_path))
    n = media["num_frames"]
    if resize_width is not None or resize_height is not None:
        plan = plan_custom_resize(media["width"], media["height"], resize_width, resize_height)
    else:
        plan = plan_canvas(media["width"], media["height"], canvas_width, canvas_height)
    output_media = dict(media)
    if output_size == "processing":
        output_media.update(width=plan.content_width, height=plan.content_height)
    chunks = plan_chunks(n, chunk_size, overlap)
    masks = Path(mask_sequence_dir)
    for frame in range(n):
        if not (masks / f"mask_{frame:06d}.png").is_file():
            raise ValueError(f"Selection mask for source frame {frame} is missing. Track the complete clip first.")
    _cancel(cancel)
    run = Path(_run_dir)
    prepared, refined_dir, alpha_dir = run / "prepared", run / "refined", run / "alpha"
    for folder in (prepared, refined_dir, alpha_dir):
        folder.mkdir(parents=True, exist_ok=False)
    notify = lambda message, fraction: progress(fraction, message) if progress else None  # noqa: E731
    unload_model()  # SAM and LTX must not occupy GPU memory concurrently.
    refiner = _refiner or ViTMatteRefiner()
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
            # Run the matting model at the same processing scale used by LTX.
            rgb_canvas, selection_canvas = pad_rgb(rgb, plan), pad_mask(selection, plan)
            refined = np.asarray(refiner(rgb_canvas, selection_canvas), dtype=np.float32)
            if refined.shape != (plan.height, plan.width) or not np.isfinite(refined).all():
                raise ValueError("ViTMatte returned an invalid or misaligned alpha frame.")
            refined = np.clip(refined, 0, 1)
            Image.fromarray(rgb_canvas).save(prepared / f"rgb_{frame:06d}.png")
            Image.fromarray((refined * 65535).round().astype(np.uint16)).save(refined_dir / f"refined_{frame:06d}.png")
            count += 1
            notify(f"Refining selection {frame + 1}/{n}", 0.2 * count / n)
        if count != n:
            raise ValueError(f"Source decoded {count} frames; expected {n}. No partial matte was published.")
    finally:
        if hasattr(refiner, "close"):
            refiner.close()
        del refiner
        gc.collect()
    if _runner is None:
        from .ltx_runner import run_alpha as _runner
    accumulator_path = run / "alpha_accumulator.f32"
    acc = np.memmap(accumulator_path, mode="w+", dtype=np.float32, shape=(n, plan.height, plan.width))
    acc[:] = 0
    weights = np.zeros(n, np.float32)
    try:
        for chunk_index, chunk in enumerate(chunks):
            _cancel(cancel)
            input_frames = []
            selections = []
            for frame in range(chunk.start, chunk.start + chunk.valid_frames):
                with Image.open(prepared / f"rgb_{frame:06d}.png") as im:
                    input_frames.append(im.convert("RGB").copy())
                with Image.open(masks / f"mask_{frame:06d}.png") as im:
                    selections.append(pad_mask(np.asarray(im.convert("L"), np.float32) / 255, plan))
            input_frames += [input_frames[-1]] * (chunk.model_frames - chunk.valid_frames)
            notify(f"LTX Alpha-Gen chunk {chunk_index + 1}/{len(chunks)}", 0.2 + 0.65 * chunk_index / len(chunks))
            pass_fraction = 0.5 if quality == "isolated" else 1
            chunk_progress = lambda fraction, message: notify(  # noqa: E731
                message,
                0.2 + 0.65 * (chunk_index + pass_fraction * fraction) / len(chunks),  # noqa: B023
            )
            outputs = _runner(
                pipeline, input_frames, seed=seed, steps=8, node_name=node_name, cancel=cancel, progress=chunk_progress
            )
            if len(outputs) < chunk.valid_frames:
                raise ValueError("LTX returned fewer frames than the source chunk.")
            full = [_as_alpha(f, plan.width, plan.height) for f in outputs[: chunk.valid_frames]]
            if quality == "isolated":
                isolated = [
                    Image.fromarray(_isolated_rgb(np.asarray(input_frames[k]), full[k], selections[k]))
                    for k in range(chunk.valid_frames)
                ]
                isolated += [isolated[-1]] * (chunk.model_frames - chunk.valid_frames)
                notify(
                    f"LTX isolated-subject pass {chunk_index + 1}/{len(chunks)}", 0.2 + 0.65 * chunk_index / len(chunks)
                )
                second_progress = lambda fraction, message: notify(  # noqa: E731
                    message,
                    0.2 + 0.65 * (chunk_index + 0.5 + 0.5 * fraction) / len(chunks),  # noqa: B023
                )
                second = _runner(
                    pipeline, isolated, seed=seed, steps=8, node_name=node_name, cancel=cancel, progress=second_progress
                )
                if len(second) < chunk.valid_frames:
                    raise ValueError("LTX isolated pass returned an incomplete chunk.")
                detail = [_as_alpha(f, plan.width, plan.height) for f in second[: chunk.valid_frames]]
            else:
                detail = full
            temporal_weights = _chunk_weights(chunk, chunk_index, chunks, overlap)
            for k in range(chunk.valid_frames):
                frame = chunk.start + k
                with Image.open(refined_dir / f"refined_{frame:06d}.png") as im:
                    refined = np.asarray(im, np.float32) / 65535
                # Retain the original whole-frame contact boundary information in
                # both modes; the isolated pass supplies background-facing detail.
                selected = blend_selected(detail[k], selections[k], refined)
                if quality == "isolated":
                    full_selected = blend_selected(full[k], selections[k], refined)
                    # Respect a refined boundary wherever another opaque object
                    # makes the full pass ambiguous; otherwise use isolated detail.
                    contact = np.clip(np.abs(full_selected - full[k]) * selections[k], 0, 1)
                    selected = selected * (1 - contact) + full_selected * contact
                acc[frame] += selected * temporal_weights[k]
                weights[frame] += temporal_weights[k]
            del input_frames, outputs, full, detail, selections
        if np.any(weights <= 0):
            raise RuntimeError("An internal chunk gap was detected; no mask was published.")
        for frame in range(n):
            _cancel(cancel)
            original = restore_mask(np.asarray(acc[frame]) / weights[frame], plan, output_size=output_size)
            Image.fromarray((original * 65535).round().astype(np.uint16)).save(alpha_dir / f"alpha_{frame:06d}.png")
            notify(f"Writing output framing {frame + 1}/{n}", 0.85 + 0.1 * (frame + 1) / n)
    finally:
        acc.flush()
        del acc
        accumulator_path.unlink(missing_ok=True)
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
        quality=quality,
        **output_media,
    )
    (run / "manifest.json").write_text(
        json.dumps(
            dict(
                result,
                source_path=str(source_path),
                alpha_pattern="alpha/alpha_%06d.png",
                alpha_bit_depth=16,
                frame_index_base=0,
                model="LTX-2.5 Alpha-Gen + SAM 3.1 + ViTMatte",
                seed=seed,
                chunks=[vars(c) for c in chunks],
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    notify("Selective alpha complete", 1.0)
    return result
