"""Selective Matte: pick a subject with PointPicker, track it with SAM 3.1, and render its alpha with LTX Alpha-Gen.

Ported from Amaru Zeas-Sigüenza's Griptape Selective Matte library.
"""

from __future__ import annotations

import hashlib
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from griptape.artifacts import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.events.execution_events import ResolveNodeRequest, ResolveNodeResultFailure
from griptape_nodes.retained_mode.events.static_file_events import (
    CreateStaticFileDownloadUrlFromPathRequest,
    CreateStaticFileDownloadUrlResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.widget import Widget

# PointPicker comes from the Point Select library, which this library declares as a dependency.
WIDGET_LIBRARY = "Griptape Nodes Point Select Library"
WIDGET_NAME = "PointPicker"
TITLE = "Selective Matte"
ACTIONS = [
    {"id": "load", "label": "Load source", "title": "Run the source node and show its first frame"},
    {"id": "preview", "label": "Preview mask", "title": "SAM 3.1 mask on this frame", "needsInclude": True},
    {"id": "track", "label": "Track subject", "title": "Track the subject through the video", "needsInclude": True},
    {
        "id": "render",
        "label": "Render alpha",
        "title": "Track, then render the matte with Alpha-Gen",
        "primary": True,
        "needsInclude": True,
    },
]
_WORK_LOCK = threading.Lock()


def default_state() -> dict[str, Any]:
    return {
        "sourceUrl": "",
        "sourceKind": "video",
        "sourceId": "",
        "width": 0,
        "height": 0,
        "fps": 24,
        "numFrames": 0,
        "frame": 0,
        "points": [],
        "maskUrl": "",
        "maskFrame": 0,
        "maskRevision": 0,
        "status": "Connect a video to begin.",
        "error": "",
        "canvasPlan": None,
        "action": "",
        "requestId": "",
        "busy": False,
        "progress": 0.0,
        "pointsRevision": 0,
        "settingsRevision": 0,
        "maskKind": "selection",
        "maskSettingsRevision": 0,
        "title": TITLE,
        "actions": ACTIONS,
    }


def source_reference(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("value", "url", "path"):
            if isinstance(value.get(key), str):
                return value[key].strip()
    ref = getattr(value, "value", None)
    return ref.strip() if isinstance(ref, str) else ""


def browser_url(path: str) -> str:
    result = GriptapeNodes.handle_request(
        CreateStaticFileDownloadUrlFromPathRequest(file_path=path, preview=False, broadcast_result=False)
    )
    if isinstance(result, CreateStaticFileDownloadUrlResultSuccess) and result.url:
        return result.url
    raise ValueError(str(getattr(result, "result_details", "Could not display this media file.")))


class SelectiveMatte(ControlNode):
    """Keeps source selection, SAM tracking and Alpha rendering in one widget."""

    def __init__(self, name: str, metadata: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self._ready = False
        self._callback_state = threading.local()
        self._dispatching = False
        self._running = False
        self._last_request = ""
        self._track_result: dict[str, Any] | None = None
        self._track_key = ""
        self._points_key = ""
        self._result_lock = threading.RLock()
        self._result_revision = 0
        merged = {
            "category": "video/matting",
            "description": "Select a subject, track with SAM3.1, and render Alpha-Gen.",
        }
        merged.update(metadata or {})
        super().__init__(name=name, metadata=merged, **kwargs)
        self.set_initial_node_size(width=960, height=820)
        self.add_parameter(
            Parameter(
                name="source",
                type="VideoUrlArtifact",
                input_types=["VideoUrlArtifact", "VideoArtifact", "str"],
                allowed_modes={ParameterMode.INPUT},
                tooltip="Original video. Framing is preserved; final Plate and Mask use the processing resolution.",
                ui_options={"display_name": "Source", "hide_property": True},
            )
        )
        self.add_parameter(
            Parameter(
                name="pipeline",
                type="Pipeline Config",
                allowed_modes={ParameterMode.INPUT},
                tooltip="Connect the LoRA Pipeline output from your working LTX-2.5 Alpha-Gen workflow.",
                ui_options={"display_name": "Alpha-Gen Pipeline", "hide_property": True},
            )
        )
        with ParameterGroup(name="Resize plate", collapsed=False) as resize:
            Parameter(
                name="custom_resize",
                type="bool",
                default_value=False,
                allowed_modes={ParameterMode.PROPERTY},
                ui_options={"display_name": "Custom resize"},
                tooltip="Resize the source for ViTMatte and LTX. Plate and Mask stay at this size; aspect ratio is preserved.",
            )
            Parameter(
                name="resize_width",
                type="int",
                default_value=960,
                allowed_modes={ParameterMode.PROPERTY},
                ui_options={"display_name": "Fit width"},
                tooltip="Maximum content width, 32–1920. Model padding is added automatically. Only used when Custom resize is enabled.",
            )
            Parameter(
                name="resize_height",
                type="int",
                default_value=540,
                allowed_modes={ParameterMode.PROPERTY},
                ui_options={"display_name": "Fit height"},
                tooltip="Maximum content height, 32–1088. Fit without stretching, cropping, or enlarging the source.",
            )
        self.add_node_element(resize)
        self.add_parameter(
            Parameter(
                name="selection",
                type="dict",
                input_types=["dict"],
                default_value=default_state(),
                allowed_modes={ParameterMode.PROPERTY},
                ui_options={"is_full_width": True, "display_name": "Selective Matte"},
                traits={Widget(name=WIDGET_NAME, library=WIDGET_LIBRARY)},
            )
        )
        with ParameterGroup(name="Processing settings", collapsed=True) as settings:
            Parameter(
                name="source_path",
                type="str",
                default_value="",
                tooltip="Optional local/project path instead of a Source wire.",
            )
            Parameter(
                name="quality",
                type="str",
                default_value="fast",
                traits={Options(choices=["fast", "isolated"])},
                tooltip="Fast restricts Alpha-Gen with the selected subject. Isolated prepares only the selected foreground before Alpha-Gen.",
            )
            Parameter(
                name="sam_checkpoint",
                type="str",
                default_value="",
                tooltip="Optional installed SAM3.1 multiplex checkpoint path.",
            )
            Parameter(
                name="sam_python",
                type="str",
                default_value="",
                tooltip="Optional separate SAM3.1 Python executable. Empty uses this library's execution environment.",
            )
            Parameter(
                name="canvas_width",
                type="int",
                default_value=1920,
                tooltip="Processing canvas width when Custom resize is off.",
            )
            Parameter(
                name="canvas_height",
                type="int",
                default_value=1088,
                tooltip="Processing canvas height when Custom resize is off; padding is removed from final outputs.",
            )
            Parameter(name="chunk_size", type="int", default_value=49, tooltip="Frames per LTX call (8n+1, 9 to 145).")
            Parameter(
                name="overlap", type="int", default_value=9, tooltip="Overlapping frames blended between LTX calls."
            )
            Parameter(name="seed", type="int", default_value=42)
            Parameter(
                name="output_directory",
                type="str",
                default_value="{outputs}/sam3_1",
                tooltip="Each render gets its own directory; existing results are kept.",
            )
            for key, label, kind, default in (
                ("alpha_sequence", "Final alpha PNG folder", "str", ""),
                ("mask_sequence", "SAM diagnostic PNG folder", "str", ""),
                ("report", "Processing report", "dict", {}),
            ):
                Parameter(
                    name=key,
                    type=kind,
                    default_value=default,
                    allowed_modes={ParameterMode.PROPERTY},
                    settable=False,
                    ui_options={"display_name": label},
                )
        self.add_node_element(settings)
        self.add_parameter(
            Parameter(
                name="operation",
                type="str",
                default_value="preview",
                allowed_modes={ParameterMode.PROPERTY},
                ui_options={"hide": True},
            )
        )
        for key, label, kind in (("plate", "Plate", "VideoUrlArtifact"), ("mask", "Mask", "VideoUrlArtifact")):
            self.add_parameter(
                Parameter(
                    name=key,
                    output_type=kind,
                    allowed_modes={ParameterMode.OUTPUT},
                    settable=False,
                    ui_options={"display_name": label, "hide_property": True},
                )
            )
        self._ready = True

    def set_parameter_value(
        self,
        param_name: str,
        value: Any,
        *,
        initial_setup: bool = False,
        emit_change: bool = True,
        skip_before_value_set: bool = False,
    ) -> None:
        previous = getattr(self._callback_state, "initial_setup", False)
        self._callback_state.initial_setup = previous or initial_setup
        try:
            # Saved requests are history, never commands to replay when loading a graph.
            if initial_setup and param_name == "selection" and isinstance(value, dict):
                value = {**value, "action": "", "busy": False}
            super().set_parameter_value(
                param_name,
                value,
                initial_setup=initial_setup,
                emit_change=emit_change,
                skip_before_value_set=skip_before_value_set,
            )
        finally:
            self._callback_state.initial_setup = previous

    @contextmanager
    def _internal_updates(self):
        # Engine callbacks can arrive on a different thread during publication.
        # Suppress only our own recursive setters; UI edits must still invalidate results.
        depth = getattr(self._callback_state, "internal_depth", 0)
        self._callback_state.internal_depth = depth + 1
        try:
            yield
        finally:
            self._callback_state.internal_depth = depth

    def _widget_state(self) -> dict[str, Any]:
        result = default_state()
        current = self.get_parameter_value("selection")
        if isinstance(current, dict):
            result.update(current)
        return result

    def _push(self, **updates: Any) -> None:
        state = self._widget_state()
        state.update(updates)
        with self._internal_updates():
            self.set_parameter_value("selection", state)
            self.publish_update_to_parameter("selection", state)

    def _output(self, name: str, value: Any) -> None:
        if name in ("plate", "mask"):
            self.parameter_output_values[name] = value
        else:
            with self._internal_updates():
                self.set_parameter_value(name, value)
        self.publish_update_to_parameter(name, value)

    def _invalidate_final(self) -> None:
        with self._result_lock:
            self._result_revision += 1
            self._output("mask", None)
            self._output("alpha_sequence", "")
            self._output("report", {"status": "Render alpha to create a current Mask output."})
            state = self._widget_state()
            changes: dict[str, Any] = {"settingsRevision": int(state.get("settingsRevision", 0)) + 1}
            if state.get("maskKind") == "final":
                changes.update(maskUrl="", maskKind="selection")
            self._push(**changes)

    def _output_root(self) -> Path:
        root = Path(File(str(self.get_parameter_value("output_directory") or "{outputs}/sam3_1")).resolve())
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _source_path(self) -> str:
        value = self.get_parameter_value("source")
        reference = source_reference(value) or str(self.get_parameter_value("source_path") or "").strip()
        if reference:
            resolved = File(reference).resolve()
            if not resolved.startswith(("http://", "https://")):
                path = Path(resolved)
                if not path.is_file():
                    raise ValueError(f"Source file not found: {path}")
                return str(path)
            # Materialize remote artifacts once; source references can still be displayed by the engine.
            data = File(reference).read_bytes()
            suffix = Path(reference.split("?", 1)[0]).suffix or ".mp4"
        else:
            data = value.get("value") if isinstance(value, dict) else getattr(value, "value", None)
            if not isinstance(data, (bytes, bytearray)):
                raise ValueError("Connect a video or supply a Source path.")
            suffix = "." + str(getattr(value, "format", "mp4")).lstrip(".")
        name = hashlib.sha256(data).hexdigest()[:20] + suffix
        path = self._output_root() / "sources" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        return str(path)

    def _refresh_source(self) -> tuple[str, dict[str, Any]]:
        from selective_matte.preprocessing import plan_canvas, plan_custom_resize, probe_media

        path = self._source_path()
        stat = Path(path).stat()
        identity = hashlib.sha256(f"{path}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()[:24]
        info = probe_media(path)
        if info["kind"] != "video":
            raise ValueError("This node requires a video source so Plate and Mask connect directly to Frame Compare.")
        if self.get_parameter_value("custom_resize"):
            plan = plan_custom_resize(
                int(info["width"]),
                int(info["height"]),
                self.get_parameter_value("resize_width"),
                self.get_parameter_value("resize_height"),
            )
        else:
            plan = plan_canvas(
                int(info["width"]),
                int(info["height"]),
                int(self.get_parameter_value("canvas_width") or 1920),
                int(self.get_parameter_value("canvas_height") or 1088),
            )
        changed = self._widget_state().get("sourceId") != identity
        updates = {
            "sourceId": identity,
            "sourceUrl": browser_url(path),
            "sourceKind": info["kind"],
            "width": info["width"],
            "height": info["height"],
            "fps": info["fps"],
            "numFrames": info["num_frames"],
            "canvasPlan": plan.widget_dict(),
            "error": "",
            # PointPicker doesn't draw canvasPlan, so the header carries the output size.
            "title": f"{TITLE} · {plan.content_width}×{plan.content_height}",
            "outputWidth": plan.content_width,
            "outputHeight": plan.content_height,
        }
        if changed:
            self._track_result = None
            self._track_key = ""
            updates.update(points=[], frame=0, maskUrl="", maskRevision=0, status="Click the subject to include it.")
            self._invalidate_final()
            self._output("mask_sequence", "")
        source = self.get_parameter_value("source")
        # Preview uses original pixels for picking. Once a final Mask exists,
        # keep its matching resized Plate even when previewing another frame.
        if self.parameter_output_values.get("mask") is None:
            self._output("plate", source if isinstance(source, VideoUrlArtifact) else VideoUrlArtifact(path))
        self._push(**updates)
        return path, info

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if (
            not self._ready
            or getattr(self._callback_state, "internal_depth", 0)
            or getattr(self._callback_state, "initial_setup", False)
        ):
            return super().after_value_set(parameter, value)
        if parameter.name in (
            "source",
            "source_path",
            "canvas_width",
            "canvas_height",
            "custom_resize",
            "resize_width",
            "resize_height",
        ):
            self._invalidate_final()
            if self._running:
                self.request_cancellation()
            try:
                self._refresh_source()
            except Exception as exc:
                if parameter.name in ("source", "source_path"):
                    self._output("plate", None)
                    self._track_result = None
                    self._track_key = ""
                    self._push(sourceId="", points=[], maskUrl="")
                self._push(error=str(exc), status="Source needs attention.", sourceUrl="", title=TITLE)
        elif parameter.name == "selection" and isinstance(value, dict):
            import json

            points_key = json.dumps(value.get("points", []), sort_keys=True)
            if points_key != self._points_key:
                self._points_key = points_key
                self._invalidate_final()
                if self._running:
                    self.request_cancellation()
            action = value.get("action")
            request = str(value.get("requestId") or "")
            if action == "cancel":
                self.request_cancellation()
                self._push(action="", status="Cancelling…")
            elif action in ("load", "preview", "track", "render") and request and request != self._last_request:
                if self._running or self._dispatching:
                    self._push(action="", error="Wait for the current operation or cancel it first.")
                else:
                    self._last_request = request
                    self.set_parameter_value("operation", action)
                    self._push(action="", error="", status=f"Starting {action}…")
                    self._dispatching = True
                    try:
                        resolved = GriptapeNodes.handle_request(ResolveNodeRequest(node_name=self.name))
                        if isinstance(resolved, ResolveNodeResultFailure) or (
                            callable(getattr(resolved, "succeeded", None)) and not resolved.succeeded()
                        ):
                            self._push(
                                error=str(resolved.result_details), busy=False, status="Could not start operation."
                            )
                    except Exception as exc:
                        self._push(error=str(exc), busy=False, status="Could not start operation.")
                    finally:
                        self._dispatching = False
        elif parameter.name in ("pipeline", "quality", "sam_checkpoint", "sam_python", "chunk_size", "overlap", "seed"):
            self._invalidate_final()
            if self._running:
                self.request_cancellation()
        return super().after_value_set(parameter, value)

    def after_incoming_connection_removed(
        self, source_node: Any, source_parameter: Parameter, target_parameter: Parameter
    ) -> None:
        if target_parameter.name == "source":
            self.set_parameter_value("source", None)
        elif target_parameter.name == "pipeline":
            self.set_parameter_value("pipeline", None)
        return super().after_incoming_connection_removed(source_node, source_parameter, target_parameter)

    def _progress(self, value: float, message: str) -> None:
        if self.is_cancellation_requested:
            return
        self._push(progress=max(0.0, min(1.0, float(value))), status=message)

    def process(self) -> AsyncResult:
        if self._running:
            raise RuntimeError("This node is already processing. Cancel or wait for it to finish.")
        self.clear_cancellation()
        yield self._work

    def _work(self) -> None:
        if not _WORK_LOCK.acquire(blocking=False):
            raise RuntimeError("Another SAM3.1/Alpha operation is running. Wait for it to finish.")
        self._running = True
        self._push(busy=True, error="", progress=0.0)
        try:
            path, info = self._refresh_source()
            state = self._widget_state()
            action = str(self.get_parameter_value("operation") or "preview")
            if action == "load":
                self._push(status="Click the subject to include it.", progress=1.0)
                return
            if action not in ("preview", "track", "render"):
                raise ValueError("Choose Load source, Preview, Track or Render alpha.")
            import json

            from selective_matte import backend

            points = state.get("points", [])
            if not isinstance(points, list) or not points:
                raise ValueError("Add an include point on the subject first.")
            if action == "render" and self.get_parameter_value("pipeline") is None:
                raise ValueError("Connect the LoRA Pipeline output from your existing Alpha-Gen workflow.")
            checkpoint = str(self.get_parameter_value("sam_checkpoint") or "").strip()
            if checkpoint:
                checkpoint = File(checkpoint).resolve()
            sam_python = str(self.get_parameter_value("sam_python") or "").strip()
            if sam_python:
                sam_python = File(sam_python).resolve()
            key = hashlib.sha256(
                json.dumps([state["sourceId"], points, checkpoint, sam_python], sort_keys=True).encode()
            ).hexdigest()
            workdir = self._output_root() / uuid.uuid4().hex[:12]
            workdir.mkdir(parents=True, exist_ok=True)
            if action == "preview" or self._track_result is None or self._track_key != key:
                segment = (
                    backend.segment
                    if not sam_python
                    else lambda *a, **kw: backend.segment_via_worker(sam_python, *a, **kw)
                )
                result = segment(
                    path,
                    points,
                    checkpoint_path=checkpoint or None,
                    mode="preview" if action == "preview" else "track",
                    selected_frame=int(state.get("frame", 0)),
                    output_dir=str(workdir / "sam"),
                    progress=self._progress,
                    cancel=lambda: self.is_cancellation_requested,
                    device="cuda",
                    make_preview_video=action != "preview",
                )
                if action != "preview":
                    self._track_result, self._track_key = result, key
            else:
                result = self._track_result
            if self.is_cancellation_requested:
                raise InterruptedError("Operation cancelled.")
            preview_path = result.get("preview_mask_path")
            selected_preview = Path(result.get("mask_sequence_dir", "")) / f"mask_{int(state.get('frame', 0)):06d}.png"
            if selected_preview.is_file():
                preview_path = str(selected_preview)
            if preview_path:
                self._push(
                    maskUrl=browser_url(preview_path),
                    maskFrame=int(state.get("frame", 0)),
                    maskRevision=int(state.get("pointsRevision", 0)),
                    maskKind="selection",
                )
            self._output("mask_sequence", result.get("mask_sequence_dir", ""))
            if action == "render":
                from selective_matte.alpha_render import render_alpha

                backend.unload_model()
                if sam_python:
                    backend.close_worker(sam_python)
                self._invalidate_final()
                render_revision = self._result_revision
                settings_revision = int(self._widget_state().get("settingsRevision", 0))
                rendered: dict[str, Any] = render_alpha(
                    self.get_parameter_value("pipeline"),
                    path,
                    result["mask_sequence_dir"],
                    str(workdir / "alpha"),
                    canvas_width=int(self.get_parameter_value("canvas_width") or 1920),
                    canvas_height=int(self.get_parameter_value("canvas_height") or 1088),
                    chunk_size=int(self.get_parameter_value("chunk_size") or 49),
                    overlap=int(self.get_parameter_value("overlap") or 9),
                    seed=int(self.get_parameter_value("seed") or 0),
                    quality=str(self.get_parameter_value("quality") or "fast"),
                    output_size="processing",
                    resize_width=self.get_parameter_value("resize_width")
                    if self.get_parameter_value("custom_resize")
                    else None,
                    resize_height=self.get_parameter_value("resize_height")
                    if self.get_parameter_value("custom_resize")
                    else None,
                    selected_frame=int(state.get("frame", 0)),
                    progress=self._progress,
                    cancel=lambda: self.is_cancellation_requested,
                    node_name=self.name,
                )
                if self.is_cancellation_requested:
                    raise InterruptedError("Operation cancelled.")
                final_preview_url = browser_url(rendered["preview_mask_path"])
                with self._result_lock:
                    if self.is_cancellation_requested or self._result_revision != render_revision:
                        raise InterruptedError("Source or settings changed before the alpha could be published.")
                    self._output("alpha_sequence", rendered.get("alpha_sequence_dir", ""))
                    self._output("report", rendered)
                    if not rendered.get("alpha_video_path"):
                        details = "; ".join(str(warning) for warning in rendered.get("warnings", []))
                        raise RuntimeError(
                            "Alpha PNG frames were saved, but the Mask video could not be encoded. "
                            "See Final alpha PNG folder and Processing report; check the PyAV/video codec installation. "
                            + details
                        )
                    if not rendered.get("plate_video_path"):
                        raise RuntimeError(
                            "Alpha frames were saved, but the matching resized Plate could not be encoded. "
                            "See Processing report; Plate and Mask were not published as a mismatched pair."
                        )
                    self._push(
                        status=f"Alpha complete: {rendered.get('width')} × {rendered.get('height')}, original frame rate.",
                        progress=1.0,
                        maskUrl=final_preview_url,
                        maskFrame=rendered.get("selected_frame", 0),
                        maskRevision=int(state.get("pointsRevision", 0)),
                        maskKind="final",
                        maskSettingsRevision=settings_revision,
                    )
                    if self.is_cancellation_requested or self._result_revision != render_revision:
                        self._invalidate_final()
                        raise InterruptedError("Source or settings changed before the alpha could be published.")
                    self._output("plate", VideoUrlArtifact(rendered["plate_video_path"]))
                    self._output("mask", VideoUrlArtifact(rendered["alpha_video_path"]))
            else:
                self._output("report", {"operation": action, "source": info, "warnings": result.get("warnings", [])})
                self._push(
                    status="Preview ready." if action == "preview" else "Subject tracking complete.", progress=1.0
                )
        except Exception as exc:
            self._push(
                error=str(exc), status="Cancelled." if isinstance(exc, InterruptedError) else "Operation failed."
            )
            raise
        finally:
            self._running = False
            self._push(busy=False, action="")
            _WORK_LOCK.release()
