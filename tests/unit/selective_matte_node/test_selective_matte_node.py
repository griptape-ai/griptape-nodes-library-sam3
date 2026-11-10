"""Node lifecycle tests with no engine, models, network or GPU work."""

import importlib.util
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "griptape_nodes_sam3_library"))


def module(name, **values):
    obj = types.ModuleType(name)
    obj.__dict__.update(values)
    return obj


class Group:
    current = None

    def __init__(self, **kw):
        self.parameters = []

    def __enter__(self):
        Group.current = self
        return self

    def __exit__(self, *a):
        Group.current = None


class Param:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        if Group.current is not None:
            Group.current.parameters.append(self)


class Control:
    def __init__(self, name, **kw):
        self.name = name
        self.parameters = {}
        self.values = {}
        self.parameter_output_values = {}
        self.is_cancellation_requested = False
        self._state = object()  # Native BaseNode owns this name.

    def set_initial_node_size(self, **kw):
        pass

    def add_parameter(self, param):
        self.parameters[param.name] = param
        self.values[param.name] = getattr(param, "default_value", None)

    def add_node_element(self, group):
        for p in group.parameters:
            self.add_parameter(p)

    def get_parameter_value(self, name):
        return self.values.get(name)

    def set_parameter_value(self, name, value, **kw):
        self.values[name] = value
        self.after_value_set(self.parameters[name], value)

    def publish_update_to_parameter(self, *a):
        pass

    def after_value_set(self, *a):
        pass

    def after_incoming_connection_removed(self, *a):
        pass

    def request_cancellation(self):
        self.is_cancellation_requested = True

    def clear_cancellation(self):
        self.is_cancellation_requested = False


class Artifact:
    def __init__(self, value):
        self.value = value


class NativeFile:
    def __init__(self, path):
        self.path = path

    def resolve(self):
        return str(Path(self.path).resolve())


class Request:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class Failure:
    result_details = "validation refused"


Engine = types.SimpleNamespace(handle_request=lambda request: None)
Modes = types.SimpleNamespace(INPUT="input", OUTPUT="output", PROPERTY="property")
stubs = {
    "griptape.artifacts": module("artifacts", VideoUrlArtifact=Artifact),
    "griptape_nodes.exe_types.core_types": module("core", Parameter=Param, ParameterGroup=Group, ParameterMode=Modes),
    "griptape_nodes.exe_types.node_types": module("nodes", AsyncResult=object, ControlNode=Control),
    "griptape_nodes.files.file": module("files", File=NativeFile),
    "griptape_nodes.retained_mode.events.execution_events": module(
        "events", ResolveNodeRequest=Request, ResolveNodeResultFailure=Failure
    ),
    "griptape_nodes.retained_mode.events.static_file_events": module(
        "static", CreateStaticFileDownloadUrlFromPathRequest=Request, CreateStaticFileDownloadUrlResultSuccess=object
    ),
    "griptape_nodes.retained_mode.griptape_nodes": module("engine", GriptapeNodes=Engine),
    "griptape_nodes.traits.options": module("options", Options=lambda **kw: object()),
    "griptape_nodes.traits.widget": module("widget", Widget=lambda **kw: object()),
}
path = Path(__file__).resolve().parents[3] / "griptape_nodes_sam3_library" / "selective_matte_node.py"
spec = importlib.util.spec_from_file_location("node_under_test", path)
node_module = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, stubs):
    spec.loader.exec_module(node_module)


class NodeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.mov"
        self.source.write_bytes(b"fixture")
        self.node = node_module.SelectiveMatte("Selective Alpha")
        self.node.set_parameter_value("output_directory", str(self.root / "outputs"))
        self.plate = Artifact(str(self.source))
        self.browser = patch.object(node_module, "browser_url", side_effect=lambda p: "served:" + p)
        self.browser.start()
        self.addCleanup(self.browser.stop)
        self.probe = patch(
            "selective_matte.preprocessing.probe_media",
            return_value={"width": 2560, "height": 1440, "fps": 24, "num_frames": 121, "kind": "video"},
        )
        self.probe.start()
        self.addCleanup(self.probe.stop)
        self.node.set_parameter_value("source", self.plate)

    def selection(self, **changes):
        state = self.node._widget_state()
        state.update(changes)
        self.node.set_parameter_value("selection", state)

    def test_original_plate_and_two_primary_outputs(self):
        outputs = {
            name for name, p in self.node.parameters.items() if Modes.OUTPUT in getattr(p, "allowed_modes", set())
        }
        self.assertEqual(outputs, {"plate", "mask"})
        self.assertIs(self.node.parameter_output_values["plate"], self.plate)
        self.assertIsNone(self.node.parameter_output_values["mask"])
        self.assertEqual(self.node._widget_state()["canvasPlan"]["contentHeight"], 1080)
        state = self.node._widget_state()
        self.assertEqual(state["title"], "Selective Matte · 1920×1080")
        self.assertEqual([a["id"] for a in state["actions"]], ["load", "preview", "track", "render"])

    def test_custom_resize_updates_plan_keeps_clicks_and_cached_tracking(self):
        points = [{"frame": 0, "x": 0.6, "y": 0.4, "label": 1}]
        self.selection(points=points)
        cached = {"mask_sequence_dir": "already-tracked"}
        self.node._track_result = cached
        self.node.parameter_output_values["mask"] = Artifact("old.mp4")
        self.node.set_parameter_value("custom_resize", True)
        state = self.node._widget_state()
        self.assertEqual((state["outputWidth"], state["outputHeight"]), (960, 540))
        self.assertEqual(state["canvasPlan"]["height"], 544)
        self.assertEqual(state["points"], points)
        self.assertIs(self.node._track_result, cached)
        self.assertIsNone(self.node.parameter_output_values["mask"])
        self.node.set_parameter_value("resize_width", 1280)
        self.node.set_parameter_value("resize_height", 720)
        self.assertEqual(self.node._widget_state()["outputHeight"], 720)
        self.node.set_parameter_value("custom_resize", False)
        self.assertEqual(self.node._widget_state()["outputHeight"], 1080)

    def test_saved_action_never_replays_and_request_runs_once(self):
        with patch.object(Engine, "handle_request") as resolve:
            self.node.set_parameter_value("selection", {"action": "render", "requestId": "old"}, initial_setup=True)
            self.assertEqual(self.node._widget_state()["action"], "")
            resolve.assert_not_called()
            self.selection(action="preview", requestId="new")
            self.selection(action="preview", requestId="new")
            self.assertEqual(resolve.call_count, 1)

    def test_load_source_dispatches_without_points_or_model_import(self):
        with patch.object(Engine, "handle_request") as resolve:
            self.selection(action="load", requestId="load-source")
            resolve.assert_called_once()
        self.assertEqual(self.node.get_parameter_value("operation"), "load")
        # A backend import would fail in this context, proving Load needs no SAM or Alpha code.
        import builtins

        native_import = builtins.__import__

        def guard_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "selective_matte" and "backend" in fromlist:
                raise AssertionError("Load source must not import the model backend.")
            return native_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=guard_import):
            self.node._work()
        self.assertIs(self.node.parameter_output_values["plate"], self.plate)
        self.assertIsNone(self.node.parameter_output_values["mask"])
        self.assertEqual(self.node._widget_state()["points"], [])
        self.assertEqual(self.node._widget_state()["status"], "Click the subject to include it.")
        self.assertFalse(self.node._widget_state()["busy"])

    def test_native_start_flow_validation_failure_is_displayed(self):
        # Native ResolveNodeRequest also returns StartFlowResultFailure for dependency errors.
        failure = types.SimpleNamespace(succeeded=lambda: False, result_details="Missing upstream model")
        with patch.object(Engine, "handle_request", return_value=failure):
            self.selection(action="load", requestId="failed-load")
        self.assertEqual(self.node._widget_state()["status"], "Could not start operation.")
        self.assertEqual(self.node._widget_state()["error"], "Missing upstream model")

    def test_picks_and_settings_invalidate_final_only(self):
        self.node.parameter_output_values["mask"] = Artifact("old-alpha.mp4")
        self.selection(points=[{"frame": 0, "x": 0.2, "y": 0.3, "label": 1}], pointsRevision=4)
        self.assertIsNone(self.node.parameter_output_values["mask"])
        self.assertIs(self.node.parameter_output_values["plate"], self.plate)
        self.node.parameter_output_values["mask"] = Artifact("second-alpha.mp4")
        self.node.set_parameter_value("sam_checkpoint", "other.pt")
        self.assertIsNone(self.node.parameter_output_values["mask"])

    def test_ui_setting_invalidates_while_worker_is_publishing(self):
        entered = threading.Event()
        release = threading.Event()
        errors = []
        self.node._running = True
        self.node.parameter_output_values["mask"] = Artifact("old-alpha.mp4")
        previous_revision = self.node._result_revision

        def publish(name, value):
            if name == "selection" and threading.current_thread().name == "worker-publication":
                entered.set()
                if not release.wait(3):
                    raise TimeoutError("Regression fixture did not release worker publication.")

        def worker():
            try:
                self.node._push(status="Rendering alpha")
            except BaseException as exc:
                errors.append(exc)

        with patch.object(self.node, "publish_update_to_parameter", side_effect=publish):
            thread = threading.Thread(target=worker, name="worker-publication")
            thread.start()
            try:
                self.assertTrue(entered.wait(3))
                self.node.set_parameter_value("sam_checkpoint", "other.pt")
                self.assertTrue(self.node.is_cancellation_requested)
                self.assertIsNone(self.node.parameter_output_values["mask"])
                self.assertGreater(self.node._result_revision, previous_revision)
            finally:
                release.set()
                thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])

    def test_preview_has_no_mask_output_and_revisions_match(self):
        from selective_matte import backend

        preview = self.root / "mask_000000.png"
        preview.write_bytes(b"fake image served by mocked URL")
        self.selection(points=[{"frame": 0, "x": 0.2, "y": 0.3, "label": 1}], pointsRevision=7)
        result = {"preview_mask_path": str(preview), "mask_sequence_dir": str(self.root), "warnings": []}
        with patch.object(backend, "segment", return_value=result):
            self.node._work()
        state = self.node._widget_state()
        self.assertEqual(state["maskRevision"], 7)
        self.assertEqual(state["maskKind"], "selection")
        self.assertIsNone(self.node.parameter_output_values["mask"])

    def test_render_only_publishes_combined_alpha_after_completion(self):
        from selective_matte import alpha_render, backend

        self.selection(points=[{"frame": 0, "x": 0.2, "y": 0.3, "label": 1}], pointsRevision=3)
        self.node.set_parameter_value("operation", "render")
        self.node.set_parameter_value("custom_resize", True)
        preview = self.root / "mask_000000.png"
        preview.write_bytes(b"mask")
        result = {
            "preview_mask_path": str(preview),
            "mask_sequence_dir": str(self.root),
            "mask_video_path": "raw-sam.mp4",
            "warnings": [],
        }
        rendered = {
            "alpha_video_path": "combined-alpha.mp4",
            "plate_video_path": "resized-plate.mp4",
            "width": 1920,
            "height": 1080,
            "alpha_sequence_dir": "final-png",
            "preview_mask_path": "final-frame.png",
            "selected_frame": 0,
        }
        with (
            patch.object(backend, "segment", return_value=result),
            patch.object(backend, "unload_model") as unload,
            patch.object(alpha_render, "render_alpha", return_value=rendered) as render,
        ):
            self.node._work()
        unload.assert_called_once()
        self.assertEqual(self.node.parameter_output_values["mask"].value, "combined-alpha.mp4")
        self.assertEqual(self.node.parameter_output_values["plate"].value, "resized-plate.mp4")
        self.assertEqual(render.call_args.kwargs["output_size"], "processing")
        self.assertIsNone(render.call_args.kwargs["alpha_path"])
        self.assertEqual(render.call_args.kwargs["resize_width"], 960)
        self.assertEqual(render.call_args.kwargs["resize_height"], 540)
        self.node._refresh_source()
        self.assertEqual(self.node.parameter_output_values["plate"].value, "resized-plate.mp4")
        self.assertEqual(self.node._widget_state()["maskKind"], "final")
        callback = render.call_args.kwargs["progress"]
        callback(0.4, "test progress")
        self.assertEqual(self.node._widget_state()["progress"], 0.4)

    def test_alpha_change_and_disconnect_invalidate_mask(self):
        self.node.parameter_output_values["mask"] = Artifact("old-alpha.mp4")
        self.node._running = True
        self.node.set_parameter_value("alpha", Artifact("whole.mp4"))
        self.assertIsNone(self.node.parameter_output_values["mask"])
        self.assertTrue(self.node.is_cancellation_requested)
        self.node.parameter_output_values["mask"] = Artifact("second-alpha.mp4")
        self.node.after_incoming_connection_removed(None, None, self.node.parameters["alpha"])
        self.assertIsNone(self.node.get_parameter_value("alpha"))
        self.assertIsNone(self.node.parameter_output_values["mask"])

    def test_empty_alpha_renders_without_it(self):
        from selective_matte import alpha_render, backend

        self.selection(points=[{"frame": 0, "x": 0.2, "y": 0.3, "label": 1}])
        self.node.set_parameter_value("alpha", Artifact(""))
        self.node.set_parameter_value("operation", "render")
        rendered = {"alpha_video_path": "m.mp4", "plate_video_path": "p.mp4", "preview_mask_path": "f.png"}
        with (
            patch.object(backend, "segment", return_value={"mask_sequence_dir": str(self.root), "warnings": []}),
            patch.object(backend, "unload_model"),
            patch.object(alpha_render, "render_alpha", return_value=rendered) as render,
        ):
            self.node._work()
        self.assertIsNone(render.call_args.kwargs["alpha_path"])

    def test_render_passes_connected_alpha_video(self):
        from selective_matte import alpha_render, backend

        self.selection(points=[{"frame": 0, "x": 0.2, "y": 0.3, "label": 1}])
        whole = self.root / "whole-alpha.mp4"
        whole.write_bytes(b"alpha")
        self.node.set_parameter_value("alpha", Artifact(str(whole)))
        self.node.set_parameter_value("operation", "render")
        result = {"mask_sequence_dir": str(self.root), "warnings": []}
        rendered = {
            "alpha_video_path": "mask.mp4",
            "plate_video_path": "plate.mp4",
            "alpha_sequence_dir": "final-png",
            "preview_mask_path": "final.png",
        }
        with (
            patch.object(backend, "segment", return_value=result),
            patch.object(backend, "unload_model"),
            patch.object(alpha_render, "render_alpha", return_value=rendered) as render,
        ):
            self.node._work()
        self.assertEqual(render.call_args.kwargs["alpha_path"], str(whole.resolve()))
        self.assertEqual(self.node.parameter_output_values["mask"].value, "mask.mp4")

    def test_missing_final_video_fails_but_keeps_alpha_sequence(self):
        from selective_matte import alpha_render, backend

        self.selection(points=[{"frame": 0, "x": 0.2, "y": 0.3, "label": 1}], pointsRevision=3)
        self.node.set_parameter_value("operation", "render")
        preview = self.root / "mask_000000.png"
        preview.write_bytes(b"mask")
        result = {"preview_mask_path": str(preview), "mask_sequence_dir": str(self.root), "warnings": []}
        rendered = {
            "alpha_video_path": None,
            "alpha_sequence_dir": "final-png",
            "preview_mask_path": "final-frame.png",
            "selected_frame": 0,
            "warnings": ["Video encoder unavailable."],
        }
        with (
            patch.object(backend, "segment", return_value=result),
            patch.object(backend, "unload_model"),
            patch.object(alpha_render, "render_alpha", return_value=rendered),
        ):
            with self.assertRaisesRegex(RuntimeError, "Mask video could not be encoded"):
                self.node._work()
        self.assertIsNone(self.node.parameter_output_values["mask"])
        self.assertEqual(self.node.get_parameter_value("alpha_sequence"), "final-png")
        self.assertEqual(self.node.get_parameter_value("report"), rendered)
        self.assertEqual(self.node._widget_state()["status"], "Operation failed.")
        self.assertIn("Video encoder unavailable", self.node._widget_state()["error"])
        self.assertFalse(self.node._widget_state()["busy"])

    def test_source_change_clears_points_and_cancels_active_work(self):
        self.selection(points=[{"frame": 0, "x": 0.2, "y": 0.3, "label": 1}])
        self.node._running = True
        replacement = self.root / "replacement.mov"
        replacement.write_bytes(b"new")
        self.node.set_parameter_value("source", Artifact(str(replacement)))
        self.assertTrue(self.node.is_cancellation_requested)
        self.assertEqual(self.node._widget_state()["points"], [])
        self.assertIsNone(self.node.parameter_output_values["mask"])

    def test_missing_resized_plate_does_not_publish_mismatched_mask(self):
        from selective_matte import alpha_render, backend

        self.selection(points=[{"frame": 0, "x": 0.2, "y": 0.3, "label": 1}])
        self.node.set_parameter_value("operation", "render")
        result = {"mask_sequence_dir": str(self.root), "warnings": []}
        rendered = {
            "alpha_video_path": "mask.mp4",
            "plate_video_path": None,
            "alpha_sequence_dir": "final-png",
            "preview_mask_path": "final.png",
        }
        with (
            patch.object(backend, "segment", return_value=result),
            patch.object(backend, "unload_model"),
            patch.object(alpha_render, "render_alpha", return_value=rendered),
        ):
            with self.assertRaisesRegex(RuntimeError, "resized Plate could not be encoded"):
                self.node._work()
        self.assertIsNone(self.node.parameter_output_values["mask"])
        self.assertEqual(self.node.get_parameter_value("alpha_sequence"), "final-png")

    def test_disconnected_source_clears_plate_and_selection(self):
        self.selection(points=[{"frame": 0, "x": 0.2, "y": 0.3, "label": 1}])
        self.node.after_incoming_connection_removed(None, None, self.node.parameters["source"])
        self.assertIsNone(self.node.parameter_output_values["plate"])
        self.assertIsNone(self.node.parameter_output_values["mask"])
        self.assertEqual(self.node._widget_state()["points"], [])


if __name__ == "__main__":
    unittest.main()
