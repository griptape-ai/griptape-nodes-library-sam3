"""Native API contract tests. No SAM package, model download, CUDA or torch required."""

import contextlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

BACKEND_FILE = Path(__file__).resolve().parents[3] / "griptape_nodes_sam3_library" / "selective_matte" / "backend.py"
spec = importlib.util.spec_from_file_location("sam31_backend_tested", BACKEND_FILE)
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


class FakeNativePredictor:
    """Implements the inspected native request/response schema, not model behavior."""

    def __init__(self, width=5, height=3, frames=4):
        self.width, self.height, self.frames = width, height, frames
        self.requests = []
        self.prompts = {}
        self.frame_order = list(range(frames))
        self.was_shutdown = False

    def outputs(self, frame):
        ids = sorted({obj for (_, obj) in self.prompts})
        masks = np.zeros((len(ids), self.height, self.width), dtype=bool)
        for i, object_id in enumerate(ids):
            masks[i, :, (object_id - 1) % self.width] = True
        return {"out_obj_ids": np.array(ids), "out_binary_masks": masks}

    def handle_request(self, request):
        self.requests.append(request)
        kind = request["type"]
        if kind == "start_session":
            self.prompts.clear()
            return {"session_id": "native-session"}
        if kind == "reset_session":
            self.prompts.clear()
        if kind == "add_prompt":
            self.prompts[(request["frame_index"], request["obj_id"])] = request
            return {"frame_index": request["frame_index"], "outputs": self.outputs(request["frame_index"])}
        return {"is_success": True}

    def handle_stream_request(self, request):
        self.requests.append(request)
        for frame in self.frame_order:
            yield {"frame_index": frame, "outputs": self.outputs(frame)}

    def shutdown(self):
        self.was_shutdown = True


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.image = self.root / "source.png"
        Image.new("RGB", (5, 3), "red").save(self.image)
        self.video = self.root / "source.mp4"
        self.video.write_bytes(b"fake video; decoder is mocked")
        self.fake = FakeNativePredictor()
        self.factory_calls = []

        def factory(checkpoint, device):
            self.factory_calls.append((checkpoint, device))
            return self.fake

        self.adapter = backend.SAM31Backend(predictor_factory=factory)

    def tearDown(self):
        self.adapter.unload_model()
        self.temporary.cleanup()

    def video_patches(self):
        def extract(source, frame_index, destination, **kwargs):
            Image.new("RGB", (5, 3), "red").save(destination)
            return str(destination)

        def stage(source, destination, media, cancel, progress):
            destination.mkdir()
            return destination

        @contextlib.contextmanager
        def extraction_and_staging():
            with (
                patch.object(backend, "extract_frame", side_effect=extract),
                patch.object(backend, "_stage_video_frames", side_effect=stage),
            ):
                yield

        return patch.object(
            backend,
            "inspect_media",
            return_value={"width": 5, "height": 3, "fps": 23.976, "num_frames": 4, "is_image": False},
        ), extraction_and_staging()

    def test_widget_aliases_and_exclude_labels(self):
        points = backend.normalize_points(
            [
                {"id": "a", "frame": 2, "x": 0.2, "y": 0.8, "label": "include", "objectId": 7},
                {"frame_index": 2, "x": 0.4, "y": 0.6, "label": "exclude", "object_id": 7},
            ],
            4,
        )
        self.assertEqual([p["label"] for p in points], [1, 0])
        self.assertEqual(points[0]["object_id"], 7)
        self.assertEqual(points[0]["frame_index"], 2)

    def test_rejects_invalid_points_before_model_loading(self):
        for change in (
            {"x": float("nan")},
            {"y": 4},
            {"frame": -1},
            {"frame": 4},
            {"frame": 1.5},
            {"objectId": 0},
            {"label": 7},
        ):
            point = {"frame": 0, "x": 0.5, "y": 0.5, "label": 1, "objectId": 1} | change
            with self.subTest(change=change), self.assertRaises(ValueError):
                backend.normalize_points([point], 4)
        with self.assertRaisesRegex(ValueError, "include click"):
            backend.normalize_points([{"frame": 0, "x": 0.2, "y": 0.3, "label": 0}], 4)

    def test_preview_maps_selected_source_frame_to_still_frame_zero(self):
        points = [
            {"frame": 2, "x": 0.25, "y": 0.75, "label": 1, "objectId": 2},
            {"frame": 2, "x": 0.8, "y": 0.8, "label": 0, "objectId": 2},
        ]
        metadata, extraction = self.video_patches()
        with metadata, extraction:
            result = self.adapter.segment(self.video, points, selected_frame=2, output_dir=self.root)
        request = next(r for r in self.fake.requests if r["type"] == "add_prompt")
        self.assertEqual(request["frame_index"], 0)
        self.assertEqual(request["points"], [[0.25, 0.75], [0.8, 0.8]])
        self.assertEqual(request["point_labels"], [1, 0])
        self.assertEqual(request["obj_id"], 2)
        self.assertTrue(request["rel_coordinates"])
        self.assertEqual(result["frame_indices"], [2])
        self.assertEqual(Path(result["preview_mask_path"]).name, "mask_000002.png")
        self.assertEqual(result["num_frames"], 4)
        with Image.open(result["preview_mask_path"]) as mask:
            self.assertEqual(mask.size, (5, 3))
            self.assertEqual(set(np.asarray(mask).ravel()), {0, 255})

    def test_repeated_preview_reuses_model_and_resets_session(self):
        point = [{"frame": 0, "x": 0.2, "y": 0.3, "label": 1}]
        first = self.adapter.segment(self.image, point, output_dir=self.root)
        second = self.adapter.segment(self.image, point, output_dir=self.root)
        self.assertEqual(len(self.factory_calls), 1)
        kinds = [r["type"] for r in self.fake.requests]
        self.assertEqual(kinds.count("start_session"), 1)
        self.assertEqual(kinds.count("reset_session"), 1)
        self.assertNotEqual(first["mask_sequence_dir"], second["mask_sequence_dir"])

    def test_track_preserves_original_frame_numbers_bidirectionally(self):
        self.fake.frame_order = [2, 3, 2, 1, 0]
        points = [{"frame": 2, "x": 0.5, "y": 0.5, "label": 1}, {"frame": 3, "x": 0.1, "y": 0.1, "label": 0}]
        metadata, extraction = self.video_patches()
        with metadata, extraction:
            result = self.adapter.segment(
                self.video, points, mode="track", selected_frame=2, output_dir=self.root, make_preview_video=False
            )
        request = next(r for r in self.fake.requests if r["type"] == "propagate_in_video")
        self.assertEqual(request["propagation_direction"], "both")
        self.assertEqual(request["start_frame_index"], 2)
        self.assertEqual(result["frame_indices"], [0, 1, 2, 3])
        self.assertEqual(result["fps"], 23.976)
        manifest = json.loads(Path(result["manifest_path"]).read_text())
        self.assertTrue(manifest["complete"])
        self.assertEqual(manifest["frame_index_base"], 0)
        self.assertEqual(len(list(Path(result["mask_sequence_dir"]).glob("*.png"))), 4)
        session = next(r for r in self.fake.requests if r["type"] == "start_session")
        self.assertEqual(Path(session["resource_path"]).name, "source_frames")
        self.assertFalse(Path(session["resource_path"]).exists())

    def test_missing_frames_fail_instead_of_silently_shifting_sequence(self):
        self.fake.frame_order = [0, 1, 3]
        metadata, extraction = self.video_patches()
        with metadata, extraction, self.assertRaisesRegex(backend.SAM31Error, "omitted 1"):
            self.adapter.segment(
                self.video, [{"x": 0.5, "y": 0.5}], mode="track", output_dir=self.root, make_preview_video=False
            )
        self.assertEqual(list(self.root.glob("sam31_*")), [])
        self.assertIn("close_session", [r["type"] for r in self.fake.requests])

    def test_last_frame_seed_is_written_before_backward_only_tracking(self):
        self.fake.frame_order = [2, 1, 0]
        metadata, extraction = self.video_patches()
        with metadata, extraction:
            result = self.adapter.segment(
                self.video,
                [{"frame": 3, "x": 0.5, "y": 0.5}],
                mode="track",
                selected_frame=3,
                output_dir=self.root,
                make_preview_video=False,
            )
        request = next(r for r in self.fake.requests if r["type"] == "propagate_in_video")
        self.assertEqual(request["propagation_direction"], "backward")
        self.assertEqual(request["start_frame_index"], 3)
        self.assertEqual(result["frame_indices"], [0, 1, 2, 3])
        self.assertTrue(Path(result["preview_mask_path"]).is_file())

    def test_cancel_closes_native_session_and_removes_partial_output(self):
        cancel = threading.Event()

        def progress(fraction, message):
            if message.startswith("Tracked 1"):
                cancel.set()

        metadata, extraction = self.video_patches()
        with metadata, extraction, self.assertRaises(backend.SegmentationCancelled):
            self.adapter.segment(
                self.video,
                [{"x": 0.5, "y": 0.5}],
                mode="track",
                output_dir=self.root,
                cancel=cancel,
                progress=progress,
                make_preview_video=False,
            )
        kinds = [r["type"] for r in self.fake.requests]
        self.assertIn("cancel_propagation", kinds)
        self.assertIn("close_session", kinds)
        self.assertEqual(list(self.root.glob("sam31_*")), [])

    def test_masks_filter_objects_and_validate_shape(self):
        masks = np.ones((2, 3, 5), dtype=bool)
        masks[0] = False
        outputs = {"out_obj_ids": np.array([1, 99]), "out_binary_masks": masks}
        self.assertFalse(backend._union_mask(outputs, [1], 5, 3).any())
        with self.assertRaisesRegex(backend.SAM31Error, "dimensions"):
            backend._union_mask(outputs, [1], 3, 5)
        empty = {"out_obj_ids": np.array([]), "out_binary_masks": np.zeros((0, 3, 5), bool)}
        self.assertFalse(backend._union_mask(empty, [1], 5, 3).any())

    def test_native_false_offload_compatibility_only(self):
        class Model:
            def init_state(self, resource_path, offload_video_to_cpu=False):
                return resource_path, offload_video_to_cpu

        class Predictor:
            model = Model()

        native = Predictor()
        backend._adapt_native_init_state(native)
        self.assertEqual(
            native.model.init_state(resource_path="video.mp4", offload_video_to_cpu=True, offload_state_to_cpu=False),
            ("video.mp4", True),
        )
        with self.assertRaisesRegex(backend.SAM31Error, "state offloading"):
            native.model.init_state(resource_path="video.mp4", offload_state_to_cpu=True)

    def test_first_click_keeps_native_refined_mask_without_prior_cache(self):
        class Model:
            def _build_sam2_output(self, inference_state, frame_idx, refined_obj_id_to_mask=None):
                if frame_idx not in inference_state["cached_frame_outputs"]:
                    return {}
                result = inference_state["cached_frame_outputs"][frame_idx].copy()
                if refined_obj_id_to_mask:
                    result.update(refined_obj_id_to_mask)
                return result

        class Predictor:
            model = Model()

        predictor = Predictor()
        state = {"cached_frame_outputs": {}}
        mask = np.ones((1, 3, 5), dtype=bool)
        self.assertEqual(predictor.model._build_sam2_output(state, 0, {1: mask}), {})
        backend._adapt_native_first_point_output(predictor)
        self.assertIs(predictor.model._build_sam2_output(state, 0, {1: mask})[1], mask)
        self.assertEqual(predictor.model._build_sam2_output(state, 1), {})
        self.assertNotIn(1, state["cached_frame_outputs"])
        state["cached_frame_outputs"][0][2] = "previous mask"
        self.assertEqual(predictor.model._build_sam2_output(state, 0, {1: mask})[2], "previous mask")

    def test_native_autocast_contexts_unwind_in_reverse_order(self):
        from types import SimpleNamespace

        order = []

        class Context:
            def __init__(self, name):
                self.name = name

            def __exit__(self, *args):
                order.append(self.name)

        native = SimpleNamespace(
            bf16_context=Context("predictor"),
            model=SimpleNamespace(tracker=SimpleNamespace(bf16_context=Context("tracker"))),
        )
        backend._release_native_autocast(native)
        self.assertEqual(order, ["predictor", "tracker"])

    def test_native_decoder_flash_only_request_allows_available_torch_kernels(self):
        from types import SimpleNamespace

        enum = SimpleNamespace(FLASH_ATTENTION="flash", EFFICIENT_ATTENTION="efficient", MATH="math")
        calls = []

        def native_context(backends, *args, **kwargs):
            calls.append((backends, args, kwargs))
            return contextlib.nullcontext()

        decoder = SimpleNamespace(sdpa_kernel=native_context)
        backend._allow_native_sdpa_fallback(decoder, enum)
        wrapped = decoder.sdpa_kernel
        with decoder.sdpa_kernel(enum.FLASH_ATTENTION, set_priority=True):
            pass
        self.assertEqual(calls[-1], (["flash", "efficient", "math"], (), {"set_priority": True}))
        with decoder.sdpa_kernel([enum.MATH]):
            pass
        self.assertEqual(calls[-1][0], ["math"])
        with decoder.sdpa_kernel([enum.FLASH_ATTENTION, enum.MATH]):
            pass
        self.assertEqual(calls[-1][0], ["flash", "math"])
        backend._allow_native_sdpa_fallback(decoder, enum)
        self.assertIs(decoder.sdpa_kernel, wrapped)

    def fake_decoder(self, frames):
        class Capture:
            released = False

            def __init__(self):
                self.frames = iter(frames)

            def isOpened(self):
                return True

            def read(self):
                value = next(self.frames, None)
                return value is not None, value

            def release(self):
                self.released = True

        capture = Capture()
        from types import SimpleNamespace

        cv2 = SimpleNamespace(
            VideoCapture=lambda source: capture, COLOR_BGR2RGB=1, cvtColor=lambda value, code: value[..., ::-1]
        )
        return capture, patch.object(backend, "_media_modules", return_value=(cv2, np, Image))

    def test_lossless_staging_keeps_original_frame_order_and_rgb_pixels(self):
        frames = [np.full((3, 5, 3), (10 + index, 20, 200), dtype=np.uint8) for index in range(3)]
        capture, decoder = self.fake_decoder(frames)
        destination = self.root / "staged"
        with decoder:
            backend._stage_video_frames(self.video, destination, {"width": 5, "height": 3, "num_frames": 3})
        self.assertTrue(capture.released)
        self.assertEqual([p.name for p in sorted(destination.iterdir())], [f"{i:06d}.png" for i in range(3)])
        for index in range(3):
            with Image.open(destination / f"{index:06d}.png") as image:
                np.testing.assert_array_equal(np.asarray(image), frames[index][..., ::-1])

    def test_lossless_staging_cancel_and_bad_source_remove_partial_files(self):
        metadata = {"width": 5, "height": 3, "num_frames": 3}
        good = np.zeros((3, 5, 3), dtype=np.uint8)
        for name, frames, cancel_after_one in (
            ("short", [good], False),
            ("shape", [good, np.zeros((4, 5, 3), dtype=np.uint8)], False),
            ("cancel", [good] * 3, True),
        ):
            with self.subTest(name=name):
                capture, decoder = self.fake_decoder(frames)
                destination = self.root / name
                cancel = threading.Event()

                def progress(fraction, message):
                    if cancel_after_one:  # noqa: B023
                        cancel.set()  # noqa: B023

                with decoder, self.assertRaises(backend.SAM31Error):
                    backend._stage_video_frames(self.video, destination, metadata, cancel, progress)
                self.assertTrue(capture.released)
                self.assertFalse(destination.exists())

    def test_cpu_request_is_rejected_without_importing_torch(self):
        with self.assertRaisesRegex(backend.SAM31Error, "requires CUDA"):
            backend._build_predictor(device="cpu")

    def test_missing_native_dependencies_are_actionable_without_loading_models(self):
        with patch("importlib.util.find_spec", return_value=None):
            info = backend.runtime_info()
            self.assertFalse(info["cuda_checked"])
            self.assertFalse(info["model_loaded"])
            self.assertIn("triton", info["missing_modules"])
            with self.assertRaisesRegex(backend.SAM31Error, "Triton unconditionally"):
                backend._check_native_dependencies()

    def test_worker_protocol_handles_multiple_requests_without_models(self):
        invalid = json.dumps({"source_path": "unused", "points": [], "mode": "invalid"})
        result = subprocess.run(
            [sys.executable, str(BACKEND_FILE), "--worker"],
            input=invalid + "\n" + invalid + "\n",
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        messages = [
            json.loads(line[len("__SAM31__") :]) for line in result.stdout.splitlines() if line.startswith("__SAM31__")
        ]
        self.assertEqual(len(messages), 2)
        self.assertTrue(all(m["type"] == "error" for m in messages))
        self.assertIn("mode must", messages[0]["message"])

    def test_browser_proxy_retains_exact_odd_and_even_dimensions(self):
        try:
            import av
        except ImportError:
            self.skipTest("PyAV not installed; run with the existing media runtime on sys.path")
        for width, height, extension, codec in ((15, 9, ".webm", "vp9"), (16, 10, ".mp4", "h264")):
            with self.subTest(width=width, height=height):
                sequence = self.root / f"frames_{width}"
                sequence.mkdir()
                for index in range(4):
                    Image.new("L", (width, height), 30 + index * 50).save(sequence / f"mask_{index:06d}.png")
                result, warning = backend._write_preview_video(
                    sequence,
                    self.root / f"proxy_{width}.mp4",
                    {"width": width, "height": height, "fps": 24, "num_frames": 4},
                    None,
                )
                self.assertIsNone(warning)
                self.assertEqual(Path(result).suffix, extension)
                with av.open(result) as container:
                    stream = container.streams.video[0]
                    self.assertEqual(stream.codec_context.name, codec)
                    frames = list(container.decode(stream))
                self.assertEqual(len(frames), 4)
                self.assertEqual([(frame.width, frame.height) for frame in frames], [(width, height)] * 4)
                self.assertTrue(all(abs(float(frame.time) - index / 24) <= 0.001 for index, frame in enumerate(frames)))
                self.assertEqual(
                    [round(float(frame.to_ndarray(format="gray").mean()) / 10) * 10 for frame in frames],
                    [30, 80, 130, 180],
                )

    def test_proxy_rejects_dimension_mismatch_without_output(self):
        try:
            import av  # noqa: F401
        except ImportError:
            self.skipTest("PyAV not installed")
        sequence = self.root / "bad_proxy_frames"
        sequence.mkdir()
        Image.new("L", (14, 9), 100).save(sequence / "mask_000000.png")
        result, warning = backend._write_preview_video(
            sequence, self.root / "bad.mp4", {"width": 15, "height": 9, "fps": 24, "num_frames": 1}, None
        )
        self.assertIsNone(result)
        self.assertIn("Exact-size preview unavailable", warning)
        self.assertFalse((self.root / "bad.webm").exists())


if __name__ == "__main__":
    unittest.main()
