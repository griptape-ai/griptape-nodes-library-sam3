"""CPU selective-alpha integration tests with injected model runners.

No diffusion pipeline, matting weights, or GPU inference is loaded. OpenCV is
used for real blending geometry; native media decoding is tested separately.
"""

import sys
import tempfile
import threading
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "griptape_nodes_sam3_library"))
from selective_matte import alpha_render, backend, preprocessing


class FakeRefiner:
    def __init__(self, result=None):
        self.result = result
        self.closed = False

    def __call__(self, rgb, selection):
        return selection.copy() if self.result is None else self.result

    def close(self):
        self.closed = True


class AlphaRenderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source.mp4"
        self.source.write_bytes(b"CPU test input; frames supplied by fake decoder")
        self.masks = self.root / "selection"
        self.masks.mkdir()
        self.destination = self.root / "output"
        self.destination.mkdir()
        self.width, self.height, self.num_frames = 15, 9, 13
        self.frames = [np.full((self.height, self.width, 3), 90 + i, np.uint8) for i in range(self.num_frames)]
        for frame in range(self.num_frames):
            Image.fromarray(np.full((self.height, self.width), 255, np.uint8)).save(
                self.masks / f"mask_{frame:06d}.png"
            )
        self.model_calls = []
        self.refiner = FakeRefiner()

    def tearDown(self):
        self.temp.cleanup()

    def runner(self, pipeline, frames, **kwargs):
        self.model_calls.append([np.asarray(frame).copy() for frame in frames])
        # The canvas's center is inside the fitted original; its source color
        # identifies the original frame, including repeated end padding.
        return [
            np.full((32, 32), 0.4 + 0.02 * (int(np.asarray(frame)[16, 16, 0]) - 90), np.float32) for frame in frames
        ]

    def render(self, **overrides):
        media = {
            "width": self.width,
            "height": self.height,
            "fps": 29.97,
            "num_frames": self.num_frames,
            "kind": "video",
        }
        args = {
            "pipeline": object(),
            "source_path": self.source,
            "mask_sequence_dir": self.masks,
            "output_dir": self.destination,
            "canvas_width": 32,
            "canvas_height": 32,
            "chunk_size": 9,
            "overlap": 1,
            "_runner": self.runner,
            "_refiner": self.refiner,
        }
        args["output_size"] = "original"  # Explicitly exercise legacy restoration separately.
        args.update(overrides)
        with ExitStack() as stack:
            stack.enter_context(patch.object(preprocessing, "probe_media", return_value=media))
            stack.enter_context(patch.object(alpha_render, "_frames", side_effect=lambda _: iter(self.frames)))
            stack.enter_context(patch.object(backend, "unload_model"))
            stack.enter_context(
                patch.object(backend, "_write_preview_video", return_value=(None, "Preview skipped by CPU test"))
            )
            return alpha_render.render_alpha(**args)

    def assert_no_published_run(self):
        self.assertEqual(list(self.destination.iterdir()), [], "Failed render left partial output artifacts.")

    def test_canvas_padding_chunk_overlap_and_repeat_are_restored(self):
        result = self.render(selected_frame=7)
        self.assertTrue(self.refiner.closed)
        self.assertEqual(len(self.model_calls), 2)
        self.assertEqual([len(call) for call in self.model_calls], [9, 9])
        # 13 originals => 9-frame call + 5 real overlapping frames + 4 repeats.
        second = self.model_calls[1]
        self.assertEqual([int(frame[16, 16, 0]) for frame in second], [98, 99, 100, 101, 102, 102, 102, 102, 102])
        first = self.model_calls[0][0]
        self.assertTrue(np.all(first[0] == 0))
        self.assertTrue(np.all(first[16] == 90))
        sequence = Path(result["alpha_sequence_dir"])
        files = sorted(sequence.glob("alpha_*.png"))
        self.assertEqual(len(files), 13)
        self.assertEqual(result["num_frames"], 13)
        self.assertEqual(result["fps"], 29.97)
        self.assertEqual(Path(result["preview_mask_path"]).name, "alpha_000007.png")
        for frame, file in enumerate(files):
            with Image.open(file) as image:
                alpha = np.asarray(image).astype(np.float32) / 65535
                self.assertEqual(image.size, (self.width, self.height))
            np.testing.assert_allclose(alpha, 0.4 + 0.02 * frame, atol=2 / 65535)
        self.assertFalse(list(self.destination.rglob("*.f32")))

    def test_soft_alpha_is_not_quantized_to_eight_bits(self):
        target = 0.432123

        def runner(pipeline, frames, **kwargs):
            return [np.full((32, 32), target, np.float32) for _ in frames]

        result = self.render(_runner=runner)
        with Image.open(result["preview_mask_path"]) as image:
            sample = int(np.asarray(image)[4, 7])
        self.assertLess(abs(sample / 65535 - target), 1 / 65535)
        self.assertNotEqual(sample % 257, 0, "Output was first rounded to eight bits.")

    def test_processing_outputs_crop_padding_without_resizing_back(self):
        result = self.render(output_size="processing")
        self.assertEqual((result["width"], result["height"]), (32, 19))
        self.assertEqual((result["source_width"], result["source_height"]), (15, 9))
        self.assertEqual(result["output_size"], "processing")
        with Image.open(result["preview_mask_path"]) as image:
            self.assertEqual(image.size, (32, 19))

    def test_custom_resize_changes_actual_model_input_and_both_video_exports(self):
        self.width, self.height = 128, 72
        self.frames = [np.full((72, 128, 3), 90 + i, np.uint8) for i in range(self.num_frames)]
        for frame in range(self.num_frames):
            Image.fromarray(np.full((72, 128), 255, np.uint8)).save(self.masks / f"mask_{frame:06d}.png")
        media = {"width": 128, "height": 72, "fps": 24.0, "num_frames": self.num_frames, "kind": "video"}
        with (
            patch.object(preprocessing, "probe_media", return_value=media),
            patch.object(alpha_render, "_frames", side_effect=lambda _: iter(self.frames)),
            patch.object(backend, "unload_model"),
            patch.object(backend, "_write_preview_video", return_value=("video.webm", None)) as encode,
        ):
            result = alpha_render.render_alpha(
                object(),
                self.source,
                self.masks,
                self.destination,
                canvas_width=1920,
                canvas_height=1088,
                resize_width=32,
                resize_height=32,
                chunk_size=9,
                overlap=1,
                _runner=self.runner,
                _refiner=self.refiner,
            )
        self.assertEqual(self.model_calls[0][0].shape, (32, 32, 3))
        self.assertEqual((result["width"], result["height"]), (32, 18))
        self.assertEqual(len(encode.call_args_list), 2)
        for call in encode.call_args_list:
            self.assertEqual((call.args[2]["width"], call.args[2]["height"]), (32, 18))
        with Image.open(result["preview_mask_path"]) as image:
            self.assertEqual(image.size, (32, 18))

    def test_processing_plate_and_mask_encode_same_geometry_and_frame_order(self):
        try:
            import av
        except ImportError:
            self.skipTest("PyAV is required for the actual encoder check")
        media = {
            "width": self.width,
            "height": self.height,
            "fps": 24.0,
            "num_frames": self.num_frames,
            "kind": "video",
        }
        with (
            patch.object(preprocessing, "probe_media", return_value=media),
            patch.object(alpha_render, "_frames", side_effect=lambda _: iter(self.frames)),
            patch.object(backend, "unload_model"),
        ):
            result = alpha_render.render_alpha(
                object(),
                self.source,
                self.masks,
                self.destination,
                canvas_width=32,
                canvas_height=32,
                chunk_size=9,
                overlap=1,
                _runner=self.runner,
                _refiner=self.refiner,
            )
        self.assertEqual(result["warnings"], [])
        for key in ("plate_video_path", "alpha_video_path"):
            self.assertTrue(result[key].endswith(".webm"))  # Exact 32x19, no codec padding.
            with av.open(result[key]) as video:
                frames = list(video.decode(video=0))
                self.assertEqual(len(frames), self.num_frames)
                self.assertTrue(all((f.width, f.height) == (32, 19) for f in frames))
                self.assertAlmostEqual(float(frames[-1].time), 12 / 24, places=4)
                if key == "plate_video_path":
                    # No black padding remains; source progression is preserved.
                    first, last = [f.to_ndarray(format="rgb24") for f in (frames[0], frames[-1])]
                    self.assertGreater(first[0].mean(), 80)
                    self.assertGreater(last.mean(), first.mean() + 8)

    def test_letterbox_bars_are_removed_from_final_alpha(self):
        def runner(pipeline, frames, **kwargs):
            alpha = np.zeros((32, 32), np.float32)
            # 15x9 fitted to 32x32 is 32x19, with six rows above/seven below.
            alpha[6:25] = 0.6
            return [alpha.copy() for _ in frames]

        result = self.render(_runner=runner)
        with Image.open(result["preview_mask_path"]) as image:
            restored = np.asarray(image).astype(np.float32) / 65535
        np.testing.assert_allclose(restored, 0.6, atol=1 / 65535)

    def test_progress_matches_backend_fraction_message_contract(self):
        seen = []

        def progress(fraction, message):
            self.assertIsInstance(fraction, (int, float))
            self.assertIsInstance(message, str)
            self.assertGreaterEqual(fraction, 0)
            self.assertLessEqual(fraction, 1)
            seen.append(fraction)

        self.render(progress=progress)
        self.assertEqual(seen[-1], 1)

    def test_missing_mask_fails_before_model_or_outputs(self):
        (self.masks / "mask_000004.png").unlink()
        with self.assertRaisesRegex(ValueError, "frame 4.*missing"):
            self.render()
        self.assertFalse(self.model_calls)
        self.assert_no_published_run()

    def test_source_mask_shape_mismatch_fails_without_outputs(self):
        Image.fromarray(np.ones((8, 15), np.uint8)).save(self.masks / "mask_000000.png")
        with self.assertRaisesRegex(ValueError, "dimensions do not match"):
            self.render()
        self.assertFalse(self.model_calls)
        self.assert_no_published_run()

    def test_refined_shape_is_checked_before_ltx(self):
        wrong = FakeRefiner(np.full((1, 1), 0.5, np.float32))
        with self.assertRaisesRegex(ValueError, "[Ss]hape|dimensions|[Rr]efined|[Rr]efiner|misaligned"):
            self.render(_refiner=wrong)
        self.assertTrue(wrong.closed)
        self.assertFalse(self.model_calls)
        self.assert_no_published_run()

    def test_nonfinite_refined_alpha_is_rejected(self):
        wrong = FakeRefiner(np.full((32, 32), np.nan, np.float32))
        with self.assertRaisesRegex(ValueError, "finite|invalid|[Rr]efined|[Rr]efiner"):
            self.render(_refiner=wrong)
        self.assertFalse(self.model_calls)
        self.assert_no_published_run()

    def test_runner_failure_removes_prepared_and_partial_outputs(self):
        def fail(pipeline, frames, **kwargs):
            raise RuntimeError("Synthetic inference failure")

        with self.assertRaisesRegex(RuntimeError, "Synthetic inference failure"):
            self.render(_runner=fail)
        self.assertTrue(self.refiner.closed)
        self.assert_no_published_run()

    def test_refiner_initialization_failure_leaves_no_run(self):
        with patch.object(alpha_render, "ViTMatteRefiner", side_effect=RuntimeError("Synthetic refiner failure")):
            with self.assertRaisesRegex(RuntimeError, "Synthetic refiner failure"):
                self.render(_refiner=None)
        self.assert_no_published_run()

    def test_cancel_event_is_supported_and_leaves_no_run(self):
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaisesRegex((RuntimeError, InterruptedError), "cancelled"):
            self.render(cancel=cancelled)
        self.assert_no_published_run()

    def test_incomplete_decoded_chunk_is_rejected_without_outputs(self):
        def short(pipeline, frames, **kwargs):
            return [np.full((32, 32), 0.5, np.float32)]

        with self.assertRaisesRegex(ValueError, "fewer frames"):
            self.render(_runner=short)
        self.assert_no_published_run()

    def test_bad_ltx_shape_is_rejected_without_outputs(self):
        def wrong(pipeline, frames, **kwargs):
            return [np.full((31, 32), 0.5, np.float32) for _ in frames]

        with self.assertRaisesRegex(ValueError, "misaligned"):
            self.render(_runner=wrong)
        self.assert_no_published_run()

    def test_touching_opaque_other_subject_is_excluded(self):
        selection = np.zeros((96, 128), np.float32)
        selection[16:80, 16:48] = 1
        full = selection.copy()
        full[16:80, 48:96] = 1  # another subject touches selected subject's right edge
        alpha = alpha_render.blend_selected(full, selection, selection)
        self.assertGreater(alpha[48, 40], 0.95)
        self.assertLess(alpha[48, 50], 0.05, "Opaque neighboring subject leaked through the selected-subject matte.")

    def test_soft_detail_outside_selection_is_preserved_away_from_others(self):
        selection = np.zeros((96, 128), np.float32)
        selection[16:80, 16:48] = 1
        full = selection.copy()
        full[30:60, 48:52] = 0.4  # fine soft detail beyond the semantic selection
        alpha = alpha_render.blend_selected(full, selection, selection)
        self.assertAlmostEqual(float(alpha[40, 50]), 0.4, places=5)


if __name__ == "__main__":
    unittest.main()
