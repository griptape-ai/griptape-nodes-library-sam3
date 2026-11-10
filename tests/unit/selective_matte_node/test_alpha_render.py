"""CPU selective-alpha integration tests with an injected refiner.

No matting weights or GPU inference are loaded. OpenCV is used for real
blending geometry; native media decoding is tested separately.
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
        self.calls = []

    def __call__(self, rgb, selection):
        self.calls.append(rgb.copy())
        return selection.copy() if self.result is None else self.result

    def close(self):
        self.closed = True


class AlphaRenderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source.mp4"
        self.source.write_bytes(b"CPU test input; frames supplied by fake decoder")
        self.alpha = self.root / "alpha.mp4"
        self.alpha.write_bytes(b"CPU test whole-frame alpha")
        self.masks = self.root / "selection"
        self.masks.mkdir()
        self.destination = self.root / "output"
        self.destination.mkdir()
        self.width, self.height, self.num_frames = 15, 9, 13
        self.frames = [np.full((self.height, self.width, 3), 90 + i, np.uint8) for i in range(self.num_frames)]
        self.alpha_frames = [np.full((self.height, self.width, 3), 100 + i, np.uint8) for i in range(self.num_frames)]
        self.alpha_count = self.num_frames
        for frame in range(self.num_frames):
            Image.fromarray(np.full((self.height, self.width), 255, np.uint8)).save(
                self.masks / f"mask_{frame:06d}.png"
            )
        self.refiner = FakeRefiner()

    def tearDown(self):
        self.temp.cleanup()

    def media(self, path):
        count = self.alpha_count if Path(path) == self.alpha else self.num_frames
        return {"width": self.width, "height": self.height, "fps": 29.97, "num_frames": count, "kind": "video"}

    def decode(self, path):
        yield from self.alpha_frames if Path(path) == self.alpha else self.frames

    def render(self, encode=("video.webm", None), **overrides):
        args = {
            "source_path": self.source,
            "mask_sequence_dir": self.masks,
            "output_dir": self.destination,
            "canvas_width": 32,
            "canvas_height": 32,
            "_refiner": self.refiner,
            "output_size": "original",  # Explicitly exercise legacy restoration separately.
        }
        args.update(overrides)
        with ExitStack() as stack:
            stack.enter_context(patch.object(preprocessing, "probe_media", side_effect=self.media))
            stack.enter_context(patch.object(alpha_render, "_frames", side_effect=self.decode))
            stack.enter_context(patch.object(backend, "unload_model"))
            if encode is not None:
                self.encode = stack.enter_context(patch.object(backend, "_write_preview_video", return_value=encode))
            return alpha_render.render_alpha(**args)

    def read_alpha(self, path):
        with Image.open(path) as image:
            return np.asarray(image).astype(np.float32) / 65535

    def assert_no_published_run(self):
        self.assertEqual(list(self.destination.iterdir()), [], "Failed render left partial output artifacts.")

    def test_without_alpha_input_mask_is_refined_selection(self):
        result = self.render(selected_frame=7)
        self.assertTrue(self.refiner.closed)
        self.assertEqual(len(self.refiner.calls), 13)
        # The refiner sees the padded processing canvas.
        self.assertTrue(np.all(self.refiner.calls[0][0] == 0))
        self.assertTrue(np.all(self.refiner.calls[0][16] == 90))
        files = sorted(Path(result["alpha_sequence_dir"]).glob("alpha_*.png"))
        self.assertEqual(len(files), 13)
        self.assertEqual(result["num_frames"], 13)
        self.assertEqual(result["fps"], 29.97)
        self.assertEqual(Path(result["preview_mask_path"]).name, "alpha_000007.png")
        for file in files:
            alpha = self.read_alpha(file)
            self.assertEqual(alpha.shape, (self.height, self.width))
            np.testing.assert_allclose(alpha, 1, atol=2 / 65535)

    def test_alpha_input_supplies_per_frame_detail(self):
        result = self.render(alpha_path=self.alpha)
        for frame, file in enumerate(sorted(Path(result["alpha_sequence_dir"]).glob("alpha_*.png"))):
            np.testing.assert_allclose(self.read_alpha(file), (100 + frame) / 255, atol=2 / 255)

    def test_alpha_input_at_another_resolution_is_resized_to_source(self):
        self.alpha_frames = [np.full((18, 30, 3), 120, np.uint8) for _ in range(self.num_frames)]
        result = self.render(alpha_path=self.alpha)
        np.testing.assert_allclose(self.read_alpha(result["preview_mask_path"]), 120 / 255, atol=2 / 255)

    def test_alpha_input_with_other_framing_is_refused(self):
        self.alpha_frames = [np.full((32, 32, 3), 120, np.uint8) for _ in range(self.num_frames)]
        with self.assertRaisesRegex(ValueError, "framing"):
            self.render(alpha_path=self.alpha)
        self.assert_no_published_run()

    def test_soft_alpha_is_not_quantized_to_eight_bits(self):
        target = 0.432123
        refiner = FakeRefiner(np.full((32, 32), target, np.float32))
        result = self.render(_refiner=refiner)
        sample = int(np.asarray(Image.open(result["preview_mask_path"]))[4, 7])
        self.assertLess(abs(sample / 65535 - target), 1 / 65535)
        self.assertNotEqual(sample % 257, 0, "Output was first rounded to eight bits.")

    def test_alpha_frame_count_mismatch_fails_before_outputs(self):
        self.alpha_count = 12
        with self.assertRaisesRegex(ValueError, "12 frames.*13"):
            self.render(alpha_path=self.alpha)
        self.assertFalse(self.refiner.calls)
        self.assert_no_published_run()

    def test_short_alpha_decode_fails_without_outputs(self):
        self.alpha_frames = self.alpha_frames[:5]
        with self.assertRaisesRegex(ValueError, "ended before the Source"):
            self.render(alpha_path=self.alpha)
        self.assertTrue(self.refiner.closed)
        self.assert_no_published_run()

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
        result = self.render(
            canvas_width=1920, canvas_height=1088, resize_width=32, resize_height=32, output_size="processing"
        )
        self.assertEqual(self.refiner.calls[0].shape, (32, 32, 3))
        self.assertEqual((result["width"], result["height"]), (32, 18))
        self.assertEqual(len(self.encode.call_args_list), 2)
        for call in self.encode.call_args_list:
            self.assertEqual((call.args[2]["width"], call.args[2]["height"]), (32, 18))
        with Image.open(result["preview_mask_path"]) as image:
            self.assertEqual(image.size, (32, 18))

    def test_processing_plate_and_mask_encode_same_geometry_and_frame_order(self):
        try:
            import av
        except ImportError:
            self.skipTest("PyAV is required for the actual encoder check")
        result = self.render(encode=None, output_size="processing")
        self.assertEqual(result["warnings"], [])
        for key in ("plate_video_path", "alpha_video_path"):
            self.assertTrue(result[key].endswith(".webm"))  # Exact 32x19, no codec padding.
            with av.open(result[key]) as video:
                frames = list(video.decode(video=0))
                self.assertEqual(len(frames), self.num_frames)
                self.assertTrue(all((f.width, f.height) == (32, 19) for f in frames))
                if key == "plate_video_path":
                    # No black padding remains; source progression is preserved.
                    first, last = [f.to_ndarray(format="rgb24") for f in (frames[0], frames[-1])]
                    self.assertGreater(first[0].mean(), 80)
                    self.assertGreater(last.mean(), first.mean() + 8)

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
        self.assertFalse(self.refiner.calls)
        self.assert_no_published_run()

    def test_source_mask_shape_mismatch_fails_without_outputs(self):
        Image.fromarray(np.ones((8, 15), np.uint8)).save(self.masks / "mask_000000.png")
        with self.assertRaisesRegex(ValueError, "dimensions do not match"):
            self.render()
        self.assert_no_published_run()

    def test_refined_shape_is_checked(self):
        wrong = FakeRefiner(np.full((1, 1), 0.5, np.float32))
        with self.assertRaisesRegex(ValueError, "misaligned"):
            self.render(_refiner=wrong)
        self.assertTrue(wrong.closed)
        self.assert_no_published_run()

    def test_nonfinite_refined_alpha_is_rejected(self):
        wrong = FakeRefiner(np.full((32, 32), np.nan, np.float32))
        with self.assertRaisesRegex(ValueError, "invalid"):
            self.render(_refiner=wrong)
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
