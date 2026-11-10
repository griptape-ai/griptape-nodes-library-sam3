import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "griptape_nodes_sam3_library"))
from selective_matte.preprocessing import (
    pad_mask,
    pad_rgb,
    plan_canvas,
    plan_custom_resize,
    restore_mask,
)


class CanvasTests(unittest.TestCase):
    def test_full_hd_is_padded_not_stretched(self):
        p = plan_canvas(1920, 1080)
        self.assertEqual((p.content_width, p.content_height, p.pad_top, p.pad_bottom), (1920, 1080, 4, 4))
        a = np.full((1080, 1920), 0.321, np.float32)
        padded = pad_mask(a, p)
        self.assertTrue((padded[:4] == 0).all())
        np.testing.assert_array_equal(restore_mask(padded, p), a)

    def test_portrait_keeps_entire_frame_and_alignment(self):
        p = plan_canvas(1080, 1920)
        self.assertEqual((p.content_width, p.content_height), (612, 1088))
        self.assertEqual(p.pad_left, 654)
        mask = np.ones((1920, 1080), np.uint8)
        rgb = np.repeat((mask * 127)[..., None], 3, -1)
        np.testing.assert_array_equal(pad_rgb(rgb, p)[..., 0] > 0, pad_mask(mask, p) > 0)
        self.assertTrue((restore_mask(pad_mask(mask, p), p) == 1).all())

    def test_odd_padding_and_precision(self):
        p = plan_canvas(101, 99, 64, 64)
        self.assertEqual(p.pad_top + p.content_height + p.pad_bottom, 64)
        a = np.full((99, 101), 0.123456, np.float32)
        np.testing.assert_allclose(restore_mask(pad_mask(a, p), p), a, atol=1e-6)

    def test_invalid_canvas(self):
        for dims in [(0, 10), (10, 10, 1920, 1080), (10, 10, 3840, 2176)]:
            with self.assertRaises(ValueError):
                plan_canvas(*dims)

    def test_custom_resize_uses_requested_content_before_model_padding(self):
        p = plan_custom_resize(3840, 2160, 960, 540)
        self.assertEqual((p.content_width, p.content_height), (960, 540))
        self.assertEqual((p.width, p.height, p.pad_top, p.pad_bottom), (960, 544, 2, 2))
        # Rounding the model canvas must not enlarge the requested content.
        p = plan_custom_resize(3840, 2160, 1000, 1000)
        self.assertEqual((p.content_width, p.content_height), (1000, 563))
        self.assertEqual((p.width, p.height), (1024, 576))

    def test_custom_resize_preserves_portrait_and_does_not_enlarge_small_source(self):
        p = plan_custom_resize(1080, 1920, 960, 540)
        self.assertEqual((p.content_width, p.content_height), (304, 540))
        p = plan_custom_resize(640, 360, 1280, 720)
        self.assertEqual((p.content_width, p.content_height), (640, 360))

    def test_custom_resize_rejects_invalid_bounds(self):
        for w, h in [(0, 540), (960, 0), (1921, 1080), (960, 1089), (960.5, 540), (True, 540), (960, None)]:
            with self.assertRaises(ValueError):
                plan_custom_resize(3840, 2160, w, h)

    def test_4k_mask_stays_at_processing_resolution(self):
        p = plan_canvas(3840, 2160)
        canvas = np.zeros((1088, 1920), np.float32)
        canvas[4:1084] = 0.4321
        result = restore_mask(canvas, p, output_size="processing")
        self.assertEqual(result.shape, (1080, 1920))
        np.testing.assert_array_equal(result, canvas[4:1084])


if __name__ == "__main__":
    unittest.main()
