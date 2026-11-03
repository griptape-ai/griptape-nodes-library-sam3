import contextlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "griptape_nodes_sam3_library"))
from selective_matte import ltx_runner


class Artifact:
    pipeline_name = "LTX2Pipeline"
    build_data = {"base_repo_id": "Lightricks/LTX-2.5-Diffusers", "is_distilled": True}
    loras = {}

    def __init__(self):
        self.entered = self.exited = self.builds = 0
        self.pipe = object()

    def runtime_adapter_steps(self):
        return [
            types.SimpleNamespace(metadata={"loras": {"alpha": {"path": "ltx-2.5-alpha-gen.safetensors", "weight": 1}}})
        ]

    def get_or_build_pipeline(self):
        self.builds += 1
        return self.pipe

    @contextlib.contextmanager
    def activate(self, pipe, node_name=None):
        assert pipe is self.pipe
        self.entered += 1
        try:
            yield pipe
        finally:
            self.exited += 1


class NativeLatent:
    def to_torch(self, **kwargs):
        return np.zeros((1, 128, 2, 1, 1))

    @classmethod
    def from_torch(cls, value, **kwargs):
        return cls()


def module(name, **values):
    obj = types.ModuleType(name)
    obj.__dict__.update(values)
    return obj


class RunnerTests(unittest.TestCase):
    def frames(self):
        return [Image.new("RGB", (32, 32)) for _ in range(9)]

    def native_modules(self, driver):
        names = {
            "torch": module("torch", inference_mode=contextlib.nullcontext, zeros_like=np.zeros_like),
            "diffusers.pipelines.ltx2.pipeline_ltx2_ic_lora": module(
                "native_ltx", LTX2ReferenceCondition=lambda **kw: kw
            ),
            "modular_diffusion_nodes_library.artifact_utils.latent_artifact": module(
                "latent", LatentArtifact=NativeLatent
            ),
            "modular_diffusion_nodes_library.artifact_utils.pipeline_artifact": module(
                "artifact", normalize_diffusion_pipeline_value=lambda x, **kw: x
            ),
            "modular_diffusion_nodes_library.latent_pipeline_drivers.driver_factory": module(
                "factory", create_driver=lambda *a: driver
            ),
            "modular_diffusion_nodes_library.latent_pipeline_drivers.driver_types": module(
                "types", GeneratorState=types.SimpleNamespace(from_seed=lambda seed: seed)
            ),
        }
        return patch.dict(sys.modules, names)

    def test_rejects_bad_alignment_and_non_alpha_pipeline_before_build(self):
        with self.assertRaises(ValueError):
            ltx_runner.validate_frames([Image.new("RGB", (31, 32))])
        with self.assertRaises(ValueError):
            ltx_runner.validate_frames(self.frames()[:8])
        artifact = Artifact()
        artifact.runtime_adapter_steps = lambda: []
        with self.assertRaisesRegex(ValueError, "no active Alpha"):
            ltx_runner.validate_pipeline(artifact)
        self.assertEqual(artifact.builds, 0)

    def test_native_conditioning_and_context_cleanup(self):
        calls = []

        class Driver:
            def create_noise_latent(self, shape, generator):
                self.shape = shape
                return NativeLatent()

            def denoise_latent(self, latent, steps, **kwargs):
                calls.append((steps, kwargs))
                kwargs["callback"](None, 0, 1, {"latents": "keep"})
                return latent

        artifact, driver = Artifact(), Driver()
        floats = [Image.fromarray(np.full((32, 32), 0.123456, np.float32)) for _ in range(9)]
        progress = []
        with self.native_modules(driver), patch.object(ltx_runner, "_decode_alpha_float", return_value=floats):
            result = ltx_runner.run_alpha(artifact, self.frames(), progress=lambda *a: progress.append(a))
        self.assertEqual((artifact.entered, artifact.exited), (1, 1))
        self.assertEqual(driver.shape, (1, 3, 9, 32, 32))
        self.assertEqual(calls[0][0], 8)
        self.assertEqual(calls[0][1]["prompt"], "")
        self.assertIsNone(calls[0][1]["ltx2_ic_lora_reference"])
        self.assertNotIn("conditioning_attention_mask", calls[0][1])
        self.assertEqual(len(calls[0][1]["reference_conditions"][0]["frames"]), 9)
        self.assertEqual(result[0].mode, "F")
        self.assertAlmostEqual(float(np.asarray(result[0])[0, 0]), 0.123456, places=6)
        self.assertEqual(progress[-1][0], 1.0)

    def test_failure_releases_adapters_and_lock(self):
        class Driver:
            def create_noise_latent(self, *a):
                return NativeLatent()

            def denoise_latent(self, *a, **kw):
                raise InterruptedError("cancelled")

        artifact = Artifact()
        with self.native_modules(Driver()):
            with self.assertRaises(InterruptedError):
                ltx_runner.run_alpha(artifact, self.frames())
        self.assertEqual(artifact.exited, 1)
        self.assertTrue(ltx_runner._RENDER_LOCK.acquire(blocking=False))
        ltx_runner._RENDER_LOCK.release()

    def test_float_decode_uses_numpy_processor_without_audio_or_pil_quantization(self):
        class Tensor:
            shape = (1, 3, 9, 32, 32)

            def to(self, **kw):
                return self

        calls = []
        processor = types.SimpleNamespace(
            postprocess_video=lambda tensor, **kw: calls.append(kw) or np.full((1, 9, 32, 32, 3), 0.123456, np.float32)
        )
        pipe = types.SimpleNamespace(
            vae=types.SimpleNamespace(
                config=types.SimpleNamespace(timestep_conditioning=False, scaling_factor=1),
                latents_mean=0,
                latents_std=1,
                decode=lambda *a, **kw: [Tensor()],
            ),
            video_processor=processor,
            _denormalize_latents=lambda *a: a[0],
        )
        driver = types.SimpleNamespace(pipe=pipe, _get_device_and_type=lambda: ("cpu", "float32"))
        latent = types.SimpleNamespace(to_torch=lambda **kw: Tensor())
        with patch.dict(sys.modules, {"torch": module("torch", float32="float32")}):
            decoded = ltx_runner._decode_alpha_float(driver, latent)
        self.assertEqual(calls, [{"output_type": "np"}])
        self.assertEqual(decoded[0].mode, "F")
        self.assertAlmostEqual(float(np.asarray(decoded[0])[0, 0]), 0.123456, places=6)


if __name__ == "__main__":
    unittest.main()
