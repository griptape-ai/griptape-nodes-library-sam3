"""Native adapter for Griptape Modular Diffusion 0.7.1's LTX2 driver.

No model imports happen at library registration. The caller supplies the existing
Alpha-Gen LoRA Pipeline artifact; its cache and activation context remain owned by
the Modular Diffusion library. Calls from this library are serialized.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

_RENDER_LOCK = threading.Lock()


def _cancelled(cancel: Any) -> bool:
    if cancel is None:
        return False
    return bool(cancel() if callable(cancel) else cancel.is_set())


def validate_frames(frames: list[Any], steps: int = 8) -> tuple[int, int, int]:
    if not frames:
        raise ValueError("Alpha generation requires at least one RGB frame.")
    width, height = frames[0].size
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError("LTX canvas dimensions must be positive multiples of 32.")
    if any(frame.size != (width, height) for frame in frames):
        raise ValueError("All LTX conditioning frames must have the same dimensions.")
    if (len(frames) - 1) % 8 or len(frames) > 145:
        raise ValueError("Alpha-Gen requires 8n+1 frames, with at most 145 per call.")
    if steps != 8:
        raise ValueError("This Alpha-Gen adapter uses the verified 8-step distilled profile.")
    return width, height, len(frames)


def validate_pipeline(artifact: Any) -> None:
    if artifact is None or not artifact.pipeline_name.startswith("LTX2"):
        raise ValueError("Connect an LTX-2.5 Alpha-Gen LoRA pipeline, not another diffusion family.")
    build = artifact.build_data
    if "ltx-2.5" not in str(build.get("base_repo_id", "")).lower() or not build.get("is_distilled"):
        raise ValueError("This adapter requires the LTX-2.5 Distilled pipeline used by Alpha-Gen.")
    loras = [(path, weight) for path, weight in artifact.loras.items()]
    for step in artifact.runtime_adapter_steps():
        for entry in step.metadata.get("loras", {}).values():
            if isinstance(entry, dict):
                loras.append((str(entry.get("path", "")), entry.get("weight", 0)))
    if not any("alpha-gen" in str(path).lower().replace("_", "-") and float(weight) != 0 for path, weight in loras):
        raise ValueError(
            "The connected pipeline has no active Alpha-Gen LoRA. Connect your Alpha-Gen LoRA Pipeline output."
        )


def _decode_alpha_float(driver: Any, latent: Any) -> list[Any]:
    """The installed LTX2 driver's VAE path, retaining float output before PIL quantization."""
    import numpy as np
    import torch
    from PIL import Image

    pipe = driver.pipe
    device, dtype = driver._get_device_and_type()
    tensor = latent.to_torch(device=device, dtype=torch.float32)
    timestep = (
        torch.zeros(tensor.shape[0], device=device, dtype=dtype) if pipe.vae.config.timestep_conditioning else None
    )
    tensor = pipe._denormalize_latents(
        tensor, pipe.vae.latents_mean, pipe.vae.latents_std, pipe.vae.config.scaling_factor
    )
    decoded = pipe.vae.decode(tensor.to(device=device, dtype=dtype), timestep, return_dict=False)[0]
    frames = pipe.video_processor.postprocess_video(decoded, output_type="np")[0]
    result = []
    for frame in frames:
        pixels = np.asarray(frame, dtype=np.float32)
        alpha = pixels[..., :3].mean(-1) if pixels.ndim == 3 else pixels
        if alpha.ndim != 2 or not np.isfinite(alpha).all():
            raise RuntimeError("LTX decoded invalid alpha values.")
        result.append(Image.fromarray(np.clip(alpha, 0, 1)))
    return result


def run_alpha(
    pipeline_artifact: Any,
    frames: list[Any],
    seed: int = 42,
    steps: int = 8,
    progress: Callable[[float, str], None] | None = None,
    cancel: Any = None,
    node_name: str | None = None,
) -> list[Any]:
    """Run Alpha-Gen using the existing pipeline, returning canvas-sized PIL frames.

    Selection is handled by the caller. This function deliberately does not
    reinterpret a subject mask as an LTX conditioning-attention mask.
    """
    width, height, count = validate_frames(frames, steps)
    if pipeline_artifact is None:
        raise ValueError("Connect your existing LTX Alpha-Gen LoRA Pipeline output.")
    if _cancelled(cancel):
        raise InterruptedError("Alpha generation cancelled.")
    try:
        import torch
        from diffusers.pipelines.ltx2.pipeline_ltx2_ic_lora import LTX2ReferenceCondition
        from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
        from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import normalize_diffusion_pipeline_value
        from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_factory import create_driver
        from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_types import GeneratorState
    except ImportError as exc:
        raise RuntimeError(
            "Alpha rendering requires Griptape Modular Diffusion Nodes Library 0.7.1 "
            "loaded in the same engine process, with its LTX2 Diffusers dependencies."
        ) from exc

    artifact = normalize_diffusion_pipeline_value(pipeline_artifact, node_name=node_name, raise_on_invalid=True)
    validate_pipeline(artifact)

    def report(fraction: float, message: str) -> None:
        if progress:
            progress(fraction, message)

    # Acquiring in short intervals also makes queued jobs cancellable.
    while not _RENDER_LOCK.acquire(timeout=0.2):
        if _cancelled(cancel):
            raise InterruptedError("Alpha generation cancelled while waiting for the pipeline.")
    try:
        if _cancelled(cancel):
            raise InterruptedError("Alpha generation cancelled.")
        report(0.0, "Using the connected Alpha-Gen pipeline")
        pipe = artifact.get_or_build_pipeline()
        with artifact.activate(pipe, node_name=node_name) as active_pipe, torch.inference_mode():
            driver = create_driver(active_pipe, artifact.pipeline_name)
            generator = GeneratorState.from_seed(int(seed))
            source_shape = (1, 3, count, height, width)
            noise = driver.create_noise_latent(source_shape, generator)
            empty = LatentArtifact.from_torch(torch.zeros_like(noise.to_torch()), source_shape=source_shape, meta={})
            del noise

            def step_callback(current_pipe: Any, index: int, timestep: Any, values: dict) -> dict:
                if _cancelled(cancel):
                    # Raising exits the native activation context and releases LoRA adapters.
                    raise InterruptedError("Alpha generation cancelled.")
                report((index + 1) / (steps + 1), f"Alpha step {index + 1} of {steps}")
                return values

            # Presence of ltx2_ic_lora_reference selects the driver's IC-LoRA
            # variant. A None payload leaves our native in-memory references
            # intact, avoiding an encode/decode round-trip through a temp video.
            latent = driver.denoise_latent(
                empty,
                steps,
                generator_state=generator,
                callback=step_callback,
                start_step=0,
                end_step=-1,
                return_fully_denoised=False,
                ltx2_ic_lora_reference=None,
                reference_conditions=[LTX2ReferenceCondition(frames=frames, strength=1.0)],
                use_stage_2=False,
                prompt="",
                negative_prompt="",
                guidance_scale=1.0,
                audio_guidance_scale=1.0,
                stg_scale=0.0,
                audio_stg_scale=0.0,
                modality_scale=1.0,
                audio_modality_scale=1.0,
                guidance_rescale=0.0,
                audio_guidance_rescale=0.0,
            )
            if _cancelled(cancel):
                raise InterruptedError("Alpha generation cancelled.")
            report(steps / (steps + 1), "Decoding alpha frames")
            decoded = _decode_alpha_float(driver, latent)
            if not isinstance(decoded, list) or len(decoded) != count:
                raise RuntimeError(f"LTX returned an unexpected frame count; expected {count}.")
            if any(frame.size != (width, height) for frame in decoded):
                raise RuntimeError("LTX output framing does not match the prepared canvas.")
            result = decoded
        report(1.0, "Alpha chunk complete")
        return result
    finally:
        _RENDER_LOCK.release()
