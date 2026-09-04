"""The SAM3 model path: everything that needs sam3 or torch importable.

Normal top-of-file imports are correct here because this module only ever loads where
the execution dependencies exist. The builders and processor are re-exported under
their upstream names so node code reads the same as importing sam3 directly.
"""

import gc

import torch
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_multiplex_video_predictor, build_sam3_video_predictor

__all__ = [
    "Sam3Processor",
    "build_sam3_image_model",
    "build_sam3_multiplex_video_predictor",
    "build_sam3_video_predictor",
    "cuda_device_count",
    "gpu_diagnostics",
    "release_cuda",
    "run_with_autocast",
]


def cuda_device_count() -> int:
    """Number of CUDA devices, 0 when CUDA is unavailable."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def gpu_diagnostics() -> str:
    """The GPU diagnostic lines the video nodes log before building a predictor."""
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    lines = (
        f"GPU diagnostics: torch={torch.__version__}, "
        f"cuda_built={torch.version.cuda}, "
        f"cuda_available={cuda_available}, "
        f"device_count={device_count}\n"
    )
    if cuda_available:
        for i in range(device_count):
            lines += f"  GPU {i}: {torch.cuda.get_device_name(i)}\n"
    return lines


def run_with_autocast(func, *args, **kwargs):
    """Run a function under bfloat16 autocast for SAM3's fused ops."""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return func(*args, **kwargs)


def release_cuda() -> bool:
    """Collect garbage and clear the CUDA cache. Returns whether a cache was cleared."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return True
    return False
