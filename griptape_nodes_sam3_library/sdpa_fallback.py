import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def allow_sdpa_fallback() -> None:
    """Let SAM3's decoder attention fall back when this torch build has no flash attention (e.g. Windows wheels)."""
    if torch.backends.cuda.is_flash_attention_available():
        return

    # The nodes add the sam3 repo to sys.path at runtime, so it can't be imported at module level.
    from sam3.model import decoder

    if not hasattr(decoder, "sdpa_kernel"):
        msg = (
            "Attempted to patch 'sam3.model.decoder.sdpa_kernel' for the flash-attention fallback. "
            "Failed because the module no longer exposes 'sdpa_kernel'; the sam3 submodule likely changed."
        )
        raise AttributeError(msg)

    # Deliberately ignores the caller's backends: the only upstream call requests FLASH_ATTENTION alone.
    def sdpa_kernel_with_fallback(backends, *args, **kwargs):
        return sdpa_kernel(
            [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH], *args, **kwargs
        )

    decoder.sdpa_kernel = sdpa_kernel_with_fallback
