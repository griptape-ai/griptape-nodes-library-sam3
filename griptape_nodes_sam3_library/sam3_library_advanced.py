"""SAM3/SAM3.1 Library Advanced

SAM3.1 adds Object Multiplex for ~7x faster multi-object tracking.
"""

import logging

from griptape_nodes.node_library.advanced_node_library import AdvancedNodeLibrary
from griptape_nodes.node_library.library_registry import Library, LibrarySchema

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sam3_library")


class Sam3LibraryAdvanced(AdvancedNodeLibrary):
    """Advanced library implementation for SAM3/SAM3.1 (Segment Anything with Concepts).

    SAM3.1 adds Object Multiplex for ~7x faster multi-object tracking.
    """

    def before_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        """Called before any nodes are loaded from the library."""
        msg = f"Starting to load nodes for '{library_data.name}' library..."
        logger.info(msg)

    def after_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        """Called after all nodes have been loaded from the library."""
        msg = f"Finished loading nodes for '{library_data.name}' library"
        logger.info(msg)

        # Configure PyTorch for optimal GPU performance
        self._configure_pytorch_settings()

    def _configure_pytorch_settings(self) -> None:
        """Configure PyTorch TF32 settings for Ampere+ GPUs.

        torch is an execution dependency, so it is importable only in the worker that hosts
        this library's execution. On the orchestrator this is a no-op.
        """
        try:
            import torch

            # Enable TF32 for Ampere+ GPUs (significant speedup with minimal precision loss)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.debug("PyTorch TF32 settings enabled for GPU acceleration")
        except ImportError:
            logger.debug("torch not importable here (orchestrator); skipping TF32 configuration")
