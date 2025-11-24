"""SAM3 Library Advanced - Handles installation and setup for SAM3 dependencies"""

import logging
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from griptape_nodes.node_library.advanced_node_library import AdvancedNodeLibrary
from griptape_nodes.node_library.library_registry import Library, LibrarySchema

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sam3_library")

# Version constants for SAM3 dependencies
PYTORCH_VERSION = "2.7.0"
CUDA_VERSION = "12.6"


class Sam3LibraryAdvanced(AdvancedNodeLibrary):
    """Advanced library implementation for SAM3 (Segment Anything with Concepts)."""

    def before_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        """Called before any nodes are loaded from the library.

        This method handles the installation of SAM3 and its dependencies.
        """
        msg = f"Starting to load nodes for '{library_data.name}' library..."
        logger.info(msg)

        # Check if all dependencies are properly installed
        if self._check_dependencies_installed():
            logger.info("All SAM3 dependencies are already installed, skipping installation")
            return

        logger.info("SAM3 or dependencies not found, beginning installation process...")

        # Install dependencies
        self._install_sam3_dependencies()

    def after_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        """Called after all nodes have been loaded from the library."""
        msg = f"Finished loading nodes for '{library_data.name}' library"
        logger.info(msg)

    def _check_dependencies_installed(self) -> bool:
        """Check if all required dependencies are properly installed.

        Verifies exactly what this script installs:
        - torch=={PYTORCH_VERSION} with CUDA {CUDA_VERSION}
        - torchvision
        - torchaudio
        - sam3 package (from submodule)
        """
        try:
            # Check PyTorch version
            torch_version = version("torch")
            if torch_version != PYTORCH_VERSION:
                logger.info(f"PyTorch version mismatch: expected {PYTORCH_VERSION}, got {torch_version}")
                return False

            # Check CUDA support (need to import torch for runtime check)
            import torch
            if not torch.cuda.is_available():
                logger.info("PyTorch CUDA support not available")
                return False
            if not torch.version.cuda.startswith(CUDA_VERSION):
                logger.info(f"CUDA version mismatch: expected {CUDA_VERSION}, got {torch.version.cuda}")
                return False
            logger.info(f"✓ torch {torch_version} with CUDA {torch.version.cuda}")

            # Check torchvision
            torchvision_version = version("torchvision")
            logger.info(f"✓ torchvision {torchvision_version}")

            # Check torchaudio
            torchaudio_version = version("torchaudio")
            logger.info(f"✓ torchaudio {torchaudio_version}")

            # Check SAM3 package
            sam3_version = version("sam3")
            logger.info(f"✓ sam3 {sam3_version}")

            return True

        except PackageNotFoundError as e:
            logger.info(f"Dependency check failed: {e}")
            return False

    def _install_sam3_dependencies(self) -> None:
        """Install SAM3 and required dependencies."""
        try:
            logger.info("=" * 80)
            logger.info("Installing SAM3 Library Dependencies...")
            logger.info("=" * 80)

            # Step 1: Install PyTorch with CUDA support
            cuda_version_short = CUDA_VERSION.replace(".", "")  # "12.6" -> "126"
            logger.info(f"Step 1/3: Installing PyTorch {PYTORCH_VERSION} with CUDA {CUDA_VERSION} support...")
            self._run_pip_install([
                f"torch=={PYTORCH_VERSION}",
                "torchvision",
                "torchaudio",
                "--index-url",
                f"https://download.pytorch.org/whl/cu{cuda_version_short}"
            ])
            logger.info("PyTorch installation complete")

            # Step 2: Initialize SAM3 submodule
            logger.info("Step 2/3: Initializing SAM3 submodule...")
            sam3_submodule_dir = self._init_sam3_submodule()
            logger.info(f"SAM3 submodule initialized at: {sam3_submodule_dir}")

            # Step 3: Install SAM3 in editable mode
            logger.info("Step 3/3: Installing SAM3 package...")
            self._install_sam3_package(sam3_submodule_dir)

            logger.info("SAM3 installation completed successfully!")
            logger.info("=" * 80)
            logger.warning("IMPORTANT: Before using SAM3, you must:")
            logger.warning("  1. Request access to SAM3 checkpoints on Hugging Face")
            logger.warning("  2. Set HF_TOKEN in the Model Management settings:")
            logger.warning("     https://app.nodes.griptape.ai/#model-management")
            logger.info("=" * 80)

        except Exception as e:
            error_msg = f"Failed to install SAM3 dependencies: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def _run_pip_install(self, packages: list[str]) -> None:
        """Run pip install with the given packages."""
        cmd = [sys.executable, "-m", "pip", "install"] + packages
        logger.info(f"Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )

        if result.stdout:
            logger.debug(result.stdout)
        if result.stderr:
            logger.debug(result.stderr)

    def _init_sam3_submodule(self) -> Path:
        """Initialize the SAM3 git submodule."""
        # Get the library root directory (parent of this file's directory)
        library_root = Path(__file__).parent.parent
        sam3_submodule_dir = library_root / "griptape_nodes_sam3_library" / "sam3"

        logger.info(f"Library root: {library_root}")
        logger.info(f"Expected submodule path: {sam3_submodule_dir}")

        # Check if submodule is already initialized (has contents)
        if sam3_submodule_dir.exists() and any(sam3_submodule_dir.iterdir()):
            logger.info("SAM3 submodule already initialized")
            return sam3_submodule_dir

        # Initialize submodule
        logger.info("Initializing git submodule...")
        result = subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=library_root,
            check=True,
            capture_output=True,
            text=True
        )

        if result.stdout:
            logger.info(result.stdout)
        if result.stderr:
            logger.debug(result.stderr)

        # Verify submodule was initialized
        if not sam3_submodule_dir.exists() or not any(sam3_submodule_dir.iterdir()):
            raise RuntimeError(
                f"Submodule initialization failed: {sam3_submodule_dir} is empty or does not exist"
            )

        return sam3_submodule_dir

    def _install_sam3_package(self, sam3_dir: Path) -> None:
        """Install the SAM3 package from the submodule."""
        cmd = [sys.executable, "-m", "pip", "install", str(sam3_dir)]
        logger.info(f"Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )

        if result.stdout:
            logger.debug(result.stdout)
        if result.stderr:
            logger.error(result.stderr)
