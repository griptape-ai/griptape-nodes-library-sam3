"""SAM3 Library Advanced - Handles installation and setup for SAM3 dependencies"""

import logging
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from griptape_nodes.node_library.advanced_node_library import AdvancedNodeLibrary
from griptape_nodes.node_library.library_registry import Library, LibrarySchema
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sam3_library")


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

        # Configure PyTorch for optimal GPU performance
        self._configure_pytorch_settings()

    def _get_library_root(self) -> Path:
        """Get the library root directory (where .venv lives)."""
        return Path(__file__).parent

    def _get_venv_python_path(self) -> Path:
        """Get the Python executable path from the library's venv.

        Returns the path to the venv's Python executable, which differs between
        Windows (Scripts/python.exe) and Unix (bin/python).
        """
        venv_path = self._get_library_root() / ".venv"

        if GriptapeNodes.OSManager().is_windows():
            venv_python_path = venv_path / "Scripts" / "python.exe"
        else:
            venv_python_path = venv_path / "bin" / "python"

        if not venv_python_path.exists():
            raise RuntimeError(
                f"Library venv Python not found at {venv_python_path}. "
                "The library venv must be initialized before loading."
            )

        logger.debug(f"Python executable found at: {venv_python_path}")
        return venv_python_path

    def _configure_pytorch_settings(self) -> None:
        """Configure PyTorch TF32 settings for Ampere+ GPUs."""
        try:
            import torch

            # Enable TF32 for Ampere+ GPUs (significant speedup with minimal precision loss)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("PyTorch TF32 settings enabled for GPU acceleration")
        except ImportError:
            logger.warning("PyTorch not available, skipping TF32 configuration")

    def _check_dependencies_installed(self) -> bool:
        """Check if sam3 is installed. Debug log found versions."""
        try:
            # Check SAM3 package (the main dependency we install)
            sam3_version = version("sam3")
            logger.debug(f"Found sam3 {sam3_version}")

            # Log other dependencies for debugging
            try:
                import torch
                logger.debug(f"Found torch {torch.__version__}, CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
            except ImportError:
                logger.debug("torch not found")

            return True

        except PackageNotFoundError:
            logger.debug("sam3 not found")
            return False

    def _install_sam3_dependencies(self) -> None:
        """Install SAM3 and required dependencies."""
        try:
            logger.info("=" * 80)
            logger.info("Installing SAM3 Library Dependencies...")
            logger.info("=" * 80)

            # Ensure pip is available in the venv
            self._ensure_pip_installed()

            # Step 1/3: Install triton (platform-specific)
            logger.info("Step 1/3: Installing triton...")
            if GriptapeNodes.OSManager().is_windows():
                self._run_pip_install(["triton-windows"])
            else:
                self._run_pip_install(["triton"])

            # Step 2/3: Initialize SAM3 submodule
            logger.info("Step 2/3: Initializing SAM3 submodule...")
            sam3_submodule_dir = self._init_sam3_submodule()

            # Step 3/3: Install SAM3 in editable mode
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

    def _ensure_pip_installed(self) -> None:
        """Ensure pip is installed in the library's venv."""
        python_path = self._get_venv_python_path()

        # Check if pip is available
        result = subprocess.run(
            [str(python_path), "-m", "pip", "--version"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            logger.debug(f"pip already installed: {result.stdout.strip()}")
            return

        # pip not found, install it using ensurepip
        logger.info("pip not found in venv, installing with ensurepip...")
        subprocess.run(
            [str(python_path), "-m", "ensurepip", "--upgrade"],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("pip installed successfully")

    def _run_pip_install(self, packages: list[str]) -> None:
        """Run pip install with the given packages using the library's venv."""
        python_path = self._get_venv_python_path()
        cmd = [str(python_path), "-m", "pip", "install"] + packages
        logger.info(f"Running: {' '.join(cmd)}")

        try:
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
        except subprocess.CalledProcessError as e:
            logger.error(f"pip install failed with exit code {e.returncode}")
            if e.stdout:
                logger.error(f"stdout: {e.stdout}")
            if e.stderr:
                logger.error(f"stderr: {e.stderr}")
            raise

    def _init_sam3_submodule(self) -> Path:
        """Initialize the SAM3 git submodule."""
        library_root = self._get_library_root()
        sam3_submodule_dir = library_root / "_sam3_repo"

        # Check if submodule is already initialized (has contents)
        if sam3_submodule_dir.exists() and any(sam3_submodule_dir.iterdir()):
            return sam3_submodule_dir

        # Initialize submodule (git command runs from repo root, one level up)
        git_repo_root = library_root.parent
        subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=git_repo_root,
            check=True,
            capture_output=True,
            text=True
        )

        # Verify submodule was initialized
        if not sam3_submodule_dir.exists() or not any(sam3_submodule_dir.iterdir()):
            raise RuntimeError(
                f"Submodule initialization failed: {sam3_submodule_dir} is empty or does not exist"
            )

        return sam3_submodule_dir

    def _install_sam3_package(self, sam3_dir: Path) -> None:
        """Install the SAM3 package from the submodule.

        NOTE: Uses editable mode (-e) because SAM3's pyproject.toml has a packaging bug
        that only includes sam3 and sam3.model, but not sam3.sam and other required
        subpackages. Editable mode symlinks to the source and includes all packages.

        Also installs with [notebooks] extras which includes einops and other
        dependencies not listed in the base requirements.
        """
        # Use compat mode for editable install to create .pth file linking to source
        self._run_pip_install([
            "--config-settings", "editable_mode=compat",
            "-e", f"{sam3_dir}[notebooks]"
        ])
