import logging
import uuid
from io import BytesIO
from typing import Any

import numpy as np
import requests
from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from PIL import Image

from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter
from griptape_nodes.exe_types.param_components.log_parameter import LogParameter
from griptape_nodes.traits.slider import Slider

# SAM3 imports are done lazily in _load_model() to allow installation first

logger = logging.getLogger("sam3_nodes_library")


class Sam3SegmentImage(SuccessFailureNode):
    """SAM3 Image Segmentation Node

    Segments objects in images using SAM3 (Segment Anything with Concepts)
    with text prompts to identify and segment specific objects.
    """

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        # Model selection parameter (triggers model manager if not downloaded)
        self._model_repo_parameter = HuggingFaceRepoParameter(
            self,
            repo_ids=["facebook/sam3"],
            parameter_name="model",
        )
        self._model_repo_parameter.add_input_parameters()

        # Input parameters
        self.add_parameter(
            Parameter(
                name="input_image",
                input_types=["ImageArtifact", "ImageUrlArtifact"],
                type="ImageArtifact",
                tooltip="The input image to segment",
            )
        )

        self.add_parameter(
            Parameter(
                name="text_prompt",
                input_types=["str"],
                type="str",
                default_value="",
                tooltip="Text prompt describing what to segment (e.g., 'person', 'car', 'dog')",
            )
        )

        self.add_parameter(
            Parameter(
                name="max_masks",
                input_types=["int"],
                type="int",
                default_value=10,
                tooltip="Maximum number of masks to return",
                traits={Slider(min_val=1, max_val=100)},
            )
        )

        self.add_parameter(
            Parameter(
                name="score_threshold",
                input_types=["float"],
                type="float",
                default_value=0.5,
                tooltip="Minimum confidence score for masks (0.0 to 1.0)",
                traits={Slider(min_val=0.0, max_val=1.0)},
            )
        )

        # Output parameters
        self.add_parameter(
            Parameter(
                name="output_masks",
                output_type="list",
                tooltip="List of segmentation masks as ImageUrlArtifacts",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            Parameter(
                name="output_composite",
                output_type="ImageUrlArtifact",
                tooltip="Composite image with all masks overlaid",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            Parameter(
                name="num_masks_found",
                output_type="int",
                tooltip="Number of masks found",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.log_params = LogParameter(self)
        self.log_params.add_output_parameters()

        # Add status parameters for success/failure tracking
        self._create_status_parameters(
            result_details_tooltip="Details about the segmentation operation result",
            result_details_placeholder="Segmentation results will appear here.",
        )

        # Model caching
        self._model = None
        self._processor = None

    def validate_before_node_run(self) -> list[Exception] | None:
        """Validate inputs before running the node"""
        errors: list[Exception] = []

        # Validate model is available
        model_errors = self._model_repo_parameter.validate_before_node_run()
        if model_errors:
            errors.extend(model_errors)

        text_prompt = self.get_parameter_value("text_prompt")
        if not text_prompt or text_prompt.strip() == "":
            errors.append(ValueError("Text prompt cannot be empty. Please provide a description of what to segment."))

        return errors if errors else None

    def process(self) -> None:
        """Main processing logic"""
        # Reset execution state at the start of each run
        self._clear_execution_status()

        try:
            # Get input parameters
            input_image_artifact = self.get_parameter_value("input_image")
            text_prompt = self.get_parameter_value("text_prompt")
            max_masks = self.get_parameter_value("max_masks")
            score_threshold = self.get_parameter_value("score_threshold")

            if not input_image_artifact:
                error_details = "No input image provided"
                self._set_status_results(was_successful=False, result_details=f"FAILURE: {error_details}")
                self._handle_failure_exception(ValueError(error_details))
                return

            self.log_params.append_to_logs("Starting SAM3 segmentation...\n")
            self.log_params.append_to_logs(f"Text prompt: {text_prompt}\n")

            # Convert image artifact to PIL Image
            input_image = self._artifact_to_pil(input_image_artifact)

            self.log_params.append_to_logs(f"Image size: {input_image.size}\n")

            # Load or initialize model
            self._load_model()

            # Set the image in the processor
            self.log_params.append_to_logs("Processing image...\n")
            inference_state = self._processor.set_image(input_image)

            # Run segmentation with text prompt
            self.log_params.append_to_logs(f"Segmenting '{text_prompt}'...\n")
            output = self._processor.set_text_prompt(
                state=inference_state,
                prompt=text_prompt
            )

            # Extract results
            masks = output["masks"]
            scores = output["scores"]

            # Filter by score threshold
            filtered_indices = [i for i, score in enumerate(scores) if score >= score_threshold]
            filtered_masks = [masks[i] for i in filtered_indices[:max_masks]]
            filtered_scores = [scores[i] for i in filtered_indices[:max_masks]]

            num_found = len(filtered_masks)
            self.log_params.append_to_logs(f"Found {num_found} masks with score >= {score_threshold}\n")

            # Convert masks to image artifacts
            mask_artifacts = []
            for i, mask in enumerate(filtered_masks):
                mask_img = self._mask_to_image(mask)
                mask_artifact = self._pil_to_artifact(mask_img)
                mask_artifacts.append(mask_artifact)
                score_val = filtered_scores[i].item() if hasattr(filtered_scores[i], 'item') else float(filtered_scores[i])
                self.log_params.append_to_logs(f"Mask {i+1}: score={score_val:.3f}\n")

            # Create composite image with all masks overlaid
            composite_image = self._create_composite(input_image, filtered_masks)
            composite_artifact = self._pil_to_artifact(composite_image)

            # Set output parameters
            self.set_parameter_value("output_masks", mask_artifacts)
            self.set_parameter_value("output_composite", composite_artifact)
            self.set_parameter_value("num_masks_found", num_found)

            self.log_params.append_to_logs("Segmentation complete!\n")

            # Publish outputs
            self.parameter_output_values["output_masks"] = mask_artifacts
            self.parameter_output_values["output_composite"] = composite_artifact
            self.parameter_output_values["num_masks_found"] = num_found

            # Set success status
            success_details = (
                f"Segmentation completed successfully\n"
                f"Text prompt: {text_prompt}\n"
                f"Masks found: {num_found}\n"
                f"Score threshold: {score_threshold}"
            )
            self._set_status_results(was_successful=True, result_details=f"SUCCESS: {success_details}")

        except Exception as e:
            error_msg = f"Error during segmentation: {str(e)}"
            self.log_params.append_to_logs(f"{error_msg}\n")
            logger.error(error_msg, exc_info=True)

            # Set failure status
            failure_details = (
                f"Segmentation failed\n"
                f"Text prompt: {text_prompt if 'text_prompt' in locals() else 'N/A'}\n"
                f"Error: {str(e)}\n"
                f"Exception type: {type(e).__name__}"
            )
            self._set_status_results(was_successful=False, result_details=f"FAILURE: {failure_details}")
            self._handle_failure_exception(e)

    def _load_model(self) -> None:
        """Load or cache the SAM3 model"""
        if self._model is not None:
            self.log_params.append_to_logs("Using cached model\n")
            return

        self.log_params.append_to_logs("Loading SAM3 model from Hugging Face...\n")

        # Add _sam3_repo to sys.path if not present (needed because .pth files
        # aren't processed when running from griptape-nodes' venv)
        import sys
        from pathlib import Path
        sam3_repo_path = str(Path(__file__).parent / "_sam3_repo")
        if sam3_repo_path not in sys.path:
            sys.path.insert(0, sam3_repo_path)

        try:
            # Lazy import SAM3 modules (installed by sam3_library_advanced.py)
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            # Load the model (downloads from Hugging Face automatically)
            self._model = build_sam3_image_model()

            # Get score threshold from parameter
            score_threshold = self.get_parameter_value("score_threshold")
            self._processor = Sam3Processor(self._model, confidence_threshold=score_threshold)

            self.log_params.append_to_logs("Model loaded successfully\n")

        except ImportError as e:
            error_msg = "SAM3 library not installed. Please check the installation logs."
            self.log_params.append_to_logs(f"{error_msg}\n")
            raise ImportError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            self.log_params.append_to_logs(f"{error_msg}\n")
            raise

    def _artifact_to_pil(self, artifact: ImageArtifact | ImageUrlArtifact) -> Image.Image:
        """Convert image artifact to PIL Image"""
        if isinstance(artifact, ImageUrlArtifact):
            # For ImageUrlArtifact, we need to fetch the image
            response = requests.get(artifact.value)
            return Image.open(BytesIO(response.content)).convert("RGB")
        elif isinstance(artifact, ImageArtifact):
            # For ImageArtifact, get the image data
            # Handle different ways ImageArtifact might store data
            if hasattr(artifact, 'value') and isinstance(artifact.value, bytes):
                return Image.open(BytesIO(artifact.value)).convert("RGB")
            elif hasattr(artifact, 'to_bytes'):
                return Image.open(BytesIO(artifact.to_bytes())).convert("RGB")
            else:
                # Try to convert to bytes
                return Image.open(BytesIO(bytes(artifact.value))).convert("RGB")
        else:
            raise ValueError(f"Unsupported artifact type: {type(artifact)}")

    def _pil_to_artifact(self, image: Image.Image) -> ImageUrlArtifact:
        """Convert PIL Image to ImageUrlArtifact"""
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        filename = f"{uuid.uuid4()}.png"
        url = GriptapeNodes.StaticFilesManager().save_static_file(image_bytes, filename)
        return ImageUrlArtifact(url)

    def _mask_to_image(self, mask: Any) -> Image.Image:
        """Convert mask array to PIL Image"""
        # Convert mask to numpy array if it isn't already
        if not isinstance(mask, np.ndarray):
            # Handle PyTorch tensors (may be on GPU)
            if hasattr(mask, 'cpu'):
                mask = mask.cpu().numpy()
            else:
                mask = np.array(mask)

        # Ensure mask is 2D
        if mask.ndim > 2:
            mask = mask.squeeze()

        # Convert to uint8 (0 or 255)
        mask_uint8 = (mask * 255).astype(np.uint8)

        return Image.fromarray(mask_uint8, mode="L")

    def _create_composite(self, original: Image.Image, masks: list) -> Image.Image:
        """Create a composite image with masks overlaid on the original"""
        # Create a copy of the original image
        composite = original.copy().convert("RGBA")

        # Color palette for different masks
        colors = [
            (255, 0, 0, 100),    # Red
            (0, 255, 0, 100),    # Green
            (0, 0, 255, 100),    # Blue
            (255, 255, 0, 100),  # Yellow
            (255, 0, 255, 100),  # Magenta
            (0, 255, 255, 100),  # Cyan
            (255, 128, 0, 100),  # Orange
            (128, 0, 255, 100),  # Purple
        ]

        for i, mask in enumerate(masks):
            # Convert mask to numpy array
            if not isinstance(mask, np.ndarray):
                # Handle PyTorch tensors (may be on GPU)
                if hasattr(mask, 'cpu'):
                    mask = mask.cpu().numpy()
                else:
                    mask = np.array(mask)

            if mask.ndim > 2:
                mask = mask.squeeze()

            # Create colored overlay
            color = colors[i % len(colors)]
            overlay = Image.new("RGBA", composite.size, (0, 0, 0, 0))

            # Resize mask to match image size if needed
            if mask.shape[:2] != (composite.size[1], composite.size[0]):
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                mask_img = mask_img.resize(composite.size, Image.Resampling.NEAREST)
                mask = np.array(mask_img) > 128

            # Apply color where mask is True
            overlay_array = np.array(overlay)
            overlay_array[mask] = color
            overlay = Image.fromarray(overlay_array)

            # Composite the overlay
            composite = Image.alpha_composite(composite, overlay)

        # Convert back to RGB
        return composite.convert("RGB")
