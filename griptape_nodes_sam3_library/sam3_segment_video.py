import logging
import shutil
import tempfile
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter
from griptape_nodes.exe_types.param_components.log_parameter import LogParameter
from griptape_nodes.traits.slider import Slider

# SAM3 imports are done lazily in _load_model() to allow installation first

logger = logging.getLogger("sam3_nodes_library")


class Sam3SegmentVideo(SuccessFailureNode):
    """SAM3 Video Segmentation Node

    Segments objects in videos using SAM3 (Segment Anything with Concepts)
    with text prompts to identify and segment specific objects across frames.
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
                name="input_video",
                input_types=["VideoUrlArtifact"],
                type="VideoUrlArtifact",
                tooltip="The input video to segment",
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
                name="prompt_frame",
                input_types=["int"],
                type="int",
                default_value=0,
                tooltip="Frame index to apply the initial prompt (0 = first frame)",
            )
        )

        self.add_parameter(
            Parameter(
                name="mask_opacity",
                input_types=["float"],
                type="float",
                default_value=0.4,
                tooltip="Opacity of the mask overlay (0.0 to 1.0)",
                traits={Slider(min_val=0.0, max_val=1.0)},
            )
        )

        # Output parameters
        self.add_parameter(
            Parameter(
                name="output_video",
                output_type="VideoUrlArtifact",
                tooltip="Video with segmentation masks overlaid",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            Parameter(
                name="num_frames_processed",
                output_type="int",
                tooltip="Number of frames processed",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            Parameter(
                name="num_objects_found",
                output_type="int",
                tooltip="Number of objects segmented",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.log_params = LogParameter(self)
        self.log_params.add_output_parameters()

        # Add status parameters for success/failure tracking
        self._create_status_parameters(
            result_details_tooltip="Details about the video segmentation operation result",
            result_details_placeholder="Segmentation results will appear here.",
        )

        # Model caching
        self._predictor = None

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

        temp_dir = None
        session_id = None

        try:
            # Get input parameters
            input_video_artifact = self.get_parameter_value("input_video")
            text_prompt = self.get_parameter_value("text_prompt")
            prompt_frame = self.get_parameter_value("prompt_frame")
            mask_opacity = self.get_parameter_value("mask_opacity")

            if not input_video_artifact:
                error_details = "No input video provided"
                self._set_status_results(was_successful=False, result_details=f"FAILURE: {error_details}")
                self._handle_failure_exception(ValueError(error_details))
                return

            self.log_params.append_to_logs("Starting SAM3 video segmentation...\n")
            self.log_params.append_to_logs(f"Text prompt: {text_prompt}\n")

            # Create temp directory for frames
            temp_dir = Path(tempfile.mkdtemp(prefix="sam3_video_"))
            frames_dir = temp_dir / "frames"
            frames_dir.mkdir()

            # Extract frames from video
            self.log_params.append_to_logs("Extracting video frames...\n")
            video_path, fps, frame_count = self._extract_frames(input_video_artifact, frames_dir)
            self.log_params.append_to_logs(f"Extracted {frame_count} frames at {fps:.2f} FPS\n")

            # Validate prompt_frame
            if prompt_frame >= frame_count:
                prompt_frame = 0
                self.log_params.append_to_logs(f"Prompt frame adjusted to 0 (was >= frame count)\n")

            # Load or initialize model
            self._load_model()

            # Start video session
            self.log_params.append_to_logs("Starting video session...\n")
            response = self._predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=str(frames_dir),
                )
            )
            session_id = response["session_id"]

            # Add text prompt
            self.log_params.append_to_logs(f"Adding prompt '{text_prompt}' at frame {prompt_frame}...\n")
            prompt_response = self._predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=prompt_frame,
                    text=text_prompt,
                )
            )

            # Get number of objects found from initial prompt
            num_objects = len(prompt_response.get("outputs", {}))
            self.log_params.append_to_logs(f"Found {num_objects} object(s) matching prompt\n")

            if num_objects == 0:
                self.log_params.append_to_logs("Warning: No objects found matching the prompt\n")

            # Propagate through video
            self.log_params.append_to_logs("Propagating segmentation through video...\n")
            outputs_per_frame = self._propagate_in_video(session_id)

            num_frames = len(outputs_per_frame)
            self.log_params.append_to_logs(f"Processed {num_frames} frames\n")

            # Create output video with masks overlaid
            self.log_params.append_to_logs("Creating output video with masks...\n")
            output_frames_dir = temp_dir / "output_frames"
            output_frames_dir.mkdir()

            self._create_masked_frames(frames_dir, outputs_per_frame, output_frames_dir, mask_opacity)

            # Encode output video
            output_video_path = temp_dir / "output.mp4"
            self._encode_video(output_frames_dir, output_video_path, fps)

            # Save to static files
            output_artifact = self._video_to_artifact(output_video_path)

            # Close session
            self._predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id,
                )
            )
            session_id = None

            # Set output parameters
            self.set_parameter_value("output_video", output_artifact)
            self.set_parameter_value("num_frames_processed", num_frames)
            self.set_parameter_value("num_objects_found", num_objects)

            self.log_params.append_to_logs("Video segmentation complete!\n")

            # Publish outputs
            self.parameter_output_values["output_video"] = output_artifact
            self.parameter_output_values["num_frames_processed"] = num_frames
            self.parameter_output_values["num_objects_found"] = num_objects

            # Set success status
            success_details = (
                f"Video segmentation completed successfully\n"
                f"Text prompt: {text_prompt}\n"
                f"Frames processed: {num_frames}\n"
                f"Objects found: {num_objects}"
            )
            self._set_status_results(was_successful=True, result_details=f"SUCCESS: {success_details}")

        except Exception as e:
            error_msg = f"Error during video segmentation: {str(e)}"
            self.log_params.append_to_logs(f"{error_msg}\n")
            logger.error(error_msg, exc_info=True)

            # Try to close session if open
            if session_id and self._predictor:
                try:
                    self._predictor.handle_request(
                        request=dict(type="close_session", session_id=session_id)
                    )
                except Exception:
                    pass

            # Set failure status
            failure_details = (
                f"Video segmentation failed\n"
                f"Text prompt: {text_prompt if 'text_prompt' in locals() else 'N/A'}\n"
                f"Error: {str(e)}\n"
                f"Exception type: {type(e).__name__}"
            )
            self._set_status_results(was_successful=False, result_details=f"FAILURE: {failure_details}")
            self._handle_failure_exception(e)

        finally:
            # Clean up temp directory
            if temp_dir and temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temp directory: {cleanup_error}")

    def _load_model(self) -> None:
        """Load or cache the SAM3 video predictor"""
        if self._predictor is not None:
            self.log_params.append_to_logs("Using cached video predictor\n")
            return

        self.log_params.append_to_logs("Loading SAM3 video predictor...\n")

        # Add _sam3_repo to sys.path if not present
        import sys
        sam3_repo_path = str(Path(__file__).parent / "_sam3_repo")
        if sam3_repo_path not in sys.path:
            sys.path.insert(0, sam3_repo_path)

        try:
            import torch
            from sam3.model_builder import build_sam3_video_predictor

            # Get available GPUs
            gpus_to_use = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []

            if not gpus_to_use:
                self.log_params.append_to_logs("Warning: No GPU available, using CPU (will be slow)\n")

            # Build the video predictor
            self._predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

            self.log_params.append_to_logs("Video predictor loaded successfully\n")

        except ImportError as e:
            error_msg = "SAM3 library not installed. Please check the installation logs."
            self.log_params.append_to_logs(f"{error_msg}\n")
            raise ImportError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to load video predictor: {str(e)}"
            self.log_params.append_to_logs(f"{error_msg}\n")
            raise

    def _extract_frames(self, video_artifact, output_dir: Path) -> tuple[Path, float, int]:
        """Extract frames from video artifact to a directory.

        Returns (video_path, fps, frame_count)
        """
        import cv2
        import requests

        # Get video data
        video_url = video_artifact.value
        if video_url.startswith(("http://", "https://")):
            response = requests.get(video_url)
            video_data = response.content
        else:
            # Local file path
            with open(video_url, "rb") as f:
                video_data = f.read()

        # Write to temp file for OpenCV
        temp_video = output_dir.parent / "input_video.mp4"
        temp_video.write_bytes(video_data)

        # Open video and extract frames
        cap = cv2.VideoCapture(str(temp_video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save frame as JPEG (SAM3 expects JPEG frames)
            frame_path = output_dir / f"{frame_count:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_count += 1

        cap.release()

        return temp_video, fps, frame_count

    def _propagate_in_video(self, session_id: str) -> dict:
        """Propagate segmentation through all video frames."""
        outputs_per_frame = {}

        for response in self._predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
            )
        ):
            outputs_per_frame[response["frame_index"]] = response["outputs"]

        return outputs_per_frame

    def _create_masked_frames(
        self, input_dir: Path, outputs_per_frame: dict, output_dir: Path, opacity: float
    ) -> None:
        """Create frames with mask overlays."""
        import cv2

        # Color palette for different objects
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
        ]

        for frame_idx, outputs in outputs_per_frame.items():
            # Load original frame
            frame_path = input_dir / f"{frame_idx:06d}.jpg"
            if not frame_path.exists():
                continue

            frame = cv2.imread(str(frame_path))
            overlay = frame.copy()

            # Apply each object's mask
            for obj_idx, (obj_id, mask) in enumerate(outputs.items()):
                # Handle tensor on GPU if needed
                if hasattr(mask, 'cpu'):
                    mask = mask.cpu().numpy()
                elif not isinstance(mask, np.ndarray):
                    mask = np.array(mask)

                if mask.ndim > 2:
                    mask = mask.squeeze()

                # Resize mask to match frame if needed
                if mask.shape[:2] != frame.shape[:2]:
                    mask = cv2.resize(mask.astype(np.float32), (frame.shape[1], frame.shape[0]))
                    mask = mask > 0.5

                # Apply colored overlay where mask is True
                color = colors[obj_idx % len(colors)]
                overlay[mask > 0] = color

            # Blend overlay with original frame
            output_frame = cv2.addWeighted(frame, 1 - opacity, overlay, opacity, 0)

            # Save output frame
            output_path = output_dir / f"{frame_idx:06d}.jpg"
            cv2.imwrite(str(output_path), output_frame)

    def _encode_video(self, frames_dir: Path, output_path: Path, fps: float) -> None:
        """Encode frames back into a video."""
        import cv2

        # Get list of frames
        frame_files = sorted(frames_dir.glob("*.jpg"))
        if not frame_files:
            raise ValueError("No frames to encode")

        # Read first frame to get dimensions
        first_frame = cv2.imread(str(frame_files[0]))
        height, width = first_frame.shape[:2]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            out.write(frame)

        out.release()

    def _video_to_artifact(self, video_path: Path):
        """Convert video file to VideoUrlArtifact."""
        from griptape.artifacts import VideoUrlArtifact

        video_bytes = video_path.read_bytes()
        filename = f"{uuid.uuid4()}.mp4"
        url = GriptapeNodes.StaticFilesManager().save_static_file(video_bytes, filename)
        return VideoUrlArtifact(url)
