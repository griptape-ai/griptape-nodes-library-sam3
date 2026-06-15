import asyncio
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from griptape.artifacts import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter
from griptape_nodes.exe_types.param_components.log_parameter import LogParameter
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.files.file import File
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
        # Use SAM3.1 Multiplex Video node for faster multi-object tracking with SAM3.1
        self._model_repo_parameter = HuggingFaceRepoParameter(
            self,
            repo_ids=["facebook/sam3.1", "facebook/sam3"],
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
                name="output_masks",
                output_type="list[VideoUrlArtifact]",
                tooltip="List of mask videos as VideoUrlArtifacts (one per segmented object)",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            Parameter(
                name="output_composite",
                output_type="VideoUrlArtifact",
                tooltip="Composite video with all masks overlaid",
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

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="segmented_video.mp4",
        )
        self._output_file.add_parameter()

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

    async def aprocess(self) -> None:
        await self._process()

    async def _process(self) -> None:
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
            video_path, fps, frame_count = await asyncio.to_thread(
                self._extract_frames, input_video_artifact, frames_dir
            )
            self.log_params.append_to_logs(f"Extracted {frame_count} frames at {fps:.2f} FPS\n")

            # Validate prompt_frame
            if prompt_frame >= frame_count:
                prompt_frame = 0
                self.log_params.append_to_logs("Prompt frame adjusted to 0 (was >= frame count)\n")

            # Load or initialize model
            # Note: SAM3 video predictor uses torch.distributed internally,
            # so predictor calls are not wrapped in asyncio.to_thread
            self._load_model()

            # Start video session
            self.log_params.append_to_logs("Starting video session...\n")
            response = self._predictor.handle_request(
                request={
                    "type": "start_session",
                    "resource_path": str(frames_dir),
                }
            )
            session_id = response["session_id"]

            # Add text prompt
            self.log_params.append_to_logs(f"Adding prompt '{text_prompt}' at frame {prompt_frame}...\n")
            prompt_response = self._predictor.handle_request(
                request={
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": prompt_frame,
                    "text": text_prompt,
                }
            )

            # Get number of objects found from initial prompt
            prompt_outputs = prompt_response.get("outputs", {})
            num_objects = len(prompt_outputs.get("out_obj_ids", []))
            self.log_params.append_to_logs(f"Found {num_objects} object(s) matching prompt\n")

            if num_objects == 0:
                self.log_params.append_to_logs("Warning: No objects found matching the prompt\n")

            # Propagate through video
            self.log_params.append_to_logs("Propagating segmentation through video...\n")
            outputs_per_frame = self._propagate_in_video(session_id)

            num_frames = len(outputs_per_frame)
            self.log_params.append_to_logs(f"Processed {num_frames} frames\n")

            # Create individual mask videos and composite video
            self.log_params.append_to_logs("Creating output videos...\n")
            composite_frames_dir = temp_dir / "composite_frames"
            composite_frames_dir.mkdir()

            # Create mask frame directories for each object
            mask_artifacts = []
            mask_frame_dirs = await asyncio.to_thread(
                self._create_masked_frames_multi,
                frames_dir,
                outputs_per_frame,
                composite_frames_dir,
                temp_dir,
                mask_opacity,
                num_objects,
            )

            # Encode individual mask videos (directly to H.264)
            for obj_idx, mask_frames_dir in enumerate(mask_frame_dirs):
                mask_video_path = temp_dir / f"mask_{obj_idx}.mp4"
                await self._encode_video(mask_frames_dir, mask_video_path, fps)
                mask_artifact = self._video_to_artifact(mask_video_path)
                mask_artifacts.append(mask_artifact)
                self.log_params.append_to_logs(f"Created mask video {obj_idx + 1}\n")

            # Encode composite video (directly to H.264)
            composite_video_path = temp_dir / "composite.mp4"
            await self._encode_video(composite_frames_dir, composite_video_path, fps)
            composite_artifact = self._video_to_artifact(composite_video_path)

            # Close session
            self._predictor.handle_request(
                request={
                    "type": "close_session",
                    "session_id": session_id,
                }
            )
            session_id = None

            # Set output parameters
            self.set_parameter_value("output_masks", mask_artifacts)
            self.set_parameter_value("output_composite", composite_artifact)
            self.set_parameter_value("num_frames_processed", num_frames)
            self.set_parameter_value("num_objects_found", len(mask_artifacts))

            self.log_params.append_to_logs("Video segmentation complete!\n")

            # Publish outputs
            self.parameter_output_values["output_masks"] = mask_artifacts
            self.parameter_output_values["output_composite"] = composite_artifact
            self.parameter_output_values["num_frames_processed"] = num_frames
            self.parameter_output_values["num_objects_found"] = len(mask_artifacts)

            # Set success status
            success_details = (
                f"Video segmentation completed successfully\n"
                f"Text prompt: {text_prompt}\n"
                f"Frames processed: {num_frames}\n"
                f"Objects found: {len(mask_artifacts)}"
            )
            self._set_status_results(was_successful=True, result_details=f"SUCCESS: {success_details}")

        except Exception as e:
            error_msg = f"Error during video segmentation: {str(e)}"
            self.log_params.append_to_logs(f"{error_msg}\n")
            logger.error(error_msg, exc_info=True)

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

            # Release VRAM - close session and shutdown predictor
            if self._predictor is not None:
                # Close session first to free GPU resources
                if session_id is not None:
                    try:
                        self._predictor.handle_request(request={"type": "close_session", "session_id": session_id})
                        self.log_params.append_to_logs("Session closed\n")
                    except Exception:
                        pass

                # Then shutdown the predictor
                try:
                    self._predictor.shutdown()
                    self.log_params.append_to_logs("Video predictor shut down\n")
                except Exception as shutdown_error:
                    logger.warning(f"Failed to shutdown predictor: {shutdown_error}")
                del self._predictor
                self._predictor = None

            # Force garbage collection and clear CUDA cache
            try:
                import gc

                gc.collect()

                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.log_params.append_to_logs("CUDA cache cleared\n")
            except Exception:
                pass

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

            # Log GPU/CUDA diagnostic info to help debug cloud deployment issues
            cuda_available = torch.cuda.is_available()
            device_count = torch.cuda.device_count() if cuda_available else 0
            self.log_params.append_to_logs(
                f"GPU diagnostics: torch={torch.__version__}, "
                f"cuda_built={torch.version.cuda}, "
                f"cuda_available={cuda_available}, "
                f"device_count={device_count}\n"
            )
            if cuda_available:
                for i in range(device_count):
                    self.log_params.append_to_logs(f"  GPU {i}: {torch.cuda.get_device_name(i)}\n")

            # Get available GPUs
            gpus_to_use = list(range(device_count))

            if not gpus_to_use:
                self.log_params.append_to_logs(
                    "No GPU available. SAM3 video segmentation requires a GPU. "
                    "If running on Griptape Cloud, ensure the GPU option is enabled on the Start Flow node.\n"
                )

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
        try:
            import cv2

            self.log_params.append_to_logs(
                f"cv2 loaded successfully (version: {getattr(cv2, '__version__', 'unknown')})\n"
            )
        except ImportError as e:
            msg = (
                f"Failed to import cv2: {e}\n"
                "This is likely caused by opencv-python (GUI variant) being installed instead of "
                "opencv-python-headless. The GUI variant requires libGL.so.1 which is not available "
                "in headless environments. Check the SAM3 library setup logs for details."
            )
            self.log_params.append_to_logs(msg + "\n")
            raise ImportError(msg) from e

        # Get video data
        if isinstance(video_artifact, dict):
            video_url = video_artifact["value"]
        else:
            video_url = video_artifact.value
        video_data = File(video_url).read_bytes()

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
            request={
                "type": "propagate_in_video",
                "session_id": session_id,
            }
        ):
            # On multi-GPU setups, non-rank-0 GPUs yield None outputs - skip these
            outputs = response.get("outputs")
            if outputs is not None:
                outputs_per_frame[response["frame_index"]] = outputs

        return outputs_per_frame

    def _create_masked_frames_multi(
        self,
        input_dir: Path,
        outputs_per_frame: dict,
        composite_dir: Path,
        temp_dir: Path,
        opacity: float,
        num_objects: int,
    ) -> list[Path]:
        """Create individual mask frame directories and composite frames.

        Returns list of mask frame directories (one per object) for encoding into separate videos.
        Also creates composite frames in composite_dir with all masks overlaid.
        """
        import cv2

        # Color palette for different objects
        colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
        ]

        if num_objects == 0:
            logger.warning("No objects found during initial prompt")
            return []

        # Create directories for each object's mask frames
        mask_frame_dirs = []
        for obj_idx in range(num_objects):
            mask_dir = temp_dir / f"mask_frames_{obj_idx}"
            mask_dir.mkdir()
            mask_frame_dirs.append(mask_dir)

        for frame_idx, outputs in outputs_per_frame.items():
            # Load original frame
            frame_path = input_dir / f"{frame_idx:06d}.jpg"
            if not frame_path.exists():
                # Create black frames for all mask videos for missing input frames
                logger.warning(f"Frame {frame_idx}: Input frame not found. Creating black mask frames.")
                for obj_idx in range(num_objects):
                    # Use dimensions from first available frame or default
                    if obj_idx < len(mask_frame_dirs):
                        existing_frames = list(mask_frame_dirs[obj_idx].glob("*.jpg"))
                        if existing_frames:
                            ref_frame = cv2.imread(str(existing_frames[0]))
                            black_frame = np.zeros_like(ref_frame)
                        else:
                            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        mask_frame_path = mask_frame_dirs[obj_idx] / f"{frame_idx:06d}.jpg"
                        cv2.imwrite(str(mask_frame_path), black_frame)
                continue

            frame = cv2.imread(str(frame_path))
            overlay = frame.copy()

            # Get masks from output structure
            binary_masks = outputs.get("out_binary_masks")

            # If no masks for this frame, create black mask frames for all objects
            if binary_masks is None:
                for obj_idx in range(num_objects):
                    if obj_idx < len(mask_frame_dirs):
                        mask_frame = np.zeros_like(frame)
                        mask_frame_path = mask_frame_dirs[obj_idx] / f"{frame_idx:06d}.jpg"
                        cv2.imwrite(str(mask_frame_path), mask_frame)
                # Save composite frame (original frame, no overlay)
                composite_path = composite_dir / f"{frame_idx:06d}.jpg"
                cv2.imwrite(str(composite_path), frame)
                continue

            # Handle tensor on GPU if needed
            if hasattr(binary_masks, "cpu"):
                binary_masks = binary_masks.cpu().numpy()
            elif not isinstance(binary_masks, np.ndarray):
                binary_masks = np.array(binary_masks)

            # binary_masks shape is [num_objects, H, W]
            if binary_masks.ndim == 2:
                # Single mask, add object dimension
                binary_masks = binary_masks[np.newaxis, ...]
            elif binary_masks.ndim != 3:
                logger.warning(
                    f"Frame {frame_idx}: Unexpected masks shape {binary_masks.shape}. Creating black mask frames."
                )
                for obj_idx in range(num_objects):
                    if obj_idx < len(mask_frame_dirs):
                        mask_frame = np.zeros_like(frame)
                        mask_frame_path = mask_frame_dirs[obj_idx] / f"{frame_idx:06d}.jpg"
                        cv2.imwrite(str(mask_frame_path), mask_frame)
                composite_path = composite_dir / f"{frame_idx:06d}.jpg"
                cv2.imwrite(str(composite_path), frame)
                continue

            # Process each object's mask
            for obj_idx in range(num_objects):
                if obj_idx >= len(mask_frame_dirs):
                    break

                # Get mask for this object, or create black frame if not available
                if obj_idx < len(binary_masks):
                    mask = binary_masks[obj_idx]

                    # Validate mask dimensions
                    if mask.shape[0] == 0 or mask.shape[1] == 0:
                        logger.warning(
                            f"Frame {frame_idx}, obj {obj_idx}: Empty mask shape={mask.shape}. Creating black frame."
                        )
                        mask_frame = np.zeros_like(frame)
                    else:
                        # Resize mask to match frame if needed
                        if mask.shape[0] != frame.shape[0] or mask.shape[1] != frame.shape[1]:
                            mask = cv2.resize(mask.astype(np.float32), (frame.shape[1], frame.shape[0]))
                            mask = mask > 0.5

                        color = colors[obj_idx % len(colors)]

                        # Save individual mask frame (white mask on black background)
                        mask_frame = np.zeros_like(frame)
                        mask_frame[mask > 0] = (255, 255, 255)

                        # Apply to composite overlay
                        overlay[mask > 0] = color
                else:
                    # Object not found in this frame, create black mask
                    mask_frame = np.zeros_like(frame)

                # Always save mask frame for every object in every frame
                mask_frame_path = mask_frame_dirs[obj_idx] / f"{frame_idx:06d}.jpg"
                cv2.imwrite(str(mask_frame_path), mask_frame)

            # Blend overlay with original frame for composite
            composite_frame = cv2.addWeighted(frame, 1 - opacity, overlay, opacity, 0)

            # Save composite frame
            composite_path = composite_dir / f"{frame_idx:06d}.jpg"
            cv2.imwrite(str(composite_path), composite_frame)

        return mask_frame_dirs

    async def _encode_video(self, frames_dir: Path, output_path: Path, fps: float) -> None:
        """Encode frames back into a web-browser-compatible video using ffmpeg."""
        # Get list of frames to verify we have frames
        frame_files = sorted(frames_dir.glob("*.jpg"))
        if not frame_files:
            raise ValueError("No frames to encode")

        # Get ffmpeg path
        ffmpeg_path, _ = self._get_ffmpeg_paths()

        # Use ffmpeg to encode frames directly to H.264
        process = await asyncio.create_subprocess_exec(
            ffmpeg_path,
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "%06d.jpg"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(output_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        _, stderr = await process.communicate()

        if process.returncode != 0:
            error_message = stderr.decode() if stderr else "Unknown error"
            raise RuntimeError(f"ffmpeg encoding failed: {error_message}")

    def _get_ffmpeg_paths(self) -> tuple[str, str]:
        """Get FFmpeg and FFprobe executable paths."""
        import static_ffmpeg

        try:
            ffmpeg_path, ffprobe_path = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
            return ffmpeg_path, ffprobe_path  # noqa: TRY300
        except Exception as e:
            error_msg = f"FFmpeg not found. Please ensure static-ffmpeg is properly installed. Error: {e!s}"
            raise ValueError(error_msg) from e

    def _video_to_artifact(self, video_path: Path):
        """Convert video file to VideoUrlArtifact."""
        video_bytes = video_path.read_bytes()
        saved = self._output_file.build_file().write_bytes(video_bytes)
        return VideoUrlArtifact(value=saved.location)
