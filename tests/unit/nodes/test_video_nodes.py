from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from griptape_nodes.exe_types.core_types import NodeError

from griptape_nodes_sam3_library.sam3_multiplex_video import Sam3MultiplexVideo
from griptape_nodes_sam3_library.sam3_segment_video import Sam3SegmentVideo


class FakePredictor:
    """Stands in for a SAM3 predictor whose prompts match no object."""

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        if request["type"] == "start_session":
            return {"session_id": "session"}
        if request["type"] == "add_prompt":
            return {"outputs": {"out_obj_ids": []}}
        return {}

    def handle_stream_request(self, request: dict[str, Any]) -> Iterator[dict[str, Any]]:
        for frame_index in range(3):
            yield {"frame_index": frame_index, "outputs": {"out_binary_masks": np.zeros((0, 4, 4), dtype=bool)}}

    def shutdown(self) -> None:
        pass


def make_node(
    monkeypatch: pytest.MonkeyPatch, node_class: type[Sam3MultiplexVideo | Sam3SegmentVideo], frame_count: int
) -> Sam3MultiplexVideo | Sam3SegmentVideo:
    node = node_class(name="video")
    prompt_parameter = "text_prompts" if node_class is Sam3MultiplexVideo else "text_prompt"
    node.set_parameter_value("input_video", {"value": "video.mp4"})
    node.set_parameter_value(prompt_parameter, "unicorn")
    node.set_parameter_value("prompt_frame", 1)

    def load_model(*_args: Any) -> None:
        node._predictor = FakePredictor()

    def extract_frames(_video_artifact: Any, _output_dir: Path) -> tuple[Path, float, int]:
        return Path("input_video.mp4"), 24.0, frame_count

    monkeypatch.setattr(node, "_load_model", load_model)
    monkeypatch.setattr(node, "_extract_frames", extract_frames)
    return node


@pytest.mark.asyncio
@pytest.mark.parametrize("node_class", [Sam3MultiplexVideo, Sam3SegmentVideo])
async def test_prompt_matching_nothing_names_the_prompt_and_frame(
    monkeypatch: pytest.MonkeyPatch, node_class: type[Sam3MultiplexVideo | Sam3SegmentVideo]
) -> None:
    node = make_node(monkeypatch, node_class, frame_count=3)

    with pytest.raises(NodeError, match="No objects matched 'unicorn' on frame 1") as excinfo:
        await node.aprocess()

    assert excinfo.value.fields["prompt_frame"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("node_class", [Sam3MultiplexVideo, Sam3SegmentVideo])
async def test_video_with_no_readable_frames_says_so(
    monkeypatch: pytest.MonkeyPatch, node_class: type[Sam3MultiplexVideo | Sam3SegmentVideo]
) -> None:
    node = make_node(monkeypatch, node_class, frame_count=0)

    with pytest.raises(NodeError, match="No frames could be read from the input video"):
        await node.aprocess()
