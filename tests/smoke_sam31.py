"""Explicit native GPU smoke check: one selected frame, never clip tracking.

Run this with the dedicated SAM Python. Without --checkpoint, invoking this
script permits the native predictor to fetch the gated official SAM 3.1 weights.
The default clicks target the right lion in the supplied two-lion example; use
--include/--exclude normalized coordinates for another source.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import re
from pathlib import Path


def _coordinate(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("Coordinates must be between 0 and 1.")
    return number


def _safe_error(exc: BaseException) -> str:
    message = " ".join(str(exc).split())
    message = re.sub(r"hf_[A-Za-z0-9_]+", "[redacted]", message)
    message = re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?[redacted]", message)
    return f"{type(exc).__name__}: {message[:900]}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Original image or video file")
    parser.add_argument("--output", required=True, type=Path, help="Output root; a new unique run directory is created")
    parser.add_argument(
        "--checkpoint", type=Path, help="Optional official sam3.1_multiplex.pt; omitted means HF cache/download"
    )
    parser.add_argument("--frame", type=int, default=0, help="Zero-based source frame; default 0")
    parser.add_argument(
        "--include",
        type=_coordinate,
        nargs=2,
        action="append",
        metavar=("X", "Y"),
        help="Normalized foreground click; repeat for more clicks. Default .65 .50",
    )
    parser.add_argument(
        "--exclude",
        type=_coordinate,
        nargs=2,
        action="append",
        metavar=("X", "Y"),
        help="Normalized background click; repeat for more clicks. Default .25 .45",
    )
    parser.add_argument("--no-default-exclude", action="store_true", help="Omit the default left-lion exclude click")
    args = parser.parse_args(argv)
    if args.frame < 0:
        parser.error("--frame must be nonnegative")
    includes = args.include or [[0.65, 0.50]]
    excludes = args.exclude if args.exclude is not None else ([] if args.no_default_exclude else [[0.25, 0.45]])
    points = [
        {"frame": args.frame, "x": x, "y": y, "label": label, "objectId": 1}
        for label, coordinates in ((1, includes), (0, excludes))
        for x, y in coordinates
    ]
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    backend_file = (
        Path(__file__).resolve().parents[1] / "griptape_nodes_sam3_library" / "selective_matte" / "backend.py"
    )
    spec = importlib.util.spec_from_file_location("sam31_native_smoke_backend", backend_file)
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    captured = io.StringIO()
    result = None
    error = None
    try:
        # Upstream imports/weight loads can be verbose. Keep credentials and
        # signed download URLs out of console output; report only our contract.
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            result = backend.segment(
                args.source,
                points,
                checkpoint_path=str(args.checkpoint) if args.checkpoint else None,
                mode="preview",
                selected_frame=args.frame,
                output_dir=args.output,
                make_preview_video=False,
            )
    except KeyboardInterrupt:
        error = "Interrupted: preview cancelled"
    except Exception as exc:
        error = _safe_error(exc)
    finally:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            try:
                backend.unload_model()
            except Exception as exc:
                if error is None:
                    error = "Cleanup failed: " + _safe_error(exc)
        captured.close()
    if error:
        print(json.dumps({"status": "failed", "error": error}), flush=True)
        return 1
    try:
        from PIL import Image

        with Image.open(result["preview_mask_path"]) as mask:
            if mask.size != (result["width"], result["height"]):
                raise RuntimeError("The preview mask dimensions do not match the original source.")
            histogram = mask.convert("L").histogram()
            foreground_pixels = sum(histogram[1:])
        if result["mode"] != "preview" or result["frame_indices"] != [args.frame]:
            raise RuntimeError("Smoke check produced more than the requested frame.")
        if foreground_pixels == 0:
            raise RuntimeError("SAM returned an empty mask; adjust the include/exclude clicks.")
        print(
            json.dumps(
                {
                    "status": "passed",
                    "model": result["model_info"],
                    "width": result["width"],
                    "height": result["height"],
                    "frame": args.frame,
                    "frames_segmented": 1,
                    "foreground_pixels": foreground_pixels,
                    "preview_mask_path": result["preview_mask_path"],
                    "source_preview_path": result["source_preview_path"],
                    "manifest_path": result["manifest_path"],
                },
                indent=2,
            ),
            flush=True,
        )
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": _safe_error(exc)}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
