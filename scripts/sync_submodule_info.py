"""Sync metadata.submodule_info in the library JSON from the SAM3 git submodule.

The library falls back to submodule_info when it is installed without git metadata,
so it must match the submodule URL in .gitmodules and the commit pinned in the git index.

Usage:
    python scripts/sync_submodule_info.py          # rewrite the JSON
    python scripts/sync_submodule_info.py --check  # exit 1 if the JSON is out of sync
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LIBRARY_JSON = REPO_ROOT / "griptape_nodes_sam3_library" / "griptape-nodes-library.json"
SUBMODULE_PATH = "griptape_nodes_sam3_library/_sam3_repo"


def _git(*args: str) -> str:
    result = subprocess.run(["git", "-C", str(REPO_ROOT), *args], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _submodule_url() -> str:
    for line in _git("config", "-f", ".gitmodules", "--get-regexp", r"^submodule\..*\.path$").splitlines():
        key, path = line.split(" ", 1)
        if path == SUBMODULE_PATH:
            name = key.removeprefix("submodule.").removesuffix(".path")
            return _git("config", "-f", ".gitmodules", f"submodule.{name}.url")
    msg = f"No submodule with path {SUBMODULE_PATH} in .gitmodules"
    raise SystemExit(msg)


def _submodule_commit() -> str:
    # Read the gitlink from the index so this works without the submodule checked out.
    # Output format: "<mode> <sha> <stage>\t<path>"
    entry = _git("ls-files", "--stage", "--", SUBMODULE_PATH)
    if not entry.startswith("160000 "):
        msg = f"{SUBMODULE_PATH} is not a submodule in the git index"
        raise SystemExit(msg)
    return entry.split()[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Exit 1 if the JSON is out of sync instead of writing it.")
    args = parser.parse_args()

    expected = {"url": _submodule_url(), "commit": _submodule_commit()}
    data = json.loads(LIBRARY_JSON.read_text())
    current = data["metadata"].get("submodule_info", {})

    if current == expected:
        print(f"submodule_info is in sync: {expected}")
        return 0

    if args.check:
        print(f"submodule_info is out of sync: JSON has {current}, git has {expected}.", file=sys.stderr)
        print("Run `make submodule/sync` and commit the result.", file=sys.stderr)
        return 1

    data["metadata"]["submodule_info"] = expected
    LIBRARY_JSON.write_text(json.dumps(data, indent=2) + "\n", newline="\n")
    print(f"Updated submodule_info: {expected}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
