"""Quick reproducibility check for judges and local development."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rescuerack.run_demo import run_headless  # noqa: E402


class Args:
    model = ROOT / "models" / "rescuerack_scene.xml"
    mode = "easy"
    duration = 28.0
    viewer = False
    record = None
    camera = "overview"
    width = 640
    height = 360
    fps = 20


def main() -> None:
    summary = run_headless(Args())
    print(json.dumps(summary, indent=2))
    if not summary["success"]:
        raise SystemExit("Smoke test failed: RescueRack did not complete the easy mission.")


if __name__ == "__main__":
    main()
