"""Generate trajectory and metrics artifacts for the RescueRack hard mission."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rescuerack.run_demo import run_headless  # noqa: E402


class Args:
    model = ROOT / "models" / "rescuerack_scene.xml"
    mode = "hard"
    duration = 40.0
    viewer = False
    record = None
    camera = "overview"
    width = 640
    height = 368
    fps = 30
    trajectory = ROOT / "media" / "hard_trajectory.json"
    sample_hz = 10.0


def main() -> None:
    summary = run_headless(Args())
    metrics = {
        "project": "RescueRack",
        "mode": "hard",
        "mission_summary": summary,
        "data_artifacts": {
            "trajectory": "media/hard_trajectory.json",
            "demo_video": "media/demo.mp4",
        },
        "rubric_evidence": {
            "long_horizon_stages": [
                "navigate_to_supply",
                "prepare_grasp",
                "secure_supply",
                "transport_to_safe_zone",
                "release_supply",
                "complete",
            ],
            "sensors_logged": [
                "base_xy",
                "base_yaw",
                "gripper_xyz",
                "medkit_xyz",
                "safe_zone_xyz",
                "distance_medkit_to_safe_xy",
            ],
            "success_threshold_m": 0.45,
        },
    }
    (ROOT / "media").mkdir(exist_ok=True)
    (ROOT / "media" / "mission_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    if not summary["success"]:
        raise SystemExit("Data generation failed: hard mission did not complete.")


if __name__ == "__main__":
    main()
