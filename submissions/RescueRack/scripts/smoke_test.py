"""Quick reproducibility check for judges and local development."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rescuerack.run_demo import run_headless  # noqa: E402
from rescuerack.triage import load_triage_simulation  # noqa: E402


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
    trajectory = None
    sample_hz = 10.0


def main() -> None:
    mission_summary = run_headless(Args())
    triage_model, triage_data, triage_controller = load_triage_simulation()
    for _ in range(int(28.0 / triage_model.opt.timestep)):
        triage_controller.step()
        mujoco.mj_step(triage_model, triage_data)
        if triage_controller.complete():
            break
    triage_summary = triage_controller.summary()
    summary = {
        "mission": mission_summary,
        "trauma_bay_dextriage": triage_summary,
    }
    print(json.dumps(summary, indent=2))
    if not mission_summary["success"]:
        raise SystemExit("Smoke test failed: RescueRack did not complete the easy mission.")
    if triage_summary["micro_tasks_completed"] != triage_summary["micro_tasks_total"]:
        raise SystemExit("Smoke test failed: Trauma Bay DexTriage did not complete all micro-tasks.")
    if not triage_summary["success"]:
        raise SystemExit("Smoke test failed: Trauma Bay DexTriage placement validation failed.")


if __name__ == "__main__":
    main()
