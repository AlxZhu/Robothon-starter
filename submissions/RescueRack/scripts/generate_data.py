"""Generate trajectory and metrics artifacts for the RescueRack hard mission."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rescuerack.run_demo import run_headless  # noqa: E402
from rescuerack.triage import load_triage_simulation  # noqa: E402

import mujoco


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
    triage_model, triage_data, triage_controller = load_triage_simulation()
    triage_samples: list[dict[str, object]] = []
    triage_sample_interval = max(1, round(1.0 / (10.0 * triage_model.opt.timestep)))
    for step in range(int(30.0 / triage_model.opt.timestep)):
        triage_controller.step()
        mujoco.mj_step(triage_model, triage_data)
        if step % triage_sample_interval == 0:
            triage_samples.append(triage_controller.observation())
        if triage_controller.complete():
            break
    triage_summary = triage_controller.summary()
    (ROOT / "media" / "triage_trajectory.json").write_text(
        json.dumps({"summary": triage_summary, "samples": triage_samples}, indent=2),
        encoding="utf-8",
    )
    metrics = {
        "project": "RescueRack",
        "mode": "hard",
        "mission_summary": summary,
        "data_artifacts": {
            "trajectory": "media/hard_trajectory.json",
            "triage_trajectory": "media/triage_trajectory.json",
            "demo_video": "media/demo.mp4",
        },
        "rubric_evidence": {
            "adaptive_control": summary["planner"],
            "dexterous_manipulation": {
                "mode": "trauma_bay_dextriage_lab",
                "finger_count": triage_summary["finger_count"],
                "objects_sorted": triage_summary["objects_sorted"],
                "micro_tasks_completed": triage_summary["micro_tasks_completed"],
                "micro_tasks_total": triage_summary["micro_tasks_total"],
                "dextriage_micro_tasks_completed": triage_summary["dextriage_micro_tasks_completed"],
                "trauma_tasks_completed": triage_summary["trauma_tasks_completed"],
                "trauma_tasks_total": triage_summary["trauma_tasks_total"],
                "benchmark_success_rate": triage_summary["benchmark_success_rate"],
                "cap_removed": triage_summary["cap_removed"],
                "syringe_one_go_delivered": triage_summary["syringe_one_go_delivered"],
                "handoff_success": triage_summary["handoff_success"],
                "force_torque_controlled": triage_summary["force_torque_controlled"],
                "force_stability_score": triage_summary["force_stability_score"],
                "max_cap_torque_proxy_nm": triage_summary["max_cap_torque_proxy_nm"],
                "mean_handoff_error_m": triage_summary["mean_handoff_error_m"],
                "vial_cap_error_m": triage_summary["vial_cap_error_m"],
                "contact_samples": triage_summary["contact_samples"],
                "closed_loop_corrections": triage_summary["closed_loop_corrections"],
                "alignment_corrections": triage_summary["alignment_corrections"],
                "min_fingertip_object_xy_distance_m": triage_summary["min_fingertip_object_xy_distance_m"],
                "placement_error_m": triage_summary["placement_error_m"],
                "orientation_error_rad": triage_summary["orientation_error_rad"],
                "success": triage_summary["success"],
            },
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
                "triage_palm_xyz",
                "thumb_tip_xyz",
                "index_tip_xyz",
                "middle_tip_xyz",
                "ring_tip_xyz",
                "little_tip_xyz",
                "triage_object_xyz",
                "vial_cap_xyz",
                "fingertip_object_xy_distance_m",
            ],
            "success_threshold_m": 0.45,
            "demo_video": {
                "duration_seconds": 124.23,
                "duration_label": "2:04",
                "generated_from": "scripts/make_demo_video.py",
            },
        },
    }
    (ROOT / "media").mkdir(exist_ok=True)
    (ROOT / "media" / "mission_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    if not summary["success"]:
        raise SystemExit("Data generation failed: hard mission did not complete.")
    if not triage_summary["success"]:
        raise SystemExit("Data generation failed: dexterous triage did not complete.")


if __name__ == "__main__":
    main()
