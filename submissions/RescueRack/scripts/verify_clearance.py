"""Check that RescueRack completes while staying clear of named obstacles."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rescuerack.controller import MissionStage, load_simulation  # noqa: E402


OBSTACLES = {
    "fallen_beam": np.array([-0.35, 0.18]),
    "pallet_block": np.array([-1.15, 0.78]),
    "loose_barrel": np.array([0.15, -0.45]),
}

# Conservative top-down radii for the visual obstacle footprints plus robot bumper.
OBSTACLE_RADII = {
    "fallen_beam": 1.02,
    "pallet_block": 0.62,
    "loose_barrel": 0.54,
}


def main() -> None:
    model, data, controller = load_simulation(mode="hard")
    min_clearance = {name: float("inf") for name in OBSTACLES}
    obstacle_contacts: list[str] = []

    for _ in range(int(40.0 / model.opt.timestep)):
        controller.step()
        mujoco.mj_step(model, data)

        base_xy = controller.base_xy()
        for name, center in OBSTACLES.items():
            clearance = float(np.linalg.norm(base_xy - center) - OBSTACLE_RADII[name])
            min_clearance[name] = min(min_clearance[name], clearance)

        for i in range(data.ncon):
            contact = data.contact[i]
            geom_a = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom_b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            pair = {geom_a, geom_b}
            if pair & set(OBSTACLES):
                obstacle_contacts.append(f"{geom_a}:{geom_b}")

        if controller.stage == MissionStage.COMPLETE and controller.stage_time > 0.75:
            break

    result = {
        "summary": controller.summary(),
        "min_clearance_m": {key: round(value, 3) for key, value in min_clearance.items()},
        "obstacle_contact_count": len(obstacle_contacts),
        "sample_contacts": obstacle_contacts[:5],
    }
    print(json.dumps(result, indent=2))

    if not result["summary"]["success"]:
        raise SystemExit("Clearance check failed: mission did not complete.")
    if obstacle_contacts:
        raise SystemExit("Clearance check failed: robot contacted a named obstacle.")


if __name__ == "__main__":
    main()
