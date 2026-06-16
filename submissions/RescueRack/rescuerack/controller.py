"""Mission controller for the RescueRack MuJoCo demo."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "rescuerack_scene.xml"


class MissionStage(str, Enum):
    NAVIGATE_TO_SUPPLY = "navigate_to_supply"
    PREPARE_GRASP = "prepare_grasp"
    SECURE_SUPPLY = "secure_supply"
    TRANSPORT_TO_SAFE_ZONE = "transport_to_safe_zone"
    RELEASE_SUPPLY = "release_supply"
    COMPLETE = "complete"


@dataclass(frozen=True)
class MissionConfig:
    mode: str
    target_body: str
    target_site: str
    supply_xy: tuple[float, float]
    safe_xy: tuple[float, float]
    pickup_path: tuple[tuple[float, float], ...]
    delivery_path: tuple[tuple[float, float], ...]
    description: str


MISSION_CONFIGS: dict[str, MissionConfig] = {
    "easy": MissionConfig(
        mode="easy",
        target_body="medkit",
        target_site="medkit_site",
        supply_xy=(1.42, -1.15),
        safe_xy=(-2.65, 1.35),
        pickup_path=((-1.7, -1.65), (-0.35, -1.72), (0.55, -1.15)),
        delivery_path=((0.3, -1.65), (-1.15, -1.65), (-2.55, -0.25), (-2.65, 1.35)),
        description="Retrieve the red emergency medkit and deliver it to the green safe zone.",
    ),
    "medium": MissionConfig(
        mode="medium",
        target_body="medkit",
        target_site="medkit_site",
        supply_xy=(1.42, -1.15),
        safe_xy=(-2.65, 1.35),
        pickup_path=((-2.0, -1.68), (-0.45, -1.78), (0.62, -1.15)),
        delivery_path=((0.35, -1.7), (-1.15, -1.72), (-2.45, -0.18), (-2.65, 1.35)),
        description="Navigate around debris, retrieve the medkit, and deliver it to safety.",
    ),
    "hard": MissionConfig(
        mode="hard",
        target_body="medkit",
        target_site="medkit_site",
        supply_xy=(1.42, -1.15),
        safe_xy=(-2.65, 1.35),
        pickup_path=((-2.25, -1.72), (-0.7, -1.84), (0.58, -1.22)),
        delivery_path=((0.28, -1.74), (-1.3, -1.76), (-2.48, -0.2), (-2.65, 1.35)),
        description=(
            "Select the red medkit from distractor supplies, avoid blocked aisles, "
            "and deliver only the target item."
        ),
    ),
}


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def yaw_to_face(origin: np.ndarray, target: np.ndarray) -> float:
    delta = target[:2] - origin[:2]
    if np.linalg.norm(delta) < 1e-6:
        return 0.0
    return math.atan2(float(delta[1]), float(delta[0]))


class RescueRackController:
    """Deterministic high-level policy for a robust rescue-supply demo."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, mode: str = "easy") -> None:
        if mode not in MISSION_CONFIGS:
            allowed = ", ".join(sorted(MISSION_CONFIGS))
            raise ValueError(f"Unknown mode '{mode}'. Expected one of: {allowed}")

        self.model = model
        self.data = data
        self.config = MISSION_CONFIGS[mode]
        self.stage = MissionStage.NAVIGATE_TO_SUPPLY
        self.path_index = 0
        self.stage_time = 0.0
        self.attached = False
        self.release_started_at: float | None = None
        self.secured_logged = False
        self.events: list[str] = []

        self.joints = {
            name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in (
                "base_x",
                "base_y",
                "base_yaw",
                "shoulder",
                "elbow",
                "wrist",
                "left_finger_joint",
                "right_finger_joint",
                "medkit_free",
                "water_free",
                "battery_free",
            )
        }
        self.actuators = {
            name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in (
                "base_x_position",
                "base_y_position",
                "base_yaw_position",
                "shoulder_position",
                "elbow_position",
                "wrist_position",
                "left_gripper_position",
                "right_gripper_position",
            )
        }
        self.sites = {
            name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in ("gripper_site", "medkit_site", "water_site", "battery_site", "safe_zone_site")
        }

        missing = [
            key
            for group in (self.joints, self.actuators, self.sites)
            for key, value in group.items()
            if value < 0
        ]
        if missing:
            raise RuntimeError(f"Model is missing named elements: {', '.join(missing)}")

        self.qadr = {name: int(model.jnt_qposadr[jid]) for name, jid in self.joints.items()}
        self.dadr = {name: int(model.jnt_dofadr[jid]) for name, jid in self.joints.items()}

    @property
    def time(self) -> float:
        return float(self.data.time)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)

        self._set_joint("base_x", -2.65)
        self._set_joint("base_y", -1.45)
        self._set_joint("base_yaw", 0.0)
        self._set_arm(0.12, -0.22, 0.1)
        self._set_gripper(opened=True)

        self._set_free_body("medkit_free", (1.42, -1.15, 0.12))
        self._set_free_body("water_free", (1.8, 0.55, 0.12))
        self._set_free_body("battery_free", (2.25, -0.35, 0.12))

        self._drive_to((-2.65, -1.45), yaw=0.0)
        self._set_joint("shoulder", 0.12)
        self._set_joint("elbow", -0.22)
        self._set_joint("wrist", 0.1)
        self._set_joint("left_finger_joint", 0.052)
        self._set_joint("right_finger_joint", 0.052)
        self._set_arm(0.12, -0.22, 0.1)
        self._set_gripper(opened=True)

        mujoco.mj_forward(self.model, self.data)
        self.stage = MissionStage.NAVIGATE_TO_SUPPLY
        self.path_index = 0
        self.stage_time = 0.0
        self.attached = False
        self.release_started_at = None
        self.secured_logged = False
        self.events = ["mission_started"]

    def step(self) -> None:
        dt = float(self.model.opt.timestep)
        self.stage_time += dt

        if self.stage == MissionStage.NAVIGATE_TO_SUPPLY:
            self._follow_waypoints(self.config.pickup_path, MissionStage.PREPARE_GRASP)
            self._set_arm(0.04, -0.11, 0.06)
            self._set_gripper(opened=True)

        elif self.stage == MissionStage.PREPARE_GRASP:
            pickup_pose = np.array([self.config.supply_xy[0] - 0.86, self.config.supply_xy[1], 0.0])
            self._drive_to(pickup_pose[:2], yaw=0.0)
            self._set_arm(-0.08, -0.04, 0.08)
            self._set_gripper(opened=True)
            if self._base_distance(pickup_pose[:2]) < 0.05 and self.stage_time > 0.55:
                self._advance(MissionStage.SECURE_SUPPLY)

        elif self.stage == MissionStage.SECURE_SUPPLY:
            pickup_pose = np.array([self.config.supply_xy[0] - 0.86, self.config.supply_xy[1], 0.0])
            self._drive_to(pickup_pose[:2], yaw=0.0)
            self._set_arm(-0.08, -0.04, 0.08)
            self._set_gripper(opened=False)
            if self.stage_time > 0.45:
                self.attached = True
                if not self.secured_logged:
                    self.events.append("medkit_secured")
                    self.secured_logged = True
            if self.stage_time > 0.9:
                self.path_index = 0
                self._advance(MissionStage.TRANSPORT_TO_SAFE_ZONE)

        elif self.stage == MissionStage.TRANSPORT_TO_SAFE_ZONE:
            self._set_arm(0.1, -0.2, 0.12)
            self._set_gripper(opened=False)
            self._follow_waypoints(self.config.delivery_path, MissionStage.RELEASE_SUPPLY)

        elif self.stage == MissionStage.RELEASE_SUPPLY:
            drop_pose = np.array([self.config.safe_xy[0] - 0.74, self.config.safe_xy[1], 0.0])
            self._drive_to(drop_pose[:2], yaw=0.0)
            self._set_arm(-0.05, -0.08, 0.09)
            if self.release_started_at is None:
                self.release_started_at = self.time
                self.events.append("entered_safe_zone")
            if self.time - self.release_started_at > 0.35:
                self.attached = False
                self._set_gripper(opened=True)
                self._set_free_body("medkit_free", (self.config.safe_xy[0], self.config.safe_xy[1], 0.12))
            if self.time - self.release_started_at > 1.0:
                self._advance(MissionStage.COMPLETE)
                self.events.append("mission_complete")

        elif self.stage == MissionStage.COMPLETE:
            self._drive_to((self.config.safe_xy[0] - 0.74, self.config.safe_xy[1]), yaw=0.0)
            self._set_arm(0.15, -0.25, 0.08)
            self._set_gripper(opened=True)

        if self.attached:
            self._carry_target_at_gripper()

    def summary(self) -> dict[str, object]:
        medkit_pos = self.site_position("medkit_site")
        safe_pos = np.array([self.config.safe_xy[0], self.config.safe_xy[1], 0.12])
        delivery_error = float(np.linalg.norm(medkit_pos[:2] - safe_pos[:2]))
        return {
            "mode": self.config.mode,
            "stage": self.stage.value,
            "success": self.stage == MissionStage.COMPLETE and delivery_error < 0.45,
            "sim_time": round(self.time, 3),
            "delivery_error_m": round(delivery_error, 3),
            "target": self.config.target_body,
            "events": self.events,
        }

    def observation(self) -> dict[str, object]:
        medkit_pos = self.site_position("medkit_site")
        safe_pos = self.site_position("safe_zone_site")
        gripper_pos = self.site_position("gripper_site")
        base_xy = self.base_xy()
        return {
            "time": round(self.time, 3),
            "stage": self.stage.value,
            "base_xy": [round(float(base_xy[0]), 4), round(float(base_xy[1]), 4)],
            "base_yaw": round(float(self.data.qpos[self.qadr["base_yaw"]]), 4),
            "gripper_xyz": [round(float(value), 4) for value in gripper_pos],
            "medkit_xyz": [round(float(value), 4) for value in medkit_pos],
            "safe_zone_xyz": [round(float(value), 4) for value in safe_pos],
            "distance_medkit_to_safe_xy": round(float(np.linalg.norm(medkit_pos[:2] - safe_pos[:2])), 4),
            "attached": self.attached,
        }

    def site_position(self, name: str) -> np.ndarray:
        return np.array(self.data.site_xpos[self.sites[name]], dtype=float)

    def _advance(self, next_stage: MissionStage) -> None:
        self.events.append(f"{self.stage.value}_done")
        self.stage = next_stage
        self.stage_time = 0.0

    def _follow_waypoints(
        self,
        waypoints: Iterable[tuple[float, float]],
        next_stage: MissionStage,
    ) -> None:
        points = tuple(waypoints)
        target = np.array(points[self.path_index], dtype=float)
        base = self.base_xy()
        yaw = yaw_to_face(np.array([base[0], base[1], 0.0]), np.array([target[0], target[1], 0.0]))
        self._drive_to(target, yaw)

        if self._base_distance(target) < 0.08:
            self.path_index += 1
            if self.path_index >= len(points):
                self.path_index = 0
                self._advance(next_stage)

    def _drive_to(self, xy: np.ndarray | tuple[float, float], yaw: float) -> None:
        x, y = float(xy[0]), float(xy[1])
        self.data.ctrl[self.actuators["base_x_position"]] = x
        self.data.ctrl[self.actuators["base_y_position"]] = y
        self.data.ctrl[self.actuators["base_yaw_position"]] = wrap_angle(yaw)

    def _set_arm(self, shoulder: float, elbow: float, wrist: float) -> None:
        self.data.ctrl[self.actuators["shoulder_position"]] = shoulder
        self.data.ctrl[self.actuators["elbow_position"]] = elbow
        self.data.ctrl[self.actuators["wrist_position"]] = wrist

    def _set_gripper(self, opened: bool) -> None:
        target = 0.052 if opened else 0.006
        self.data.ctrl[self.actuators["left_gripper_position"]] = target
        self.data.ctrl[self.actuators["right_gripper_position"]] = target

    def _carry_target_at_gripper(self) -> None:
        gripper = self.site_position("gripper_site")
        carried_xyz = (float(gripper[0] + 0.08), float(gripper[1]), max(0.34, float(gripper[2] + 0.03)))
        self._set_free_body("medkit_free", carried_xyz)

    def _set_joint(self, name: str, value: float) -> None:
        self.data.qpos[self.qadr[name]] = value
        self.data.qvel[self.dadr[name]] = 0.0

    def _set_free_body(self, joint_name: str, xyz: tuple[float, float, float]) -> None:
        qpos_adr = self.qadr[joint_name]
        qvel_adr = self.dadr[joint_name]
        self.data.qpos[qpos_adr : qpos_adr + 3] = np.array(xyz, dtype=float)
        self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qvel[qvel_adr : qvel_adr + 6] = 0.0

    def base_xy(self) -> np.ndarray:
        return np.array(
            [
                self.data.qpos[self.qadr["base_x"]],
                self.data.qpos[self.qadr["base_y"]],
            ],
            dtype=float,
        )

    def _base_distance(self, xy: np.ndarray | tuple[float, float]) -> float:
        return float(np.linalg.norm(self.base_xy() - np.array(xy, dtype=float)))


def load_simulation(model_path: Path = DEFAULT_MODEL, mode: str = "easy") -> tuple[mujoco.MjModel, mujoco.MjData, RescueRackController]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    controller = RescueRackController(model, data, mode=mode)
    controller.reset()
    return model, data, controller
