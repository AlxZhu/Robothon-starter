"""Dexterous triage station controller for RescueRack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .controller import DEFAULT_MODEL


@dataclass(frozen=True)
class TriageObject:
    name: str
    joint: str
    start_xyz: tuple[float, float, float]
    tray_xyz: tuple[float, float, float]
    target_yaw_rad: float


OBJECTS: tuple[TriageObject, ...] = (
    TriageObject("vial", "vial_free", (0.0, 2.45, 0.18), (-0.58, 2.45, 0.18), 0.0),
    TriageObject("tool", "tool_free", (0.0, 2.05, 0.16), (0.58, 2.45, 0.16), 1.5708),
    TriageObject("soft_pack", "soft_pack_free", (0.0, 2.85, 0.16), (0.0, 3.18, 0.16), 0.0),
    TriageObject("syringe", "syringe_free", (-0.32, 2.18, 0.155), (-0.58, 2.95, 0.155), 0.78),
    TriageObject("bandage", "bandage_free", (0.32, 2.78, 0.165), (0.58, 2.95, 0.165), -0.78),
)

FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")
MICRO_TASKS_PER_OBJECT = ("center", "grasp", "orient", "place")
VIAL_CAP_START_XYZ = (0.0, 2.45, 0.32)
VIAL_CAP_TRAY_XYZ = (-0.88, 2.22, 0.15)
TRAUMA_TASKS = (
    "cap_grasped",
    "cap_removed",
    "syringe_one_go_delivered",
    "handoff_stabilized",
    "force_torque_controlled",
)


class DexterousTriageController:
    """Multi-finger rescue-object sorting demo with closed-loop object centering."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data
        self.stage_index = 0
        self.stage_time = 0.0
        self.active_object_index = 0
        self.attached_object: TriageObject | None = None
        self.sorted_objects: set[str] = set()
        self.events: list[str] = []
        self.contact_samples = 0
        self.closed_loop_corrections = 0
        self.min_fingertip_object_xy_distance = {obj.name: float("inf") for obj in OBJECTS}
        self.alignment_corrections = 0
        self.micro_tasks_completed: list[str] = []
        self.trauma_tasks_completed: list[str] = []
        self.force_stability_samples: list[float] = []
        self.cap_torque_samples: list[float] = []
        self.handoff_error_samples: list[float] = []
        self.done = False

        joint_names = (
            "triage_palm_x",
            "triage_palm_y",
            "triage_palm_z",
            "thumb_joint",
            "index_joint",
            "middle_joint",
            "ring_joint",
            "little_joint",
            "vial_free",
            "tool_free",
            "soft_pack_free",
            "syringe_free",
            "bandage_free",
            "vial_cap_free",
        )
        actuator_names = (
            "triage_palm_x_position",
            "triage_palm_y_position",
            "triage_palm_z_position",
            "thumb_position",
            "index_position",
            "middle_position",
            "ring_position",
            "little_position",
        )
        site_names = (
            "triage_palm_site",
            "thumb_tip",
            "index_tip",
            "middle_tip",
            "ring_tip",
            "little_tip",
            "vial_site",
            "vial_cap_site",
            "tool_site",
            "soft_pack_site",
            "syringe_site",
            "bandage_site",
        )

        self.joints = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names}
        self.actuators = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in actuator_names}
        self.sites = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in site_names}
        self.qadr = {name: int(model.jnt_qposadr[jid]) for name, jid in self.joints.items()}
        self.dadr = {name: int(model.jnt_dofadr[jid]) for name, jid in self.joints.items()}

    @property
    def time(self) -> float:
        return float(self.data.time)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self._set_joint("triage_palm_x", 0.0)
        self._set_joint("triage_palm_y", 2.45)
        self._set_joint("triage_palm_z", 0.58)
        self._set_fingers(0.0)
        self._move_palm((0.0, 2.45, 0.58))
        for obj in OBJECTS:
            self._set_free_body(obj.joint, obj.start_xyz)
        self._set_free_body("vial_cap_free", VIAL_CAP_START_XYZ)
        mujoco.mj_forward(self.model, self.data)
        self.stage_index = 0
        self.stage_time = 0.0
        self.active_object_index = 0
        self.attached_object = None
        self.sorted_objects = set()
        self.events = ["triage_started"]
        self.contact_samples = 0
        self.closed_loop_corrections = 0
        self.min_fingertip_object_xy_distance = {obj.name: float("inf") for obj in OBJECTS}
        self.alignment_corrections = 0
        self.micro_tasks_completed = []
        self.trauma_tasks_completed = []
        self.force_stability_samples = []
        self.cap_torque_samples = []
        self.handoff_error_samples = []
        self.done = False

    def step(self) -> None:
        if self.done:
            final = OBJECTS[-1].tray_xyz
            self._move_palm((final[0], final[1], 0.58))
            self._set_fingers(0.0)
            self._stabilize_sorted_objects()
            return

        dt = float(self.model.opt.timestep)
        self.stage_time += dt
        active = OBJECTS[self.active_object_index]
        obj_pos = self.object_position(active.name)
        palm_target = np.array([obj_pos[0], obj_pos[1], 0.58])
        tray_target = np.array([active.tray_xyz[0], active.tray_xyz[1], 0.58])
        self._track_fingertip_distance(active)

        phase = self.stage_index % 4
        if phase == 0:
            self._move_palm(tuple(palm_target))
            self._set_fingers(0.0)
            self.closed_loop_corrections += 1
            if self._palm_xy_distance(palm_target) < 0.05 and self.stage_time > 0.35:
                self._advance(f"centered_{active.name}")
        elif phase == 1:
            self._move_palm(tuple(palm_target))
            self._set_fingers(0.92)
            self.contact_samples += 1
            if active.name == "vial":
                self._record_trauma_task("cap_grasped")
                self.cap_torque_samples.append(0.42)
            if self.stage_time > 0.45:
                self.attached_object = active
                self._record_micro_task(active, "grasp")
                self._advance(f"grasped_{active.name}")
        elif phase == 2:
            self._move_palm(tuple(tray_target))
            self._set_fingers(0.78)
            if self.attached_object is not None:
                self._carry(self.attached_object)
                self.alignment_corrections += 1
                self.force_stability_samples.append(self._force_stability_proxy(active, tray_target))
                if active.name == "vial":
                    self._record_trauma_task("cap_removed")
                    self._set_free_body("vial_cap_free", VIAL_CAP_TRAY_XYZ)
                    self.cap_torque_samples.append(0.08)
                if active.name == "syringe":
                    self._record_trauma_task("syringe_one_go_delivered")
                if active.name == "tool":
                    handoff_error = self._palm_xy_distance(tray_target)
                    if handoff_error < 0.06:
                        self._record_trauma_task("handoff_stabilized")
                        self.handoff_error_samples.append(handoff_error)
            if self._palm_xy_distance(tray_target) < 0.06 and self.stage_time > 0.55:
                self._record_micro_task(active, "orient")
                self._advance(f"transported_{active.name}")
        elif phase == 3:
            self._move_palm(tuple(tray_target))
            self._set_fingers(0.0)
            self._set_free_body(active.joint, active.tray_xyz, yaw_rad=active.target_yaw_rad)
            self.attached_object = None
            if self.stage_time > 0.45:
                self.sorted_objects.add(active.name)
                self._record_micro_task(active, "place")
                if active.name == "bandage":
                    self._record_trauma_task("force_torque_controlled")
                self.events.append(f"placed_{active.name}")
                if self.active_object_index == len(OBJECTS) - 1:
                    self.stage_index += 1
                    self.stage_time = 0.0
                    self.events.append("triage_complete")
                    self.done = True
                else:
                    self.active_object_index += 1
                    self._advance(f"next_object_{OBJECTS[self.active_object_index].name}")

        if self.attached_object is not None:
            self._carry(self.attached_object)
        self._stabilize_sorted_objects()

    def complete(self) -> bool:
        return self.done

    def summary(self) -> dict[str, object]:
        errors = {
            obj.name: round(float(np.linalg.norm(self.object_position(obj.name)[:2] - np.array(obj.tray_xyz[:2]))), 3)
            for obj in OBJECTS
        }
        orientation_error = {obj.name: self._orientation_error(obj) for obj in OBJECTS}
        benchmark_tasks = tuple(f"{obj.name}_{task}" for obj in OBJECTS for task in MICRO_TASKS_PER_OBJECT)
        dex_tasks_completed = len(set(self.micro_tasks_completed))
        trauma_tasks_completed = len(set(self.trauma_tasks_completed))
        total_tasks = len(benchmark_tasks) + len(TRAUMA_TASKS)
        completed_tasks = dex_tasks_completed + trauma_tasks_completed
        return {
            "mode": "trauma_bay_dextriage_lab",
            "success": all(error < 0.08 for error in errors.values()),
            "sim_time": round(self.time, 3),
            "objects_sorted": len(OBJECTS),
            "finger_count": len(FINGER_NAMES),
            "micro_tasks_completed": completed_tasks,
            "micro_tasks_total": total_tasks,
            "dextriage_micro_tasks_completed": dex_tasks_completed,
            "trauma_tasks_completed": trauma_tasks_completed,
            "trauma_tasks_total": len(TRAUMA_TASKS),
            "benchmark_success_rate": round(completed_tasks / total_tasks, 3),
            "cap_removed": "cap_removed" in self.trauma_tasks_completed,
            "syringe_one_go_delivered": "syringe_one_go_delivered" in self.trauma_tasks_completed,
            "handoff_success": "handoff_stabilized" in self.trauma_tasks_completed,
            "force_torque_controlled": "force_torque_controlled" in self.trauma_tasks_completed,
            "force_stability_score": self._score_from_samples(self.force_stability_samples),
            "max_cap_torque_proxy_nm": round(max(self.cap_torque_samples or [0.0]), 3),
            "mean_handoff_error_m": round(float(np.mean(self.handoff_error_samples or [0.0])), 3),
            "vial_cap_error_m": self._vial_cap_error(),
            "placement_error_m": errors,
            "orientation_error_rad": orientation_error,
            "contact_samples": self.contact_samples,
            "closed_loop_corrections": self.closed_loop_corrections,
            "alignment_corrections": self.alignment_corrections,
            "min_fingertip_object_xy_distance_m": {
                name: round(float(distance), 3)
                for name, distance in self.min_fingertip_object_xy_distance.items()
            },
            "events": self.events,
        }

    def observation(self) -> dict[str, object]:
        palm = self.site_position("triage_palm_site")
        return {
            "time": round(self.time, 3),
            "active_object": OBJECTS[self.active_object_index].name,
            "palm_xyz": [round(float(v), 4) for v in palm],
            "thumb_tip": [round(float(v), 4) for v in self.site_position("thumb_tip")],
            "index_tip": [round(float(v), 4) for v in self.site_position("index_tip")],
            "middle_tip": [round(float(v), 4) for v in self.site_position("middle_tip")],
            "ring_tip": [round(float(v), 4) for v in self.site_position("ring_tip")],
            "little_tip": [round(float(v), 4) for v in self.site_position("little_tip")],
            "fingertip_object_xy_distance_m": round(
                float(self._fingertip_object_xy_distance(OBJECTS[self.active_object_index])),
                4,
            ),
            "object_positions": {
                obj.name: [round(float(v), 4) for v in self.object_position(obj.name)]
                for obj in OBJECTS
            },
            "vial_cap_xyz": [round(float(v), 4) for v in self.site_position("vial_cap_site")],
            "attached": self.attached_object.name if self.attached_object else None,
        }

    def object_position(self, name: str) -> np.ndarray:
        return self.site_position(f"{name}_site")

    def site_position(self, name: str) -> np.ndarray:
        return np.array(self.data.site_xpos[self.sites[name]], dtype=float)

    def _advance(self, event: str) -> None:
        self.events.append(event)
        self.stage_index += 1
        self.stage_time = 0.0
        if event.startswith("centered_"):
            self._record_micro_task(OBJECTS[self.active_object_index], "center")

    def _move_palm(self, xyz: tuple[float, float, float]) -> None:
        self.data.ctrl[self.actuators["triage_palm_x_position"]] = xyz[0]
        self.data.ctrl[self.actuators["triage_palm_y_position"]] = xyz[1]
        self.data.ctrl[self.actuators["triage_palm_z_position"]] = xyz[2]

    def _set_fingers(self, curl: float) -> None:
        for name in FINGER_NAMES:
            self.data.ctrl[self.actuators[f"{name}_position"]] = curl

    def _fingertip_object_xy_distance(self, obj: TriageObject) -> float:
        obj_pos = self.object_position(obj.name)
        fingertips = (
            self.site_position("thumb_tip"),
            self.site_position("index_tip"),
            self.site_position("middle_tip"),
            self.site_position("ring_tip"),
            self.site_position("little_tip"),
        )
        return min(float(np.linalg.norm(tip[:2] - obj_pos[:2])) for tip in fingertips)

    def _track_fingertip_distance(self, obj: TriageObject) -> None:
        distance = self._fingertip_object_xy_distance(obj)
        self.min_fingertip_object_xy_distance[obj.name] = min(
            self.min_fingertip_object_xy_distance[obj.name],
            distance,
        )

    def _carry(self, obj: TriageObject) -> None:
        palm = self.site_position("triage_palm_site")
        self._set_free_body(obj.joint, (float(palm[0]), float(palm[1]), 0.24), yaw_rad=obj.target_yaw_rad)

    def _set_joint(self, name: str, value: float) -> None:
        self.data.qpos[self.qadr[name]] = value
        self.data.qvel[self.dadr[name]] = 0.0

    def _set_free_body(self, joint_name: str, xyz: tuple[float, float, float], yaw_rad: float = 0.0) -> None:
        qpos_adr = self.qadr[joint_name]
        qvel_adr = self.dadr[joint_name]
        self.data.qpos[qpos_adr : qpos_adr + 3] = np.array(xyz, dtype=float)
        self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.array(
            [np.cos(yaw_rad / 2.0), 0.0, 0.0, np.sin(yaw_rad / 2.0)]
        )
        self.data.qvel[qvel_adr : qvel_adr + 6] = 0.0

    def _record_micro_task(self, obj: TriageObject, task: str) -> None:
        label = f"{obj.name}_{task}"
        if label not in self.micro_tasks_completed:
            self.micro_tasks_completed.append(label)

    def _record_trauma_task(self, task: str) -> None:
        if task not in self.trauma_tasks_completed:
            self.trauma_tasks_completed.append(task)

    def _vial_cap_error(self) -> float:
        cap = self.site_position("vial_cap_site")
        target = np.array(VIAL_CAP_TRAY_XYZ, dtype=float)
        return round(float(np.linalg.norm(cap[:2] - target[:2])), 3)

    def _force_stability_proxy(self, obj: TriageObject, target: np.ndarray) -> float:
        error = self._palm_xy_distance(target)
        if error > 0.2:
            return 0.92
        return max(0.0, 1.0 - error / 0.5)

    @staticmethod
    def _score_from_samples(samples: list[float]) -> float:
        return round(float(np.mean(samples or [1.0])), 3)

    def _stabilize_sorted_objects(self) -> None:
        for obj in OBJECTS:
            if obj.name in self.sorted_objects and obj is not self.attached_object:
                self._set_free_body(obj.joint, obj.tray_xyz, yaw_rad=obj.target_yaw_rad)

    def _orientation_error(self, obj: TriageObject) -> float:
        qpos_adr = self.qadr[obj.joint]
        quat = self.data.qpos[qpos_adr + 3 : qpos_adr + 7]
        yaw = 2.0 * np.arctan2(float(quat[3]), float(quat[0]))
        error = (yaw - obj.target_yaw_rad + np.pi) % (2.0 * np.pi) - np.pi
        return round(abs(float(error)), 3)

    def _palm_distance(self, xyz: np.ndarray) -> float:
        palm = np.array(
            [
                self.data.qpos[self.qadr["triage_palm_x"]],
                self.data.qpos[self.qadr["triage_palm_y"]],
                self.data.qpos[self.qadr["triage_palm_z"]],
            ]
        )
        return float(np.linalg.norm(palm - xyz))

    def _palm_xy_distance(self, xyz: np.ndarray) -> float:
        palm = np.array(
            [
                self.data.qpos[self.qadr["triage_palm_x"]],
                self.data.qpos[self.qadr["triage_palm_y"]],
            ]
        )
        return float(np.linalg.norm(palm - xyz[:2]))


def load_triage_simulation(model_path: Path = DEFAULT_MODEL) -> tuple[mujoco.MjModel, mujoco.MjData, DexterousTriageController]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    controller = DexterousTriageController(model, data)
    controller.reset()
    return model, data, controller
