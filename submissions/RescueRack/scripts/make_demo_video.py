"""Render the RescueRack submission demo video from MuJoCo simulation frames."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import mujoco

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from rescuerack.controller import MissionStage, load_simulation  # noqa: E402
from rescuerack.triage import load_triage_simulation  # noqa: E402


WIDTH = 640
HEIGHT = 368
FPS = 30
VIDEO_SECONDS = 154


def render_clip(
    writer: imageio.Writer,
    mode: str,
    seconds: float,
    camera: str,
    label: str,
    sample_stride: int = 4,
) -> dict[str, object]:
    model, data, controller = load_simulation(mode=mode)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    steps_per_frame = max(1, round(1.0 / (FPS * model.opt.timestep))) * sample_stride
    total_frames = round(seconds * FPS)

    for _ in range(total_frames):
        for _ in range(steps_per_frame):
            controller.step()
            mujoco.mj_step(model, data)
            if controller.stage == MissionStage.COMPLETE and controller.stage_time > 0.75:
                break
        renderer.update_scene(data, camera=camera)
        writer.append_data(renderer.render())

    renderer.close()
    summary = controller.summary()
    summary["clip_label"] = label
    return summary


def render_hold(writer: imageio.Writer, mode: str, seconds: float, camera: str) -> dict[str, object]:
    model, data, controller = load_simulation(mode=mode)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    while not (controller.stage == MissionStage.COMPLETE and controller.stage_time > 0.75):
        controller.step()
        mujoco.mj_step(model, data)

    for _ in range(round(seconds * FPS)):
        renderer.update_scene(data, camera=camera)
        writer.append_data(renderer.render())

    renderer.close()
    summary = controller.summary()
    summary["clip_label"] = "final_success_hold"
    return summary


def render_triage_clip(writer: imageio.Writer, seconds: float, label: str) -> dict[str, object]:
    model, data, controller = load_triage_simulation()
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    steps_per_frame = max(1, round(1.0 / (FPS * model.opt.timestep))) * 3
    total_frames = round(seconds * FPS)

    for _ in range(total_frames):
        for _ in range(steps_per_frame):
            controller.step()
            mujoco.mj_step(model, data)
            if controller.complete():
                break
        renderer.update_scene(data, camera="triage_cam")
        writer.append_data(renderer.render())

    renderer.close()
    summary = controller.summary()
    summary["clip_label"] = label
    return summary


def main() -> None:
    media_dir = ROOT / "media"
    media_dir.mkdir(exist_ok=True)
    output = media_dir / "demo.mp4"
    output.unlink(missing_ok=True)

    summaries: list[dict[str, object]] = []
    with imageio.get_writer(str(output), fps=FPS, codec="libx264", quality=8) as writer:
        summaries.append(render_clip(writer, "easy", 22, "overview", "easy_direct_retrieval"))
        summaries.append(render_clip(writer, "medium", 26, "overview", "medium_debris_route"))
        summaries.append(render_clip(writer, "hard", 44, "overview", "hard_astar_planned_route"))
        summaries.append(render_clip(writer, "hard", 20, "robot_chase", "hard_robot_chase_view", sample_stride=3))
        summaries.append(render_triage_clip(writer, 32, "dexterous_triage_station"))
        summaries.append(render_hold(writer, "hard", 10, "overview"))

    data_args = SimpleNamespace(
        model=ROOT / "models" / "rescuerack_scene.xml",
        mode="hard",
        duration=40.0,
        viewer=False,
        record=None,
        camera="overview",
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        trajectory=media_dir / "hard_trajectory.json",
        sample_hz=10.0,
    )
    from rescuerack.run_demo import run_headless

    hard_summary = run_headless(data_args)
    (media_dir / "demo_timeline.json").write_text(
        json.dumps(
            {
                "video": "media/demo.mp4",
                "duration_seconds": VIDEO_SECONDS,
                "fps": FPS,
                "clips": summaries,
                "hard_mode_data_summary": hard_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
