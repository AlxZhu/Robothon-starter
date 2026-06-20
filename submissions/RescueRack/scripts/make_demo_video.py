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
HARD_STEPS_PER_FRAME = 7
TRIAGE_STEPS_PER_FRAME = 5
HARD_HOLD_SECONDS = 6
TRIAGE_HOLD_SECONDS = 5


def append_frame(
    writer: imageio.Writer,
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    camera: str,
) -> None:
    renderer.update_scene(data, camera=camera)
    writer.append_data(renderer.render())


def render_hard_rescue_story(writer: imageio.Writer) -> dict[str, object]:
    model, data, controller = load_simulation(mode="hard")
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    frames = 0
    stage_segments: list[dict[str, object]] = []
    active_stage = controller.stage.value
    active_camera = "overview"
    segment_start_frame = 0

    def camera_for_stage() -> str:
        if controller.stage in {MissionStage.PREPARE_GRASP, MissionStage.SECURE_SUPPLY}:
            return "robot_chase"
        if controller.stage == MissionStage.TRANSPORT_TO_SAFE_ZONE and controller.base_xy()[0] > -1.8:
            return "robot_chase"
        return "overview"

    def close_segment(end_frame: int) -> None:
        stage_segments.append(
            {
                "stage": active_stage,
                "camera": active_camera,
                "start_second": round(segment_start_frame / FPS, 2),
                "end_second": round(end_frame / FPS, 2),
            }
        )

    while True:
        for _ in range(HARD_STEPS_PER_FRAME):
            controller.step()
            mujoco.mj_step(model, data)

        next_stage = controller.stage.value
        next_camera = camera_for_stage()
        if next_stage != active_stage or next_camera != active_camera:
            close_segment(frames)
            active_stage = next_stage
            active_camera = next_camera
            segment_start_frame = frames

        append_frame(writer, renderer, data, active_camera)
        frames += 1

        if controller.stage == MissionStage.COMPLETE and controller.stage_time > HARD_HOLD_SECONDS:
            break

    close_segment(frames)

    renderer.close()
    summary = controller.summary()
    summary["clip_label"] = "hard_single_take_rescue"
    summary["video_frames"] = frames
    summary["video_seconds"] = round(frames / FPS, 2)
    summary["stage_segments"] = stage_segments
    return summary


def render_triage_story(writer: imageio.Writer) -> dict[str, object]:
    model, data, controller = load_triage_simulation()
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    frames = 0

    while not controller.complete():
        for _ in range(TRIAGE_STEPS_PER_FRAME):
            controller.step()
            mujoco.mj_step(model, data)
            if controller.complete():
                break
        append_frame(writer, renderer, data, "triage_cam")
        frames += 1

    for _ in range(round(TRIAGE_HOLD_SECONDS * FPS)):
        append_frame(writer, renderer, data, "triage_cam")
        frames += 1

    renderer.close()
    summary = controller.summary()
    summary["clip_label"] = "dexterous_triage_followup"
    summary["video_frames"] = frames
    summary["video_seconds"] = round(frames / FPS, 2)
    return summary


def main() -> None:
    media_dir = ROOT / "media"
    media_dir.mkdir(exist_ok=True)
    output = media_dir / "demo.mp4"
    output.unlink(missing_ok=True)

    summaries: list[dict[str, object]] = []
    with imageio.get_writer(str(output), fps=FPS, codec="libx264", quality=8) as writer:
        summaries.append(render_hard_rescue_story(writer))
        summaries.append(render_triage_story(writer))

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
    total_frames = sum(int(summary["video_frames"]) for summary in summaries)
    duration_seconds = round(total_frames / FPS, 2)
    (media_dir / "demo_timeline.json").write_text(
        json.dumps(
            {
                "video": "media/demo.mp4",
                "duration_seconds": duration_seconds,
                "fps": FPS,
                "storyboard": [
                    "single continuous hard-mode rescue mission with camera cuts",
                    "five-finger DexTriage Lab follow-up with 20 micro-tasks",
                ],
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
