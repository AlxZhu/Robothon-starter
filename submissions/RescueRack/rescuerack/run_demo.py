"""Command-line runner for the RescueRack MuJoCo demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import mujoco

from .controller import DEFAULT_MODEL, MISSION_CONFIGS, MissionStage, load_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RescueRack MuJoCo demo.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to the MJCF scene.")
    parser.add_argument("--mode", choices=sorted(MISSION_CONFIGS), default="easy", help="Mission difficulty.")
    parser.add_argument("--duration", type=float, default=28.0, help="Maximum simulated seconds.")
    parser.add_argument("--viewer", action="store_true", help="Open the interactive MuJoCo viewer.")
    parser.add_argument("--record", type=Path, help="Optional MP4 path for a rendered demo video.")
    parser.add_argument("--camera", default="overview", help="Camera name used for rendering.")
    parser.add_argument("--width", type=int, default=640, help="Video width.")
    parser.add_argument("--height", type=int, default=368, help="Video height.")
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate.")
    return parser.parse_args()


def run_headless(args: argparse.Namespace) -> dict[str, object]:
    model, data, controller = load_simulation(args.model, mode=args.mode)

    writer = None
    renderer = None
    frame_interval = max(1, round(1.0 / (args.fps * model.opt.timestep)))

    if args.record:
        import imageio.v2 as imageio

        args.record.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(args.record), fps=args.fps, codec="libx264", quality=8)
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    steps = int(args.duration / model.opt.timestep)
    for step in range(steps):
        controller.step()
        mujoco.mj_step(model, data)

        if writer is not None and renderer is not None and step % frame_interval == 0:
            renderer.update_scene(data, camera=args.camera)
            writer.append_data(renderer.render())

        if controller.stage == MissionStage.COMPLETE and controller.stage_time > 0.75:
            break

    if renderer is not None:
        renderer.close()
    if writer is not None:
        writer.close()

    return controller.summary()


def run_viewer(args: argparse.Namespace) -> dict[str, object]:
    import mujoco.viewer

    model, data, controller = load_simulation(args.model, mode=args.mode)
    steps = int(args.duration / model.opt.timestep)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for _ in range(steps):
            step_start = time.time()
            controller.step()
            mujoco.mj_step(model, data)
            viewer.sync()
            sleep_time = model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
            if controller.stage == MissionStage.COMPLETE and controller.stage_time > 0.75:
                break

    return controller.summary()


def main() -> None:
    args = parse_args()
    summary = run_viewer(args) if args.viewer else run_headless(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
