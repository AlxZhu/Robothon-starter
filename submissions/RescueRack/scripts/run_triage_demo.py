"""Run the RescueRack dexterous triage station demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rescuerack.triage import load_triage_simulation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RescueRack dexterous triage demo.")
    parser.add_argument("--duration", type=float, default=28.0)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--sample-hz", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, data, controller = load_triage_simulation()
    samples: list[dict[str, object]] = []
    sample_interval = max(1, round(1.0 / (args.sample_hz * model.opt.timestep)))

    for step in range(int(args.duration / model.opt.timestep)):
        controller.step()
        mujoco.mj_step(model, data)
        if args.trajectory and step % sample_interval == 0:
            samples.append(controller.observation())
        if controller.complete():
            break

    summary = controller.summary()
    print(json.dumps(summary, indent=2))
    if args.trajectory:
        args.trajectory.parent.mkdir(parents=True, exist_ok=True)
        args.trajectory.write_text(json.dumps({"summary": summary, "samples": samples}, indent=2), encoding="utf-8")
    if not summary["success"]:
        raise SystemExit("Dexterous triage demo did not complete.")


if __name__ == "__main__":
    main()
