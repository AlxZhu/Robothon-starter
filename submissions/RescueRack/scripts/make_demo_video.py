"""Render a short RescueRack demo video."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    output = ROOT / "media" / "demo.mp4"
    output.unlink(missing_ok=True)
    command = [
        sys.executable,
        "-m",
        "rescuerack.run_demo",
        "--mode",
        "hard",
        "--duration",
        "24",
        "--record",
        str(output),
        "--width",
        "640",
        "--height",
        "368",
        "--fps",
        "30",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
