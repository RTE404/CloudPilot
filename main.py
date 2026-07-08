"""CloudPilot local development launcher."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse launcher arguments."""
    parser = argparse.ArgumentParser(description="Run CloudPilot locally.")
    parser.add_argument(
        "--backend-only",
        action="store_true",
        help="Run only the FastAPI backend on port 8000.",
    )
    parser.add_argument(
        "--frontend-only",
        action="store_true",
        help="Run only the Vite frontend on port 5173.",
    )
    return parser.parse_args()


def main() -> None:
    """Start backend and frontend development servers."""
    args = parse_args()
    processes: list[subprocess.Popen] = []

    if not args.frontend_only:
        backend = subprocess.Popen(
            [sys.executable, "-m", "backend.main"],
            cwd=Path.cwd(),
        )
        processes.append(backend)

    if not args.backend_only:
        frontend_dir = Path("frontend")
        if not (frontend_dir / "node_modules").exists():
            raise RuntimeError(
                "Frontend dependencies are missing. Run `npm install` in "
                "the frontend directory, then run `python main.py` again."
            )
        frontend = subprocess.Popen(["npm.cmd", "run", "dev"], cwd=frontend_dir)
        processes.append(frontend)

    print("CloudPilot backend: http://127.0.0.1:8000")
    if not args.backend_only:
        print("CloudPilot dashboard: http://127.0.0.1:5173")

    try:
        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()


if __name__ == "__main__":
    main()
