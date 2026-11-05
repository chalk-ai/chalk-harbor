"""Run the inactive CPU checkpoint checks with an existing Python environment.

No installs, model downloads, service calls or GPU claims. The tests emulate absent
accelerator capability on GPU-capable workstations but execute actual CPU Torch math.
"""

import argparse
import os
import signal
import subprocess
from pathlib import Path


def run(command: list[str], root: Path) -> None:
    child = subprocess.Popen(command, cwd=root, start_new_session=True)
    try:
        result = child.wait(timeout=60)
    except BaseException:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()
        raise
    if result:
        raise SystemExit(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        required=True,
        help="Existing interpreter with torch/numpy/pytest/ruff",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sources = [
        "chalk_harbor/post_training/full_checkpoint.py",
        "tests/test_full_checkpoint.py",
        "scripts/check_full_checkpoint.py",
    ]
    run(
        [
            args.python,
            "-c",
            "import sys,torch; print('Validation runtime:',sys.version,torch.__version__,flush=True); print('Project requires Python >=3.12,<3.15; compare the version above. GPU-unavailability responses are mocked only in tests.',flush=True)",
        ],
        root,
    )
    run([args.python, "-m", "ruff", "check", *sources], root)
    run([args.python, "-m", "ruff", "format", "--check", *sources], root)
    run(
        [
            args.python,
            "-B",
            "-m",
            "pytest",
            "-q",
            "--tb=short",
            "tests/test_full_checkpoint.py",
        ],
        root,
    )


if __name__ == "__main__":
    main()
