#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["chalkcompute>=2.13"]
# ///
"""Build the evaluation post-training trainer image with Chalk's image builder.

    ./scripts/build_trainer_image.py [--ref <git ref of chalk-ai/chalk-harbor>]

Prints the image URI to give as the post-training config's ``trainer_image``. The image is
``docker/trainer.Dockerfile`` expressed as a chalkcompute ``Image``: CUDA torch, transformers,
peft, chalkcompute, chalkpy, and chalk-harbor installed from GitHub at ``--ref`` (a pushed
commit, so the image holds exactly that trainer). Chalk's builder pushes to the registry the
environment's training runs pull from, so no registry credentials are needed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import chalkcompute
from chalkcompute import Image

REPO = "https://github.com/chalk-ai/chalk-harbor"
TORCH = "torch==2.9.1"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"


def trainer_image(ref: str) -> Image:
    return (
        Image.debian_slim("3.12")
        .apt_install(["git"])
        .run_commands(f"pip install --no-cache-dir '{TORCH}' --index-url {TORCH_INDEX}")
        .pip_install(
            [f"chalk-harbor[post-training] @ git+{REPO}@{ref}", "chalkpy", "polars"]
        )
        .env(
            {
                "HF_HOME": "/tmp/hf",
                "PYTHONUNBUFFERED": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
    )


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref",
        default=None,
        help="chalk-harbor commit to install (default: this checkout's HEAD, which must be pushed)",
    )
    args = parser.parse_args(argv)
    ref = args.ref or _head()
    uri = chalkcompute.build_image(trainer_image(ref))
    print(uri)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
