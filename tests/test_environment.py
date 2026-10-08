from __future__ import annotations

import base64
import io
import os
import re
import tarfile
from pathlib import Path

import pytest

from chalk_harbor.environment import _embedded_copy, _image_from_dockerfile


def _unpack(step: str) -> tarfile.TarFile:
    payload = re.search(r"echo (\S+) \| base64 -d", step).group(1)
    return tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload)), mode="r:gz")


def test_the_same_files_always_embed_as_the_same_step(tmp_path: Path) -> None:
    # The image service caches builds by their instructions; a COPY that re-embeds with a new
    # timestamp would rebuild the image for every trial.
    context = tmp_path / "environment"
    (context / "app" / "lib").mkdir(parents=True)
    (context / "app" / "lib" / "b.py").write_text("b = 2\n")
    (context / "app" / "a.py").write_text("a = 1\n")
    dockerfile = context / "Dockerfile"
    first = _embedded_copy(context, "app /opt/app", dockerfile)
    for path in (context / "app").rglob("*"):
        os.utime(path, (1_000_000, 1_000_000))
    assert _embedded_copy(context, "app /opt/app", dockerfile) == first

    with _unpack(first[0]) as tar:
        names = tar.getnames()
        assert names == [".", "./a.py", "./lib", "./lib/b.py"]
        assert {member.mtime for member in tar.getmembers()} == {0}
        assert tar.extractfile("./lib/b.py").read() == b"b = 2\n"


def test_a_dockerfile_too_large_for_one_build_fails_before_building(
    tmp_path: Path,
) -> None:
    import random

    context = tmp_path / "environment"
    (context / "data").mkdir(parents=True)
    # Incompressible, so its embedded COPY alone exceeds what one build accepts.
    (context / "data" / "blob.bin").write_bytes(random.Random(1).randbytes(200_000))
    (context / "Dockerfile").write_text("FROM python:3.13-slim\nCOPY data /opt/data\n")

    with pytest.raises(ValueError, match="128 KiB"):
        _image_from_dockerfile(context / "Dockerfile", {})
