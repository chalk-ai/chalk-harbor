from __future__ import annotations

import base64
import io
import os
import re
import tarfile
from pathlib import Path

from chalk_harbor.environment import _embedded_copy


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
