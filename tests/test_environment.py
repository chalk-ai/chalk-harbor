import os
import shlex
import subprocess

import pytest
from harbor.models.task.config import NetworkMode, NetworkPolicy

from chalk_harbor.environment import (
    _DOCKERFILE_INSTRUCTION_BUDGET_BYTES,
    ChalkSandboxEnvironment,
    _embedded_copy,
    _image_from_dockerfile,
)


def _run_steps(steps):
    for step in steps:
        assert step.startswith("RUN ")
        subprocess.run(["sh", "-c", step[4:]], check=True)


def test_large_copy(tmp_path):
    source = tmp_path / "context"
    source.mkdir()
    data = os.urandom(int(3.9 * 1024 * 1024))
    (source / "data.bin").write_bytes(data)
    (source / "nested").mkdir()
    (source / "nested" / "hello.txt").write_text("hello\n")
    dest = tmp_path / "output with spaces"

    steps = _embedded_copy(
        tmp_path, f"context {shlex.quote(str(dest))}", tmp_path / "Dockerfile"
    )

    assert 56 <= len(steps) <= 60
    assert all(1 <= step.count("printf %s") <= 3 for step in steps[:-1])
    assert all(len(step[4:].encode()) < 128 * 1024 for step in steps)
    assert all(
        len(line.encode()) < 65535 for step in steps for line in step.splitlines()
    )
    steps = [
        step.replace("/tmp/.harbor-copy-", f"{tmp_path}/.harbor-copy-")
        for step in steps
    ]
    _run_steps(steps[:-1])
    [chunk_dir] = tmp_path.glob(".harbor-copy-*")
    chunks = sorted(chunk_dir.iterdir())
    assert all(chunk.stat().st_size <= 32 * 1024 for chunk in chunks)
    assert [chunk.name for chunk in chunks] == [
        f"{index:04d}" for index in range(len(chunks))
    ]
    assert len(steps) == (len(chunks) + 2) // 3 + 1
    assert len(data) * 4 / 3 <= sum(chunk.stat().st_size for chunk in chunks)
    _run_steps(steps[-1:])
    assert not chunk_dir.exists()
    assert (dest / "data.bin").read_bytes() == data
    assert (dest / "nested" / "hello.txt").read_text() == "hello\n"


def test_renamed_copy_preserves_existing_file(tmp_path):
    source = tmp_path / "agent.py"
    source.write_text("original\n")
    dest = tmp_path / "agent dir"
    dockerfile = tmp_path / "Dockerfile"
    _run_steps(
        _embedded_copy(
            tmp_path, f"agent.py {shlex.quote(str(dest / 'agent.py'))}", dockerfile
        )
    )
    source.write_text("scoped\n")

    _run_steps(
        _embedded_copy(
            tmp_path,
            f"agent.py {shlex.quote(str(dest / 'scoped_router.py'))}",
            dockerfile,
        )
    )

    assert (dest / "agent.py").read_text() == "original\n"
    assert (dest / "scoped_router.py").read_text() == "scoped\n"


def test_wildcard_network_policy():
    environment = object.__new__(ChalkSandboxEnvironment)
    environment._network_policy = NetworkPolicy(
        network_mode=NetworkMode.ALLOWLIST,
        allowed_hosts=["*.chalk.dev", "api.chalk.ai", "10.0.0.0/8"],
    )

    assert environment.capabilities.network_allowlist_wildcard_hostnames
    assert environment._sandbox_network_policy().allowed_hosts == {
        "*.chalk.dev": [],
        "api.chalk.ai": [],
    }


def test_copy_instructions_are_deterministic(tmp_path, monkeypatch):
    (tmp_path / "data.bin").write_bytes(os.urandom(300 * 1024))
    monkeypatch.setattr("gzip.time.time", lambda: 1000)
    first = _embedded_copy(tmp_path, "data.bin /data.bin", tmp_path / "Dockerfile")
    monkeypatch.setattr("gzip.time.time", lambda: 2000)

    assert (
        _embedded_copy(tmp_path, "data.bin /data.bin", tmp_path / "Dockerfile") == first
    )


@pytest.mark.parametrize("offset", [-1, 1])
def test_copy_instruction_budget_boundary(tmp_path, offset):
    assert _DOCKERFILE_INSTRUCTION_BUDGET_BYTES == 122_880
    (tmp_path / "data.bin").write_bytes(
        os.urandom((_DOCKERFILE_INSTRUCTION_BUDGET_BYTES - 4096) * 3 // 4)
    )
    dockerfile = tmp_path / "Dockerfile"
    steps = _embedded_copy(tmp_path, "data.bin /data.bin", dockerfile)
    copy_total = sum(len(step.encode()) for step in steps)
    assert 64 * 1024 < copy_total < _DOCKERFILE_INSTRUCTION_BUDGET_BYTES
    label = "LABEL data="
    padding = _DOCKERFILE_INSTRUCTION_BUDGET_BYTES + offset - copy_total - len(label)
    dockerfile.write_text(
        "FROM ubuntu:24.04\nCOPY data.bin /data.bin\n" + label + "x" * padding + "\n"
    )

    if offset > 0:
        with pytest.raises(ValueError) as exc:
            _image_from_dockerfile(dockerfile, {})
        message = str(exc.value)
        assert "total 122881 bytes" in message
        assert "122880-byte instruction budget" in message
        assert (
            "Chalk's image builder fails silently on Dockerfiles over about 128 KiB; "
            "mount larger data with --ek volumes=..."
        ) in message
        return

    image = _image_from_dockerfile(dockerfile, {})

    commands = image._steps[0].step.value.commands
    total = sum(len(command.encode()) for command in commands)
    assert total == _DOCKERFILE_INSTRUCTION_BUDGET_BYTES - 1


def test_multiple_copies_exceed_instruction_budget(tmp_path):
    for source in ("first.bin", "second.bin"):
        (tmp_path / source).write_bytes(os.urandom(48 * 1024))
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "FROM ubuntu:24.04\nCOPY first.bin /first.bin\nCOPY second.bin /second.bin\n"
    )
    source_totals = [
        sum(
            len(step.encode())
            for step in _embedded_copy(tmp_path, f"{source} /{source}", dockerfile)
        )
        for source in ("first.bin", "second.bin")
    ]
    assert all(total < _DOCKERFILE_INSTRUCTION_BUDGET_BYTES for total in source_totals)
    expected_total = sum(source_totals)
    assert expected_total > _DOCKERFILE_INSTRUCTION_BUDGET_BYTES

    with pytest.raises(ValueError) as exc:
        _image_from_dockerfile(dockerfile, {})

    message = str(exc.value)
    assert f"total {expected_total} bytes" in message
    assert f"{_DOCKERFILE_INSTRUCTION_BUDGET_BYTES}-byte instruction budget" in message
    assert "first.bin" in message
    assert "second.bin" in message
    assert "--ek volumes=..." in message


def test_other_instructions_count_toward_budget(tmp_path):
    dockerfile = tmp_path / "Dockerfile"
    # UTF-8 byte count, including non-RUN instructions.
    dockerfile.write_text(
        'FROM ubuntu:24.04\nLABEL data="'
        + "é" * (_DOCKERFILE_INSTRUCTION_BUDGET_BYTES // 2)
        + '"\n'
    )

    with pytest.raises(ValueError, match="instruction budget"):
        _image_from_dockerfile(dockerfile, {})
