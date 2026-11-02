import os
import shlex
import subprocess

import pytest
from harbor.models.task.config import NetworkMode, NetworkPolicy

from chalk_harbor.environment import (
    _EMBED_BUDGET_BYTES,
    ChalkSandboxEnvironment,
    _embedded_copy,
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
    assert (
        sum(chunk.stat().st_size for chunk in chunks) <= _EMBED_BUDGET_BYTES * 4 / 3 + 4
    )
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


def test_copy_embed_budget(tmp_path):
    (tmp_path / "data.bin").write_bytes(os.urandom(_EMBED_BUDGET_BYTES + 1))

    with pytest.raises(ValueError, match="over the .* embed budget"):
        _embedded_copy(tmp_path, "data.bin /data.bin", tmp_path / "Dockerfile")
