"""Which Chalk evaluation, run and row a Harbor trial belongs to.

A trial launched from a Chalk evaluation task runs in two processes: the task function,
where the call's evaluation metadata and row session are visible, and the ``harbor run``
it starts, where the sandbox is created and none of that is. ``evaluation_env()`` carries
the ids across as environment variables; ``sandbox_tags()`` reads them back (or the call
context, when Harbor runs in-process) and renders them as sandbox tags.

Sandbox tags become Kubernetes labels verbatim, so every value must be a valid label value:
at most 63 characters of ``[A-Za-z0-9._-]``, alphanumeric at both ends. An evaluation row's
session id is ``<run id>:<row id>`` (73 characters), which is not; it is tagged as its row
id, alongside the run id that, joined with a colon, rebuilds it.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any

# Environment variables `harbor run` inherits from the evaluation task.
EVALUATION_ID_ENV = "CHALK_HARBOR_EVALUATION_ID"
EVALUATION_RUN_ID_ENV = "CHALK_HARBOR_EVALUATION_RUN_ID"
SESSION_ID_ENV = "CHALK_HARBOR_SESSION_ID"

# Call metadata a Chalk evaluation attaches to every task and scorer call.
_EVALUATION_ID_HEADER = "x-chalk-evaluation-id"
_EVALUATION_RUN_ID_HEADER = "x-chalk-evaluation-run-id"
_SESSION_ID_HEADER = "x-chalk-session-id"

_LABEL_VALUE = re.compile(r"^(?:[A-Za-z0-9](?:[A-Za-z0-9._-]{0,61}[A-Za-z0-9])?)$")


def evaluation_context() -> dict[str, str]:
    """The current Chalk call's evaluation id, run id and session id, where present.

    Keys are ``evaluation_id``, ``evaluation_run_id`` and ``session_id``. Empty outside a
    Chalk function call, or where chalkcompute is not installed.
    """
    try:
        import chalkcompute
    except ImportError:
        return {}
    call_context = {
        key.lower(): value for key, value in chalkcompute.get_call_context().items()
    }
    context = {
        "evaluation_id": call_context.get(_EVALUATION_ID_HEADER),
        "evaluation_run_id": call_context.get(_EVALUATION_RUN_ID_HEADER),
        "session_id": _current_session_id() or call_context.get(_SESSION_ID_HEADER),
    }
    return {key: value for key, value in context.items() if value}


def evaluation_env() -> dict[str, str]:
    """``evaluation_context()`` as the environment variables ``sandbox_tags`` reads.

    Merge into the environment of the ``harbor run`` a Chalk evaluation task starts.
    """
    context = evaluation_context()
    names = {
        "evaluation_id": EVALUATION_ID_ENV,
        "evaluation_run_id": EVALUATION_RUN_ID_ENV,
        "session_id": SESSION_ID_ENV,
    }
    return {names[key]: value for key, value in context.items()}


def sandbox_tags(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Sandbox tags naming the evaluation, run and row a trial belongs to.

    Read from the variables ``evaluation_env`` sets, falling back to the current call's
    context when Harbor runs in the task's own process. Values that are not valid label
    values are left out rather than allowed to fail the sandbox's creation.
    """
    environ = os.environ if environ is None else environ
    context = {
        "evaluation_id": environ.get(EVALUATION_ID_ENV),
        "evaluation_run_id": environ.get(EVALUATION_RUN_ID_ENV),
        "session_id": environ.get(SESSION_ID_ENV),
    }
    if not any(context.values()):
        context = dict(evaluation_context())
    tags = {
        "chalk.evaluation.id": context.get("evaluation_id"),
        "chalk.evaluation.run_id": context.get("evaluation_run_id"),
    }
    session_id = context.get("session_id")
    if session_id:
        run_id, separator, row_id = session_id.partition(":")
        if separator and (
            not tags["chalk.evaluation.run_id"]
            or run_id == tags["chalk.evaluation.run_id"]
        ):
            tags["chalk.evaluation.run_id"] = run_id
            tags["chalk.evaluation.row_id"] = row_id
        else:
            tags["chalk.evaluation.session_id"] = session_id
    return {
        key: value for key, value in tags.items() if value and _LABEL_VALUE.match(value)
    }


def evaluation_run_metadata(evaluation_run_id: str | None) -> dict[str, Any]:
    """The metadata of an evaluation run, or ``{}`` without a run or when it can't be read.

    A run's metadata is how one deployed task takes per-run settings, such as the model a
    post-training rollout calls. Read failures return ``{}`` so the task falls back to its
    deployed defaults instead of failing the row.
    """
    if not evaluation_run_id:
        return {}
    try:
        import chalkcompute

        run = chalkcompute.EvaluationRun.from_id(evaluation_run_id)
    except Exception as exc:  # noqa: BLE001 - see the docstring
        print(
            f"evaluation run {evaluation_run_id} metadata unavailable: {exc}",
            flush=True,
        )
        return {}
    return dict(run.metadata or {})


def trial_tag(metadata: Mapping[str, Any], run_tag: str) -> str:
    """The directory a run's trial records go under in a traces volume.

    ``trace_tag`` in the run's metadata names it. Post-training rollouts name none, and go
    under ``<run_tag>/posttrain-<id>/iter-<k>/sample-<s>`` from their ``post_training_id``,
    ``iteration`` and ``sample``: a post-training's runs share one dataset, and with it one
    ``run_tag``, so a shared directory would mix every sample's trials. Otherwise it is
    ``run_tag``.
    """
    tag = metadata.get("trace_tag")
    if tag:
        return str(tag)
    if metadata.get("post_training_id"):
        return "/".join(
            [
                run_tag,
                f"posttrain-{metadata['post_training_id']}",
                f"iter-{_whole(metadata.get('iteration', 0))}",
                f"sample-{_whole(metadata.get('sample', 0))}",
            ]
        )
    return run_tag


def _whole(value: Any) -> str:
    # Run metadata travels as a protobuf Struct, where every number is a double.
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _current_session_id() -> str | None:
    # The session chalkcompute is stamping spans with: the row's, set from row metadata or
    # the session header. Private in chalkcompute, so its absence is not an error.
    try:
        from chalkcompute._tracing import current_session_id
    except ImportError:
        return None
    return current_session_id()


__all__ = [
    "EVALUATION_ID_ENV",
    "EVALUATION_RUN_ID_ENV",
    "SESSION_ID_ENV",
    "evaluation_context",
    "evaluation_env",
    "evaluation_run_metadata",
    "sandbox_tags",
    "trial_tag",
]
