import pytest

from chalk_harbor.evaluation import (
    EVALUATION_ID_ENV,
    EVALUATION_RUN_ID_ENV,
    SESSION_ID_ENV,
    sandbox_tags,
)

EVALUATION_ID = "cmuuf4o4d005w0116nmsftjao"
RUN_ID = "cb8593d3-fa16-4ae5-a7f0-bec54d08d4f8"
ROW_ID = "9700da0e-9909-449b-b627-e336a527f399"


def test_tags_name_the_evaluation_run_and_row() -> None:
    tags = sandbox_tags(
        {
            EVALUATION_ID_ENV: EVALUATION_ID,
            EVALUATION_RUN_ID_ENV: RUN_ID,
            SESSION_ID_ENV: f"{RUN_ID}:{ROW_ID}",
        }
    )

    # The 73-character session id is no label value; its run id and row id are, and
    # `run_id:row_id` rebuilds it.
    assert tags == {
        "chalk.evaluation.id": EVALUATION_ID,
        "chalk.evaluation.run_id": RUN_ID,
        "chalk.evaluation.row_id": ROW_ID,
    }


def test_the_session_supplies_the_run_when_the_call_metadata_does_not() -> None:
    tags = sandbox_tags({SESSION_ID_ENV: f"{RUN_ID}:{ROW_ID}"})

    assert tags == {
        "chalk.evaluation.run_id": RUN_ID,
        "chalk.evaluation.row_id": ROW_ID,
    }


def test_a_session_from_another_run_is_kept_whole() -> None:
    # Only an evaluation row's `<run id>:<row id>` session is split; any other session id
    # is tagged as it is.
    tags = sandbox_tags(
        {EVALUATION_RUN_ID_ENV: RUN_ID, SESSION_ID_ENV: "plain-session"}
    )

    assert tags == {
        "chalk.evaluation.run_id": RUN_ID,
        "chalk.evaluation.session_id": "plain-session",
    }


@pytest.mark.parametrize(
    "value", ["", "x" * 64, "has space", "-leading", "trailing.", "a:b"]
)
def test_invalid_label_values_are_left_out(value: str) -> None:
    assert sandbox_tags({EVALUATION_ID_ENV: value, EVALUATION_RUN_ID_ENV: RUN_ID}) == {
        "chalk.evaluation.run_id": RUN_ID
    }


def test_outside_an_evaluation_there_are_no_tags() -> None:
    assert sandbox_tags({}) == {}
