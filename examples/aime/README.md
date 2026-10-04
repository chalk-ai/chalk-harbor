# AIME as a Chalk evaluation

`aime_eval.py` runs Harbor's `aime@1.0` (60 tasks) as a Chalk evaluation: one row per task,
each row one Harbor trial with the `terminus-2` agent on `openai/gpt-5-mini`, in a Chalk
sandbox. The model key is the environment's `OPENAI_API_KEY` Chalk secret.

```bash
harbor download aime@1.0 -o /tmp/aime && mv /tmp/aime/aime tasks
./aime_eval.py            # or --limit 3 for a smoke run
```

Every row keeps Harbor's native record. The trial directory exactly as Harbor wrote it goes to
the `harbor-traces` volume under `<tag>/<task>/`: `result.json`, the ATIF
`agent/trajectory.json`, the asciinema recording, verifier output, plus the task's problem and
answer key and a `chalk.json` naming the row's session. The same record is replayed as spans
into the row's Chalk trace (`chalk_harbor.tracing`). At the end, `<tag>/manifest.json` holds the
evaluation and run ids and every row's result.

## Comparing in Braintrust

`to_braintrust.py <tag>` reads one run back from the volume and logs it as a Braintrust
experiment. Each task becomes one row with the same reward the Chalk evaluation scored, with
the problem as input, the agent's answer and final message as output, and the answer key as
expected. Harbor's phases, each model turn and each command become the row's span tree,
taken from the same files and timestamps.

```bash
BRAINTRUST_API_KEY=... ./to_braintrust.py aime-20261004-221734 --project harbor-aime
./to_braintrust.py aime-20261004-221734 --jsonl out.jsonl   # no key: write the rows locally
```

The `answer` field is parsed from the agent's commands and is best-effort. The score is
Harbor's verifier reward.
