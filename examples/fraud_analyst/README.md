# Kestrel Pay: an agentic fraud case analyst as a Harbor eval on Chalk

A fraud-review benchmark for a fictional payments company, Kestrel Pay. Each case is a customer payout held
for manual review. An agent investigates it with tools and must **approve** (legitimate) or
**deny** (fraud) it, with an analysis a human reviewer can audit. `fraud_eval.py` runs the cases as
a Chalk evaluation and scores the decision, the quality of the analysis, and what the
investigation spent.

```bash
./build_tasks.py                                  # population.py -> tasks/<case>/ (40 cases)
uv run --with pytest pytest test_cases.py         # every case's reference investigation scores 1.0
./fraud_eval.py                                   # 12 demo cases as a Chalk evaluation (--all for 40)
./fraud_eval.py --rescore <run-id>                # re-judge an earlier run without re-running agents
```

## The population

`population.py` generates a seeded population: 360 accounts with logins and transactions, of
which 320 are historical with known outcomes (about 20% fraud, including eight fraud rings that
share devices), and a review queue of 40 held payouts (20 fraud, 20 legitimate) in seven
archetypes:

| Archetype | Label | What settles it |
| --- | --- | --- |
| `ring_member` | fraud | free: shares a login device with closed fraud accounts (SQL); paid network search confirms consortium links |
| `account_takeover` | fraud | free: old account, new device and country, VPN, password reset, new payout destination; deep verification shows a SIM swap |
| `synthetic_identity` | fraud | paid: features look clean (risk 0.2-0.4); deep verification fails SSN/name match with a high synthetic-identity score |
| `legit_clean` | legit | free: long tenure, consistent behaviour, low risk score |
| `legit_traveler` | legit | looks risky (new country, VPN, risk 0.5-0.7); same device as always, deep verification clean |
| `legit_thin_file` | legit | looks risky (new account, risk 0.45-0.7); deep verification and network search clean |
| `legit_household` | legit | shares a device with another customer in good standing, same surname and city |

The model's `risk_score` is deliberately misleading on synthetic identities and risky-looking
legitimate customers, so an agent that thresholds the score fails those cases.

## What the agent gets

| Tool | Cost | Runs |
| --- | --- | --- |
| `run_sql` | free | read-only SQLite over `accounts`, `logins`, `transactions`, `review_queue` |
| `chalk_query` | free | 12 online features per account (age, risk score, velocity, devices, countries, VPN ratio, password reset, new payout destination, shared-device count, ...) |
| `run_python` | free | Python 3.13 in the sandbox as an unprivileged user, no network |
| `deep_verification` | **$5** per call | document/SSN/liveness match, synthetic-identity score, phone tenure, SIM swap, address history |
| `social_network_search` | **$2** per call | identities linked across the fraud consortium and their status elsewhere |
| `submit_decision` | free | `approve` or `deny`, confidence, and the analysis; closes the case |

All tools run in the trial's Chalk sandbox (no network) through the `fraudlab` backend, which
meters every call into the case's ledger. The paid tools' results are sealed root-only, out of
reach of the agent's python. `analyst_agent.py` is the Harbor agent: an OpenAI-compatible
tool-calling loop (Chalk's AI router works with a Chalk token) that writes an ATIF trajectory,
so each turn streams into the row's Chalk trace.

The data is generated inside the image at build time (`fraudlab/gen/materialize.py` with a copy
of `population.py`), rather than copied in: an image build accepts at most 128 KiB of Dockerfile,
and the warehouse alone would exceed it.

## Scoring

Harbor's verifier grades the ledger: reward 1 if the decision matches the hidden label, and it
records the paid-check spend. The Chalk evaluation adds:

| Scorer | Score |
| --- | --- |
| `kestrel-decision-correct` | 1 if approve/deny matches the label |
| `kestrel-analysis-quality` | LLM judge (blind to the label) rating 1-5 each: evidence grounding, coherence, counter-evidence, decision support, proportionality of spend; mean mapped to 0-1, with the issues it found in metadata |
| `kestrel-investigation-cost` | `1 / (1 + dollars / 7)`; dollars and paid calls in metadata |

## Files

| Path | What |
| --- | --- |
| `population.py` | the seeded population and review queue |
| `build_tasks.py` | renders `tasks/<case>/` (instruction with data dictionary, task.toml, shared environment, rubric, oracle) |
| `fraudlab/` | the in-sandbox backend (`lib/fraudlab`), CLI (`bin/fraudlab`) and data generator (`gen/`) |
| `analyst_agent.py` | the Harbor agent |
| `fraud_eval.py` | the Chalk evaluation: trial function, scorers, run, rescore |
| `test_cases.py` | self-checks: reference investigations, billing, read-only SQL, closed cases |
