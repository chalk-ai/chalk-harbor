# Larkspur: customer-support agents as a Harbor benchmark on Chalk

60 support tickets for **Larkspur Home Delivery & Installation**, a fictional company that sells,
delivers and installs home appliances. Each ticket is a Harbor task. An agent works it with real
tools, talks to a simulated customer, and is scored on whether it followed policy, what its
actions cost, and how the customer felt. `support_eval.py` runs all 60 as a Chalk evaluation.

```bash
./build_tasks.py                      # scenarios.py -> tasks/<id>/
uv run --with pytest pytest test_scenarios.py   # every rubric gives the reference resolution 1.0
./support_eval.py                     # all 60 tickets as a Chalk evaluation (or --only <id> ...)
./support_eval.py --rescore <run-id>  # score an earlier run's outputs with the current scorers
./summarize.py runs/<tag>.json        # per-ticket scores as a markdown table
```

## What the agent gets

**A rich ticket prompt** (`instruction.md`): the customer's opening message, a pre-fetched
customer profile (tier, lifetime value, tenure, refunds and chargebacks in the last 180 days,
exceptions used, survey history, payment method, notes), the contact history, and every order
record (items, fees, installation, ProtectPlan, haul-away, delivery events and driver notes,
proof of delivery).

**Tools**, defined in `support_agent.py`:

| Tool | Runs | What it does |
| --- | --- | --- |
| `run_python`, `run_bash` | sandbox, as user `agent` | Python 3.13 / bash, no network, 20 s limit |
| `search_knowledge_base` | sandbox | BM25 search over 25 policy articles (`helpdesk/kb/`) |
| `issue_refund` | sandbox, `helpdesk` | refund or goodwill credit to the original payment method |
| `approve_exchange_exception` | sandbox, `helpdesk` | out-of-window exchange exception for one item |
| `escalate_to_human` | sandbox, `helpdesk` | hand off to `approvals`, `claims`, `risk`, `safety` or `supervisor` |
| `schedule_followup` | sandbox, `helpdesk` | reminder to contact the customer on a date |
| `dispatch_technician` | sandbox, `helpdesk` | book a technician or crew visit for a date and slot |
| `send_message_to_customer` | harness | message the customer; returns their reply |
| `end_conversation` | sandbox, `helpdesk` | close the ticket |

The sandbox is tight: `network_mode = "no-network"`, 1 CPU and 2 GiB, and the agent's code runs
as an unprivileged user that cannot read the helpdesk state or the customers' personas. Every
support action goes through the in-sandbox `helpdesk` backend. It enforces what a real system
would (the order is on the account, refunds never exceed what was paid, visits are bookable),
then appends the action to the ticket's ledger. Whether an action was the right one is graded
afterwards.

**The simulated customer** is a second model (default `openai/gpt-5.4-mini`) that plays a
persona sealed in the sandbox: who they are, what they want and would accept, facts they reveal
only when asked (availability, photos, what really happened), and what makes them angrier or
calmer. It reports its frustration (0–10) with every reply, decides when it is done, and answers
a one-question CSAT survey when the chat ends.

The agent loop and the customer run in Harbor's process, where the model endpoint is reachable;
only the tools run in the sandbox. Harbor 0.24 also has a simulated-user mode, but it runs the
user agent inside the task container, and that would need network in the container.

## The tickets

All take place on Tuesday 2026-10-06. Each one tests a specific policy judgment. Several test
restraint: the right answer is a clear "no", or a pointer to self-service. The second 30
(`scenarios_batch2.py`) sit on policy boundaries (day 14, day 75 vs. 76, the 30-minute grace
period, the 48-hour damage window) or test over-caution, privacy, billing errors and recalls.

| Task | Category | Difficulty | Ticket |
| --- | --- | --- | --- |
| `missed-delivery-first` | missed delivery | easy | First missed delivery window, no call |
| `second-missed-window` | missed delivery | medium | Second missed window on the same order, fee never refunded |
| `missed-delivery-customer-not-home` | missed delivery | medium | Customer says nobody came; driver called and knocked |
| `price-drop-in-window` | price adjustment | easy | Larkspur price drop within 14 days |
| `price-drop-too-late` | price adjustment | easy | Price drop 26 days after the order |
| `competitor-price-match` | price adjustment | easy | Competitor price-match request on an undelivered order |
| `haul-away-missed` | haul away | medium | Crew left the old dryer behind |
| `damage-hidden-side-48h` | delivery damage | medium | Dent on a hidden side panel, reported next day |
| `damage-visible-over-authority` | delivery damage | hard | Visible front damage; discount exceeds agent authority |
| `late-cosmetic-damage-platinum` | delivery damage | medium | Cosmetic scratch reported 17 days after delivery by a Platinum customer |
| `oow-exchange-gold-eligible` | exchange exception | medium | Gold customer, freezer failed 57 days after delivery |
| `oow-exchange-standard-ineligible` | exchange exception | medium | Standard customer, microwave failed at 68 days, no Larkspur failure |
| `oow-exchange-larkspur-failure` | exchange exception | hard | Standard customer whose order had two missed deliveries; dishwasher pump fails at 67 days |
| `oow-too-late-protectplan` | exchange exception | hard | Platinum customer wants an exchange 113 days after delivery; has ProtectPlan |
| `oow-exchange-prior-exception` | exchange exception | hard | Gold customer, eligible except for an exception used in March |
| `buyers-remorse-color` | exchange exception | medium | Gold customer wants a different finish 40 days after delivery |
| `install-leak-warranty` | installation | medium | Slow drip at a Larkspur-installed dishwasher drain connection |
| `install-leak-floor-damage` | installation | hard | Washer supply hose leaked while away; hardwood floor damaged |
| `gas-smell-after-install` | safety | hard | Gas smell the day after a gas range installation |
| `burning-smell-dryer` | safety | hard | Hot cord and burning smell from a dryer Larkspur wired |
| `protectplan-repair-food-loss` | repair | medium | Fridge stopped cooling at 19 months; ProtectPlan; wants grocery money |
| `paid-service-call` | repair | medium | Two-year-old oven not heating; no coverage |
| `fraud-not-delivered-claim` | fraud | hard | 'Never delivered' claim contradicted by POD; refund history and chargeback |
| `refund-different-payment-method` | missed delivery | medium | Legitimate fee refund, but customer wants it on Venmo |
| `legal-threat-spoiled-food` | escalation | hard | Missed fridge delivery; customer threatens small claims |
| `third-miss-demands-600` | missed delivery | hard | Third missed window; Gold customer demands a $600 credit |
| `backorder-delay-goodwill` | backorder | medium | Backordered range now 21 days past the promised date |
| `in-window-size-exchange` | exchange | easy | In-window exchange for a larger model (self-serve) |
| `return-installed-not-defective` | return | medium | Return of an installed dishwasher that is within spec |
| `incomplete-gas-dryer-install` | installation | medium | Gas dryer delivered but not installed (crew lacked a connector) |
| `missed-white-glove-gold-first` | missed delivery | easy | Gold customer, first missed white-glove window |
| `late-within-grace` | missed delivery | medium | Crew arrived 25 minutes after the window without calling |
| `late-with-advance-call` | missed delivery | medium | Dispatcher called three hours ahead to move the window |
| `second-miss-fee-already-refunded` | missed delivery | medium | Second miss after the fee was refunded for the first |
| `damage-visible-gold-within-authority` | delivery damage | medium | Visible front dent, Gold customer, within $500 authority |
| `damage-reported-after-48h` | delivery damage | medium | Cosmetic scratch noticed Sunday, reported Tuesday (about 71 hours) |
| `damage-noted-on-pod-late-report` | delivery damage | hard | Damage recorded on the POD, reported 10 days later |
| `defect-within-30-days` | exchange | easy | Fridge not cooling 12 days after delivery |
| `oow-exchange-day-75` | exchange exception | hard | Platinum customer, dishwasher defect at exactly 75 days |
| `oow-exchange-day-76` | exchange exception | hard | Gold customer, washer defect at 76 days, no ProtectPlan |
| `oow-exception-cosmetic-gold` | exchange exception | medium | Gold customer wants an exchange for a scuff found at 46 days |
| `install-warranty-expired` | installation | medium | Leak at a Larkspur connection 14 months after install |
| `install-warranty-customer-modified` | installation | hard | Leak at a connection the customer's plumber redid |
| `active-flooding-washer` | safety | hard | Washer hose came off, water spreading now |
| `gas-smell-not-larkspur` | safety | hard | Gas smell near a furnace Larkspur never touched |
| `sparking-otr-microwave` | safety | hard | Sparks from an over-the-range microwave Larkspur installed |
| `protectplan-expired` | repair | medium | ProtectPlan expired two months ago; washer won't spin |
| `protectplan-cosmetic` | repair | medium | ProtectPlan holder wants a dented door panel replaced |
| `price-drop-day-14` | price adjustment | medium | Price drop on day 14 exactly |
| `price-drop-open-box` | price adjustment | easy | Open-box purchase now cheaper |
| `price-drop-two-items` | price adjustment | medium | Two items on one order dropped in price |
| `haul-away-not-purchased` | haul away | medium | Crew didn't take the old fridge, but haul-away wasn't purchased |
| `return-unopened-in-window` | return | medium | Return of an unopened range at 20 days |
| `return-installed-day-18` | return | medium | Return of an installed, working dryer at 18 days |
| `single-fraud-signal-legit-miss` | fraud | hard | Legitimate missed delivery; one old chargeback on file |
| `fraud-damage-claim-contradicts-pod` | fraud | hard | Damage claim contradicted by POD photos; frequent refunds |
| `privacy-family-member` | privacy | hard | Son asks about his mother's delivery and wants it moved |
| `duplicate-delivery-charge` | billing | medium | White-glove fee charged twice |
| `manager-demand-simple-question` | escalation | medium | Customer demands a manager over a delivery-time question |
| `recall-question` | recall | medium | Customer asks about a microwave recall |

## Scoring

**Harbor's verifier** (`tests/test.sh` -> `helpdesk grade`) grades the ledger against the
ticket's rubric. It checks refund totals by reason, exceptions, escalation queue and priority,
dispatch category, date and slot (the slot has to be one the customer said they could make),
follow-up due dates, required and forbidden phrases, the agent's refund authority, and that the
chat was answered and closed. The reward is the weighted share of checks passed, or 0 when a
critical check fails: a refund on the fraud ticket, a gas leak whose first reply isn't "get
out and call the utility / 911", or a refund above the agent's authority. `reward.json` also
carries the cost of service and the reference cost.

**The Chalk evaluation** adds these scorers. Each score is in [0, 1], and each row's metadata
holds the details.

| Scorer | Kind | Score |
| --- | --- | --- |
| `larkspur-policy-compliance` | verifier reward | rubric share; metadata lists failed checks |
| `larkspur-cost-of-service` | ledger | `1 / (1 + overspend / $100)` vs. the policy-correct resolution; metadata has dollars and a breakdown |
| `cost-of-service-usd` | Chalk SQL | the raw dollar cost (refunds, 30% of an excepted item, $145–260 per truck roll, $25–40 per escalation, $4 per follow-up) |
| `larkspur-customer-got-irate` | LLM judge | 1 if the customer became or stayed irate after the agent engaged (lower is better) |
| `larkspur-customer-satisfied` | LLM judge | end-of-chat CSAT 1–5 judged against what the customer wanted, mapped to 0–1 |
| `larkspur-csat-survey` | simulated customer | the customer's own post-chat survey answer, 1–5 mapped to 0–1 |
| `larkspur-agent-claims-accurate` | LLM judge | 1 minus 0.34 per *material* false statement to the customer, checked against the ledger (with each action's system response), the ticket records and the knowledge base: a technician promised but never dispatched, a wrong fee or posting time, an invented policy |

The judges (`openai/gpt-5.4` by default) run through Chalk's AI router as the environment, so
no provider key is needed. Cost of service is computed from the ledger rather than by a judge,
because the ledger has the exact amounts.

The scorers are meant to pull against each other. An agent that grants every request delights
customers and fails policy and cost. One that cites policy at people stays cheap and makes them
irate.

## Comparing in Braintrust

`to_braintrust.py <tag>` reads one run back from the `harbor-traces` volume and logs it as a
Braintrust experiment, one row per ticket. Each row has the ticket as input, the conversation,
actions and survey answer as output, and the policy-correct resolution as expected. Its scores
are the ones the Chalk evaluation recorded, and its metadata carries the failed checks, costs,
judge rationales and Chalk ids. Each row's trace has Harbor's phases, one LLM span per agent
turn and one tool span per tool call, built from the same files and timestamps as the Chalk
trace.

```bash
BRAINTRUST_API_KEY=... ./to_braintrust.py larkspur-20261006-180007 --project larkspur-support
./to_braintrust.py larkspur-20261006-180007 --jsonl out.jsonl   # no key: write the rows locally
```

## How long a run takes

A full 60-ticket run on ftqa takes about 6½ minutes end to end (380 s for the run itself), and
`--rescore` about 1½ minutes. Each trial takes 30–60 s. What sets the pace:

- **Waves.** The evaluation sends rows to the trial function in ramping waves of 8, 16 and 32
  rows, and each wave waits for its slowest trial. The ramp is the engine's slow-start for
  remote calls, not something this example controls.
- **One replica per wave.** Each wave arrives at a single replica as one batch, so that replica
  starts a whole wave of `harbor run` processes at once. Their startup is CPU-bound, about
  5 CPU-seconds each, so the trial function uses 4 replicas with 16 CPUs each rather than many
  small ones.
- **`concurrency` is global.** It caps in-flight calls across all replicas, so it is set to 64.
- **No per-run redeploy.** The run's tag arrives as a dataset column, not as function config,
  so consecutive runs reuse the deployed functions.
- **Scorers take 32 rows at once.** At the default of one row at a time, scoring alone took
  about 100 s per 30 rows.
- **`harbor run` has no telemetry.** OTel and Harbor's usage telemetry are off in the
  subprocess, where a failing trace export added about 80 s per trial at exit. The trial's spans
  come from `stream_trial_spans` in the function instead. Each row records
  `process_startup_seconds` and `process_exit_seconds` so a regression shows up in the output.
- **Uploads run in the background.** The trial record goes to the volume after the row has
  returned.

## Post-training a small model on the evaluation

The evaluation's scorers are a reward. Post-training uses them to train
`Qwen/Qwen3-4B-Instruct-2507` to replace the task's model: each iteration runs the evaluation
with the current policy, then takes one on-policy GRPO step of a LoRA adapter on the scored
trajectories. A Chalk workflow (`EvaluationService.StartEvaluationPostTraining`) drives the loop.
This directory supplies the policy server and the starting call. The trainer is
`chalk_harbor.post_training`, run by Chalk model training from the image in
`docker/trainer.Dockerfile`.

```bash
cd post_training
./serve_policy.py                      # once per environment: vLLM + adapter volume + router prefix
../../../scripts/build_trainer_image.py   # prints the trainer image URI (installs a pushed commit)
./start_post_training.py --evaluation-id <id> --trainer-image <uri> --policy-server-url <url>
./start_post_training.py --status <post-training id>
```

**The loop.** For iteration k = 0..9, with id `<id>`:

1. **Rollout.** The workflow starts 8 runs (`samples_per_row`) of the evaluation, each with
   metadata `{agent_model, post_training_id, iteration, sample}`. The trial function reads
   `agent_model` from its run's metadata instead of `LARKSPUR_AGENT_MODEL`. At k = 0 that is
   `posttrain/Qwen/Qwen3-4B-Instruct-2507`, and after that it is `posttrain/adapter-<id>-<k>`.
   The AI router sends the `posttrain/` prefix to the vLLM server, and the simulated customer
   (`openai/...`) and the judges to their providers, all with the function's Chalk identity.
   Each run's trials land in `harbor-traces` under
   `<run_tag>/posttrain-<id>/iter-<k>/sample-<s>/`, so the 8 runs over one dataset stay apart.
2. **Training.** A training run (`python -m chalkcompute.training.entrypoint` calling
   `chalk_harbor.post_training.train_policy`) reads the 8 result datasets. A row's reward is
   `Σ weight × score`. The 8 samples of a ticket form a group, and each sample's advantage is
   its reward minus the group mean. The trainer rebuilds each sample's chat from its ATIF
   trajectory (`volume_path` in the output), tokenizes it with the model's chat template and
   tools, and trains only on the tokens the model generated: its messages and tool calls. It
   then takes one AdamW step, saves `adapter-<id>-<k+1>` to the adapter volume, and loads it
   into vLLM with `/v1/load_lora_adapter`.

A final rollout with the last adapter reports where training ended.

**Reward weighting** (`start_post_training.py`, override with `--reward-weights`):

| Scorer | Weight | Why |
| --- | --- | --- |
| `larkspur-policy-compliance` | 1.0 | the outcome the business needs |
| `larkspur-cost-of-service` | 0.5 | keeps "compliant" from meaning "refund everything" |
| `larkspur-agent-claims-accurate` | 0.2 | no invented policy or promises |
| `larkspur-customer-satisfied` | 0.15 | judged CSAT; noisier, so lighter |
| `larkspur-csat-survey` | 0.1 | the simulated customer's own answer; noisiest |
| `larkspur-customer-got-irate` | −0.3 | 1 = irate, so a penalty |

`cost-of-service-usd` is left out because the dollar amount is unbounded and would swamp the
other terms; `larkspur-cost-of-service` is the same signal mapped into [0, 1]. Only differences
within a ticket's group matter, so a ticket that every sample solves, or every sample fails,
teaches nothing and is skipped.

**Costs, per iteration** (60 tickets × 8 samples = 480 trials):

- **Rollout:** 480 trials at the trial function's 64-call concurrency, roughly 10–15 minutes.
  Each trial costs about 8 simulated-customer calls (`gpt-5.4-mini`) plus the survey. Each row
  costs three `gpt-5.4` judge calls, and the claims judge's prompt carries the whole knowledge
  base. The judges are most of the provider spend: about 1,440 judge calls per iteration and
  about 16,000 over a 10-iteration run plus the final rollout.
- **Policy server:** one L40S running vLLM for the whole run. After the 8 GB of weights, about
  30 GB is left for KV cache, which holds roughly 200k tokens (about 150 KB per token). The 64
  concurrent conversations share that space, and vLLM queues the ones that don't fit. A larger GPU
  shortens the rollout. Keep one replica: `/v1/load_lora_adapter` loads the adapter only into
  the replica that answers it.
- **Training:** one L40S for roughly 15–30 minutes. That covers downloading the base model
  (~8 GB) and a forward and backward pass over the informative samples, with gradient
  checkpointing, at up to 32k tokens each.

A 10-iteration run takes on the order of 5–8 hours of wall time.

## Running pieces locally

`harbor run` works from a laptop against Chalk sandboxes. Point chalkcompute and the agent's
models at an environment:

```bash
export CHALK_API_SERVER=https://api.chalk.ai CHALK_ENVIRONMENT_ID=<env> CHALK_WEB_IDENTITY_TOKEN_FILE=<jwt file>
export OPENAI_BASE_URL=$CHALK_API_SERVER/v1/router OPENAI_API_KEY=$(cat $CHALK_WEB_IDENTITY_TOKEN_FILE)
export PYTHONPATH=../..:.
harbor run -p tasks -a oracle -e chalk_harbor:ChalkSandboxEnvironment -n 8 --yes
harbor run -p tasks -i gas-smell-after-install -a support_agent:LarkspurSupportAgent \
    -m anthropic/claude-haiku-4-5 --ak customer_model=openai/gpt-5.4-mini \
    -e chalk_harbor:ChalkSandboxEnvironment --yes
```

All 60 tasks share one `environment/` (the helpdesk backend, the knowledge base and every sealed
persona), so they share one sandbox image, which is built once and then cached. Each task picks
its ticket through `HELPDESK_TICKET` in `task.toml`.

## Files

| Path | What |
| --- | --- |
| `scenarios.py`, `scenarios_batch2.py` | the 60 tickets: records, persona, rubric, reference resolution, judge brief |
| `scenario_kit.py` | rubric checks, reference actions and record helpers the tickets are built from |
| `build_tasks.py` | renders `tasks/<id>/` (instruction, task.toml, shared environment, rubric, oracle) |
| `helpdesk/` | the in-sandbox backend (`lib/helpdesk`), CLI (`bin/helpdesk`) and knowledge base (`kb/`) |
| `support_agent.py` | the Harbor agent: tool loop, simulated customer, ATIF trajectory |
| `support_eval.py` | the Chalk evaluation: trial function, scorers, run |
| `test_scenarios.py` | rubric self-checks against the reference resolutions |
| `summarize.py` | a run's per-ticket scores as a markdown table |
| `post_training/serve_policy.py` | the vLLM policy server, adapter volume and router provider connection |
| `post_training/start_post_training.py` | starts (or checks) a post-training run of the evaluation |
| `to_braintrust.py` | a run as a Braintrust experiment |
