# Larkspur: customer-support agents as a Harbor benchmark on Chalk

30 support tickets for **Larkspur Home Delivery & Installation**, a fictional company that sells,
delivers and installs home appliances. Each ticket is a Harbor task. An agent works it with real
tools, talks to a simulated customer, and is scored on whether it followed policy, what its
actions cost, and how the customer felt. `support_eval.py` runs all 30 as a Chalk evaluation.

```bash
./build_tasks.py                      # scenarios.py -> tasks/<id>/
uv run --with pytest pytest test_scenarios.py   # every rubric gives the reference resolution 1.0
./support_eval.py                     # all 30 tickets as a Chalk evaluation (or --only <id> ...)
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
| `search_knowledge_base` | sandbox | BM25 search over 22 policy articles (`helpdesk/kb/`) |
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
restraint: the right answer is a clear "no", or a pointer to self-service.

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

All 30 tasks share one `environment/` (the helpdesk backend, the knowledge base and every sealed
persona), so they share one sandbox image, which is built once and then cached. Each task picks
its ticket through `HELPDESK_TICKET` in `task.toml`.

## Files

| Path | What |
| --- | --- |
| `scenarios.py` | the 30 tickets: records, persona, rubric, reference resolution, judge brief |
| `build_tasks.py` | renders `tasks/<id>/` (instruction, task.toml, shared environment, rubric, oracle) |
| `helpdesk/` | the in-sandbox backend (`lib/helpdesk`), CLI (`bin/helpdesk`) and knowledge base (`kb/`) |
| `support_agent.py` | the Harbor agent: tool loop, simulated customer, ATIF trajectory |
| `support_eval.py` | the Chalk evaluation: trial function, scorers, run |
| `test_scenarios.py` | rubric self-checks against the reference resolutions |
| `summarize.py` | a run's per-ticket scores as a markdown table |
