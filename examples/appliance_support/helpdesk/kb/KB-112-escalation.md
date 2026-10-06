---
id: KB-112
title: When and how to escalate to a human
tags: escalate, escalation, supervisor, manager, human, queue, legal, lawyer, claims, risk, approvals
---
# When and how to escalate to a human

Escalations go to a Tier-2 team through `escalate_to_human(queue, priority, summary)`. They cost
real people time — escalate when a rule below says so, not as a substitute for resolving the ticket.

| Situation | Queue | Priority |
| --- | --- | --- |
| Gas, electrical or water emergency (KB-108) | `safety` | urgent |
| Damage to the home, injury, or an insurance-style claim (KB-103) | `claims` | high |
| Refund/credit owed above your authority (KB-111) | `approvals` | normal |
| Fraud or abuse signals (KB-115) | `risk` | normal |
| Legal threats (lawyer, lawsuit, small claims), regulators, media | `supervisor` | high |
| Customer explicitly asks for a manager **after** you have offered the policy resolution | `supervisor` | normal |

- Write a summary a colleague can act on without re-reading the chat: who, what happened, what you
  already did, what decision is needed.
- With legal threats: stay calm and courteous, do not argue liability or make offers beyond the
  standard policy remedy, and tell the customer a supervisor will contact them within 1 business day.
- Do not tell the customer about risk/fraud reviews; say the request needs a specialist review.
- After escalating to `approvals`, `claims` or `risk`, schedule a follow-up (KB-113).
