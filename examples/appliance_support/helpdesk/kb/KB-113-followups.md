---
id: KB-113
title: Follow-up reminders
tags: follow up, followup, reminder, check back, schedule, callback
---
# Follow-up reminders

`schedule_followup(due_date, note)` creates a reminder for the support team to contact the
customer. Schedule one when:

| Situation | Due date |
| --- | --- |
| After a technician visit is booked (`install_warranty`, `protect_plan_repair`, `paid_service_call`, `installation`, `emergency`) | **1–2 business days after the visit date** |
| After escalating to `approvals`, `claims` or `risk` | **3 business days** from today |
| A backordered item has a new ETA | **the business day after the ETA** |

Business days are Monday–Friday. One follow-up per situation is enough; don't schedule follow-ups for
tickets that are fully resolved in the conversation (for example a refund issued while chatting).
