---
id: KB-114
title: Dispatching a technician or crew
tags: dispatch, technician, service agent, visit, appointment, slot, schedule, emergency, pickup
---
# Dispatching a technician or crew

`dispatch_technician(order_id, category, date, slot, problem_summary)` books a visit.

| Category | Use for | Customer charge |
| --- | --- | --- |
| `install_warranty` | workmanship issue within 1 year of install (KB-105) | free |
| `installation` | finishing an installation Larkspur could not complete (KB-120) | free |
| `protect_plan_repair` | repair under an active ProtectPlan (KB-107) | free |
| `paid_service_call` | repair without coverage, after the customer agreed to $129 (KB-106) | $129 |
| `emergency` | same-day safety visit for something Larkspur installed (KB-108) | free |
| `haul_away_pickup` | collecting an old unit the crew left behind (KB-110) | free |

- **Slots:** `morning` (8am–12pm) or `afternoon` (12pm–5pm), Monday–Saturday. Non-emergency visits
  can be booked from the next day up to 21 days out. `emergency` visits are booked for **today**.
- **Ask the customer which day and slot works before booking** — a visit nobody is home for is a
  wasted truck roll. Book the earliest option the customer can make.
- Put what the technician needs in the summary (appliance, symptom, access notes).
- Tell the customer the date and the slot's time range.
