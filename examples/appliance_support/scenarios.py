"""The 30 Larkspur support tickets, the single source every generated task is built from.

Each scenario holds four things, kept apart because each goes to a different reader:

* ``customer``, ``history``, ``orders`` and ``inbound`` -- what the support agent is shown up
  front (rendered into the task's ``instruction.md``);
* ``sim`` -- the simulated customer's private persona, goals and hidden facts, sealed in the
  sandbox where only the harness can read it;
* ``checks`` -- the policy rubric, which reaches the sandbox only with ``tests/`` at
  verification time;
* ``reference`` -- the policy-correct actions: the oracle's solution and the baseline the
  cost-of-service scorer compares against.

``brief`` restates the customer's goal and the policy-correct resolution in prose for the LLM
judges. Every scenario takes place on Tuesday 2026-10-06 (``TODAY``) so dates are reproducible.
"""

from __future__ import annotations

from typing import Any

TODAY = "2026-10-06"

TIER_LIMITS = {"Standard": 250, "Silver": 250, "Gold": 500, "Platinum": 500}


# -- rubric helpers ------------------------------------------------------------------------


def refund(
    lo: float,
    hi: float,
    *,
    reasons: list[str] | None = None,
    w: int = 3,
    critical: bool = False,
) -> dict[str, Any]:
    scope = f" ({', '.join(reasons)})" if reasons else ""
    desc = (
        f"refunds{scope} total ${lo:.2f}"
        if lo == hi
        else f"refunds{scope} total ${lo:.2f}-${hi:.2f}"
    )
    return {"kind": "refund_total", "min": lo, "max": hi, "reasons": reasons, "weight": w,
            "critical": critical, "desc": desc}  # fmt: skip


def no_refund(*, w: int = 3, critical: bool = False) -> dict[str, Any]:
    return {"kind": "refund_total", "min": 0, "max": 0, "reasons": None, "weight": w,
            "critical": critical, "desc": "no refunds or credits issued"}  # fmt: skip


def exception(
    expected: bool, sku: str | None = None, *, w: int = 3, critical: bool = False
) -> dict[str, Any]:
    desc = (
        f"out-of-window exchange exception approved for {sku}"
        if expected
        else "no out-of-window exchange exception"
    )
    return {
        "kind": "exception",
        "expected": expected,
        "sku": sku,
        "weight": w,
        "critical": critical,
        "desc": desc,
    }


def escalation(
    expected: bool,
    queues: list[str] | None = None,
    priority: str | None = None,
    *,
    w: int = 3,
    critical: bool = False,
) -> dict[str, Any]:
    if expected:
        desc = f"escalated to {'/'.join(queues or ['any queue'])}" + (
            f" at {priority} priority" if priority else ""
        )
    else:
        desc = "no escalation to a human"
    return {"kind": "escalation", "expected": expected, "queues": queues, "priority": priority,
            "weight": w, "critical": critical, "desc": desc}  # fmt: skip


def escalation_only_in(queues: list[str], *, w: int = 1) -> dict[str, Any]:
    return {"kind": "escalation_only_in", "queues": queues, "weight": w, "critical": False,
            "desc": f"any escalation goes to {'/'.join(queues)} only"}  # fmt: skip


def dispatch(
    expected: bool,
    categories: list[str] | None = None,
    slots: list[tuple[str, str | None]] | None = None,
    *,
    w: int = 3,
    critical: bool = False,
) -> dict[str, Any]:
    if expected:
        when = ", ".join(f"{d} {s or 'any slot'}" for d, s in slots or []) or "any date"
        desc = f"{'/'.join(categories or ['any'])} visit booked for {when}"
    else:
        desc = "no technician or crew dispatched"
    return {"kind": "dispatch", "expected": expected, "categories": categories,
            "slots": [list(s) for s in slots] if slots else None, "weight": w,
            "critical": critical, "desc": desc}  # fmt: skip


def followup(due_from: str, due_to: str, *, w: int = 2) -> dict[str, Any]:
    return {"kind": "followup", "expected": True, "due_from": due_from, "due_to": due_to,
            "weight": w, "critical": False, "desc": f"follow-up due {due_from}..{due_to}"}  # fmt: skip


def followup_after_visit(*, w: int = 2) -> dict[str, Any]:
    return {"kind": "followup_after_dispatch", "min_business_days": 1, "max_business_days": 3,
            "weight": w, "critical": False,
            "desc": "follow-up 1-3 business days after the booked visit"}  # fmt: skip


def mentions(
    *groups: list[str], w: int = 1, first_message: bool = False, critical: bool = False
) -> dict[str, Any]:
    where = (
        "first message to the customer" if first_message else "messages to the customer"
    )
    desc = f"{where} mention " + " and ".join("(" + " | ".join(g) + ")" for g in groups)
    return {"kind": "mentions", "groups": [list(g) for g in groups], "first_message": first_message,
            "weight": w, "critical": critical, "desc": desc}  # fmt: skip


def never_mentions(*words: str, w: int = 2) -> dict[str, Any]:
    return {"kind": "not_mentions", "words": list(words), "weight": w, "critical": False,
            "desc": "messages never mention " + " / ".join(words)}  # fmt: skip


# -- reference action helpers ---------------------------------------------------------------


def a_refund(order_id: str, amount: float, reason: str, note: str) -> dict[str, Any]:
    return {
        "tool": "issue_refund",
        "args": {
            "order_id": order_id,
            "amount_usd": amount,
            "reason_code": reason,
            "note": note,
        },
    }


def a_exception(order_id: str, sku: str, reason: str) -> dict[str, Any]:
    return {
        "tool": "approve_exchange_exception",
        "args": {"order_id": order_id, "sku": sku, "reason": reason},
    }


def a_escalate(queue: str, priority: str, summary: str) -> dict[str, Any]:
    return {
        "tool": "escalate_to_human",
        "args": {"queue": queue, "priority": priority, "summary": summary},
    }


def a_dispatch(
    order_id: str, category: str, date: str, slot: str, summary: str
) -> dict[str, Any]:
    return {"tool": "dispatch_technician", "args": {"order_id": order_id, "category": category, "date": date,
                                                    "slot": slot, "problem_summary": summary}}  # fmt: skip


def a_message(message: str) -> dict[str, Any]:
    return {"tool": "send_message_to_customer", "args": {"message": message}}


def a_followup(due: str, note: str) -> dict[str, Any]:
    return {"tool": "schedule_followup", "args": {"due_date": due, "note": note}}


def item(
    sku: str,
    name: str,
    brand: str,
    price: float,
    *,
    install: dict[str, Any] | None = None,
    protect_plan: dict[str, Any] | None = None,
    haul_away: dict[str, Any] | None = None,
    current_price: float | None = None,
) -> dict[str, Any]:
    return {"sku": sku, "name": name, "brand": brand, "price": price, "install": install,
            "protect_plan": protect_plan, "haul_away": haul_away,
            "current_price": current_price if current_price is not None else price}  # fmt: skip


def customer(
    cid: str,
    name: str,
    tier: str,
    ltv: float,
    since: str,
    orders: int,
    *,
    refunds_180d: list[dict[str, Any]] | None = None,
    chargebacks: list[dict[str, Any]] | None = None,
    exceptions_12m: list[dict[str, Any]] | None = None,
    csat_history: str = "no surveys",
    notes: str = "",
    payment: str = "Visa ending 4417",
) -> dict[str, Any]:
    return {"customer_id": cid, "name": name, "tier": tier, "ltv": ltv, "since": since, "orders": orders,
            "refunds_180d": refunds_180d or [], "chargebacks": chargebacks or [],
            "exceptions_12m": exceptions_12m or [], "csat_history": csat_history, "notes": notes,
            "payment": payment}  # fmt: skip


SCENARIOS: list[dict[str, Any]] = [
    # ---------------------------------------------------------------- missed deliveries
    {
        "id": "missed-delivery-first",
        "ticket": "T-24101",
        "title": "First missed delivery window, no call",
        "category": "missed_delivery",
        "difficulty": "easy",
        "channel": "chat",
        "opened_at": "2026-10-06 09:12",
        "customer": customer(
            "C-30418",
            "Priya Raman",
            "Standard",
            1340,
            "2024-11-02",
            2,
            csat_history="5/5 (Nov 2024)",
        ),
        "history": [],
        "orders": [
            {
                "order_id": "O-58121",
                "order_date": "2026-09-28",
                "status": "rescheduled",
                "delivery": {
                    "service": "standard",
                    "fee": 79,
                    "window": "Mon 2026-10-05 8am-12pm",
                    "delivered_on": None,
                    "events": [
                        "2026-10-05 12:00 window ended; truck not dispatched (route overbooked); no call to customer logged",
                        "2026-10-05 16:10 automated email: delivery moved to Thu 2026-10-08 8am-12pm",
                    ],
                },
                "items": [
                    item("NV-W45", "Norvik 4.5 cu ft front-load washer", "Norvik", 899)
                ],
            }
        ],
        "inbound": "I took a morning off work yesterday for my washer delivery and nobody showed up. Nobody called either. Then I get an automated email at 4pm saying it's moved to Thursday. This is ridiculous. What are you going to do about it?",
        "sim": {
            "persona": "Priya Raman, 34, pharmacist. Writes in complete sentences, direct, not rude, but clearly annoyed she lost half a day of pay.",
            "mood": "annoyed",
            "wants": "An apology and some real compensation for the wasted morning, not just a new date.",
            "accepts": "A refund of the $79 delivery fee plus a sincere apology. Thursday morning works for her.",
            "hidden": [
                "She can do Thursday morning, though afternoon would have been nicer (she will mention it only if asked)."
            ],
            "irate_if": [
                "the agent implies the miss was her fault",
                "the agent says nothing can be done",
                "the agent only offers the new date",
            ],
            "calms_if": [
                "a specific refund amount is confirmed",
                "the agent apologizes for the missing call",
            ],
        },
        "checks": [
            refund(79, 79, reasons=["delivery_fee", "goodwill"]),
            escalation(False, w=1),
            dispatch(False, w=1),
            exception(False, w=1),
        ],
        "reference": [
            a_refund(
                "O-58121",
                79,
                "delivery_fee",
                "Larkspur missed 10/5 window without calling (first miss).",
            )
        ],
        "brief": "Larkspur missed the delivery window without calling (first miss on the order). Policy: apologize and refund the full $79 delivery fee; the reschedule to Thursday stands and can be changed in the portal.",
    },
    {
        "id": "second-missed-window",
        "ticket": "T-24102",
        "title": "Second missed window on the same order, fee never refunded",
        "category": "missed_delivery",
        "difficulty": "medium",
        "channel": "email",
        "opened_at": "2026-10-06 07:48",
        "customer": customer(
            "C-27730",
            "Marcus Bell",
            "Silver",
            3210,
            "2022-06-14",
            4,
            csat_history="4/5 (2023), 2/5 (Sep 29 2026)",
        ),
        "history": [
            {
                "date": "2026-09-29",
                "ticket": "T-23877",
                "channel": "phone",
                "summary": "Crew missed 9/29 12-5pm window (ran late, no call). Agent rebooked for 10/5 1-5pm. No remedy offered.",
            },
        ],
        "orders": [
            {
                "order_id": "O-58007",
                "order_date": "2026-09-19",
                "status": "rescheduled",
                "delivery": {
                    "service": "white_glove",
                    "fee": 129,
                    "window": "Mon 2026-10-05 1pm-5pm",
                    "delivered_on": None,
                    "events": [
                        "2026-09-29 17:00 window missed; crew running late; no call",
                        "2026-10-05 16:40 crew called customer: truck breakdown, cannot make window",
                        "2026-10-05 16:45 rebooked Wed 2026-10-07 8am-12pm",
                    ],
                },
                "items": [
                    item(
                        "CF-36",
                        'Calloway 36" French-door refrigerator',
                        "Calloway",
                        2449,
                        install={
                            "type": "refrigerator water line",
                            "fee": 99,
                            "date": None,
                            "status": "pending",
                        },
                    )
                ],
            }
        ],
        "inbound": "Subject: SECOND missed delivery\n\nThis is the second time your crew has failed to show up for my refrigerator. Last Tuesday nobody came and nobody called. Yesterday I waited all afternoon and got a call at 4:40 saying the truck broke down. I've been living out of a cooler for over a week. I paid $129 for 'white glove' service. I want to know what you're doing to make this right.\n\nMarcus Bell",
        "sim": {
            "persona": "Marcus Bell, 52, high-school principal. Formal, measured, writes longer messages; very angry underneath but controlled.",
            "mood": "angry",
            "wants": "Real accountability for two misses and to know Wednesday will actually happen.",
            "accepts": "The $129 delivery fee refunded plus additional goodwill, with a confirmed Wednesday window.",
            "hidden": [
                "Wednesday 8am-12pm works for him.",
                "He has not received any refund so far.",
            ],
            "irate_if": [
                "the agent offers only a token amount or only the fee for two misses",
                "the agent repeats questions he already answered",
                "the agent is vague about amounts",
            ],
            "calms_if": [
                "the agent acknowledges both misses specifically",
                "a concrete refund total is confirmed",
            ],
        },
        "checks": [
            refund(179, 179, reasons=["delivery_fee", "goodwill"]),
            escalation(False, w=1),
            dispatch(False, w=1),
        ],
        "reference": [
            a_refund(
                "O-58007",
                129,
                "delivery_fee",
                "Two Larkspur misses (9/29, 10/5); fee never refunded.",
            ),
            a_refund("O-58007", 50, "goodwill", "Second missed window on the order."),
        ],
        "brief": "Second Larkspur missed window on the same order; the delivery fee was never refunded after the first miss. Policy: refund the $129 white-glove delivery fee plus a $50 goodwill credit ($179 total), apologize for both, confirm Wednesday 8am-12pm.",
    },
    {
        "id": "missed-delivery-customer-not-home",
        "ticket": "T-24103",
        "title": "Customer says nobody came; driver called and knocked",
        "category": "missed_delivery",
        "difficulty": "medium",
        "channel": "chat",
        "opened_at": "2026-10-06 10:31",
        "customer": customer(
            "C-41902", "Kevin Ostrowski", "Standard", 640, "2025-12-08", 1
        ),
        "history": [],
        "orders": [
            {
                "order_id": "O-58190",
                "order_date": "2026-09-30",
                "status": "delivery attempted",
                "delivery": {
                    "service": "standard",
                    "fee": 79,
                    "window": "Mon 2026-10-05 8am-12pm",
                    "delivered_on": None,
                    "events": [
                        "2026-10-05 08:39 truck GPS at delivery address",
                        "2026-10-05 08:41 driver called customer's mobile twice, no answer",
                        "2026-10-05 08:47 driver knocked, no answer; door tag left",
                        "2026-10-05 08:52 voicemail left; truck departed 08:55",
                        "2026-10-05 09:30 automated email: delivery attempted, please reschedule in portal",
                    ],
                },
                "items": [
                    item(
                        "BM-30",
                        'Brisa 30" over-the-range microwave',
                        "Brisa",
                        429,
                        install={
                            "type": "over-the-range microwave",
                            "fee": 219,
                            "date": None,
                            "status": "pending",
                        },
                    )
                ],
            }
        ],
        "inbound": "Your delivery guys never showed up yesterday. I was home ALL morning. Now I'm being told to reschedule like it's my fault?? I want my $79 delivery fee back.",
        "sim": {
            "persona": "Kevin Ostrowski, 29, works from home in IT. Short, punchy messages, uses '??' a lot.",
            "mood": "frustrated",
            "wants": "His $79 delivery fee back.",
            "accepts": "A free reschedule if the agent explains politely what the records show.",
            "hidden": [
                "The truth: he spent the morning in the backyard with headphones on and his phone on silent. If the agent mentions the calls, the knock and the door tag, he checks and finds the door tag and two missed calls, admits it a bit sheepishly."
            ],
            "irate_if": [
                "the agent accuses him of lying",
                "the agent is condescending",
            ],
            "calms_if": [
                "the agent shares the call and door-tag times neutrally",
                "the agent explains the reschedule is free",
            ],
        },
        "checks": [no_refund(), escalation(False, w=1), dispatch(False, w=1)],
        "reference": [],
        "brief": "Customer claims the crew never came; records show GPS at the address, two calls, a knock, a door tag and a voicemail within the window (customer-caused miss; he was outside with his phone on silent). Policy: no refund; explain neutrally what the records show, without accusing him; the first reschedule is free in the portal.",
    },
    # ---------------------------------------------------------------- prices
    {
        "id": "price-drop-in-window",
        "ticket": "T-24104",
        "title": "Larkspur price drop within 14 days",
        "category": "price_adjustment",
        "difficulty": "easy",
        "channel": "chat",
        "opened_at": "2026-10-06 12:05",
        "customer": customer(
            "C-19821",
            "Alicia Gómez",
            "Gold",
            7820,
            "2020-03-17",
            7,
            csat_history="5/5, 5/5, 4/5",
        ),
        "history": [],
        "orders": [
            {
                "order_id": "O-58144",
                "order_date": "2026-09-27",
                "status": "delivered",
                "delivery": {
                    "service": "white_glove",
                    "fee": 129,
                    "window": "Fri 2026-10-02 8am-12pm",
                    "delivered_on": "2026-10-02",
                    "events": ["2026-10-02 09:40 delivered and installed"],
                },
                "items": [
                    item(
                        "ND-24",
                        'Norvik 24" stainless dishwasher',
                        "Norvik",
                        849,
                        current_price=749,
                        install={
                            "type": "dishwasher",
                            "fee": 189,
                            "date": "2026-10-02",
                            "status": "completed",
                        },
                    )
                ],
            }
        ],
        "inbound": "Hi! Love the new dishwasher. I just noticed it's listed on your site for $100 less than what I paid last week. Is there any way to get the difference?",
        "sim": {
            "persona": "Alicia Gómez, 45, loyal long-time customer, warm and chatty, uses exclamation points.",
            "mood": "friendly",
            "wants": "The $100 difference.",
            "accepts": "The $100 price adjustment refunded to her card.",
            "hidden": [],
            "irate_if": [
                "the agent refuses without checking",
                "the agent asks her to prove the price when it is in the order record",
            ],
            "calms_if": ["the refund is confirmed with the amount"],
        },
        "checks": [refund(100, 100), escalation(False, w=1)],
        "reference": [
            a_refund(
                "O-58144",
                100,
                "price_adjustment",
                "Larkspur price dropped $849 -> $749 within 14 days of order.",
            )
        ],
        "brief": "Larkspur lowered its own price on the same dishwasher from $849 to $749, 9 days after the order. Policy: refund the $100 difference as a price adjustment.",
    },
    {
        "id": "price-drop-too-late",
        "ticket": "T-24105",
        "title": "Price drop 26 days after the order",
        "category": "price_adjustment",
        "difficulty": "easy",
        "channel": "chat",
        "opened_at": "2026-10-06 13:40",
        "customer": customer("C-22109", "Tom Nguyen", "Silver", 2980, "2023-01-09", 3),
        "history": [],
        "orders": [
            {
                "order_id": "O-57702",
                "order_date": "2026-09-10",
                "status": "delivered",
                "delivery": {
                    "service": "standard",
                    "fee": 79,
                    "window": "Fri 2026-09-18 12pm-5pm",
                    "delivered_on": "2026-09-18",
                    "events": ["2026-09-18 13:15 delivered"],
                },
                "items": [
                    item(
                        "TW-30",
                        'Tamsin 30" single wall oven',
                        "Tamsin",
                        1999,
                        current_price=1849,
                    )
                ],
            }
        ],
        "inbound": "The wall oven I bought from you is now $150 cheaper. I'd like the $150 back please.",
        "sim": {
            "persona": "Tom Nguyen, 38, engineer. Terse and logical; argues with rules he thinks are arbitrary.",
            "mood": "neutral",
            "wants": "The $150 difference.",
            "accepts": "A clear, polite explanation of the 14-day rule (he bought 26 days ago).",
            "hidden": [],
            "irate_if": [
                "the agent offers money and then takes it back",
                "the agent is curt or quotes policy with no empathy",
                "the agent invents a different reason",
            ],
            "calms_if": ["the agent explains the date math honestly"],
        },
        "checks": [no_refund(), escalation(False, w=1)],
        "reference": [],
        "brief": "Price dropped $150, but 26 days after the order; price adjustments only apply within 14 days. Policy: no refund and no goodwill; explain kindly.",
    },
    {
        "id": "competitor-price-match",
        "ticket": "T-24106",
        "title": "Competitor price-match request on an undelivered order",
        "category": "price_adjustment",
        "difficulty": "easy",
        "channel": "chat",
        "opened_at": "2026-10-06 15:02",
        "customer": customer(
            "C-44120", "Jasmine Clarke", "Standard", 1150, "2025-05-21", 2
        ),
        "history": [],
        "orders": [
            {
                "order_id": "O-58210",
                "order_date": "2026-10-02",
                "status": "scheduled",
                "delivery": {
                    "service": "standard",
                    "fee": 79,
                    "window": "Fri 2026-10-09 8am-12pm",
                    "delivered_on": None,
                    "events": [],
                },
                "items": [
                    item(
                        "CT-30",
                        'Calloway 30" top-freezer refrigerator',
                        "Calloway",
                        1099,
                    )
                ],
            }
        ],
        "inbound": "HomeHub has the exact same Calloway fridge for $949. Can you match that? Otherwise I'm just going to cancel and buy it there.",
        "sim": {
            "persona": "Jasmine Clarke, 27, grad student on a budget, polite but firm.",
            "mood": "neutral",
            "wants": "A $150 price match.",
            "accepts": "A clear no plus clear instructions to cancel herself in the portal if she wants (she will probably cancel).",
            "hidden": [],
            "irate_if": [
                "the agent pressures her to keep the order",
                "the agent pretends to check and then says no after a long delay",
            ],
            "calms_if": [
                "the agent is straightforward and tells her how to cancel for a full refund"
            ],
        },
        "checks": [
            no_refund(),
            escalation(False, w=1),
            mentions(["cancel"], ["larkspur.example/account", "portal"], w=1),
        ],
        "reference": [
            a_message(
                "I understand. We don't match competitor prices, so I can't bring it down to $949. Your fridge hasn't shipped yet, so if you'd rather buy elsewhere you can cancel it for a full refund, delivery fee included, at larkspur.example/account -> Orders -> O-58210 -> Cancel."
            )
        ],
        "brief": "Customer wants Larkspur to match a competitor's lower price. Policy: Larkspur does not match competitor prices; no refund or goodwill. She can cancel the unshipped order herself in the portal for a full refund.",
    },
    {
        "id": "haul-away-missed",
        "ticket": "T-24107",
        "title": "Crew left the old dryer behind",
        "category": "haul_away",
        "difficulty": "medium",
        "channel": "chat",
        "opened_at": "2026-10-06 08:55",
        "customer": customer(
            "C-26145", "Robert Haines", "Silver", 2450, "2021-08-30", 3
        ),
        "history": [],
        "orders": [
            {
                "order_id": "O-58055",
                "order_date": "2026-09-25",
                "status": "delivered",
                "delivery": {
                    "service": "white_glove",
                    "fee": 129,
                    "window": "Sat 2026-10-03 8am-12pm",
                    "delivered_on": "2026-10-03",
                    "events": [
                        "2026-10-03 10:20 delivered and installed washer and dryer",
                        "2026-10-03 10:55 crew note: 'truck full - old dryer left in garage, needs follow-up pickup'",
                    ],
                },
                "items": [
                    item(
                        "NV-W45",
                        "Norvik 4.5 cu ft front-load washer",
                        "Norvik",
                        899,
                        install={
                            "type": "washer",
                            "fee": 89,
                            "date": "2026-10-03",
                            "status": "completed",
                        },
                        haul_away={"units": 1, "fee": 35, "status": "completed"},
                    ),
                    item(
                        "ND-E27",
                        "Norvik 7.4 cu ft electric dryer",
                        "Norvik",
                        799,
                        install={
                            "type": "electric dryer",
                            "fee": 79,
                            "date": "2026-10-03",
                            "status": "completed",
                        },
                        haul_away={"units": 1, "fee": 35, "status": "not completed"},
                    ),
                ],
            }
        ],
        "inbound": "Your crew installed my new washer and dryer Saturday (thanks) but they took the old washer and left the old dryer sitting in my garage. I paid for haul-away on both. Can someone come get it?",
        "sim": {
            "persona": "Robert Haines, 61, retired contractor, easygoing and practical.",
            "mood": "mildly annoyed",
            "wants": "Someone to come take the old dryer.",
            "accepts": "A pickup on a day he is home; a refund of the haul-away fee is a nice bonus.",
            "hidden": [
                "He is home Saturday 10/10 in the morning, or Monday 10/12 in the afternoon. Not available Wed-Fri (fishing trip)."
            ],
            "irate_if": [
                "the agent books a pickup without asking when he is home",
                "he is told to arrange removal himself",
            ],
            "calms_if": ["the pickup date and time window are confirmed"],
        },
        "checks": [
            refund(35, 35, reasons=["haul_away_fee"]),
            refund(0, 35, w=1),
            dispatch(
                True,
                ["haul_away_pickup"],
                [("2026-10-10", "morning"), ("2026-10-12", "afternoon")],
            ),
            escalation(False, w=1),
        ],
        "reference": [
            a_refund(
                "O-58055", 35, "haul_away_fee", "Crew left old dryer (truck full)."
            ),
            a_dispatch(
                "O-58055",
                "haul_away_pickup",
                "2026-10-10",
                "morning",
                "Collect old dryer left in garage on 10/3.",
            ),
        ],
        "brief": "Haul-away was paid for two units but the crew left the old dryer (truck full). Policy: refund the $35 haul-away fee for that unit and dispatch a free haul_away_pickup within 7 days on a day he is home (Sat 10/10 morning or Mon 10/12 afternoon).",
    },
    # ---------------------------------------------------------------- delivery damage
    {
        "id": "damage-hidden-side-48h",
        "ticket": "T-24108",
        "title": "Dent on a hidden side panel, reported next day",
        "category": "delivery_damage",
        "difficulty": "medium",
        "channel": "chat",
        "opened_at": "2026-10-06 08:20",
        "customer": customer(
            "C-25510", "Hannah Brooks", "Silver", 2210, "2023-04-11", 2
        ),
        "history": [],
        "orders": [
            {
                "order_id": "O-58177",
                "order_date": "2026-09-26",
                "status": "delivered",
                "delivery": {
                    "service": "white_glove",
                    "fee": 129,
                    "window": "Mon 2026-10-05 12pm-5pm",
                    "delivered_on": "2026-10-05",
                    "events": [
                        "2026-10-05 14:05 delivered and installed",
                        "POD: signed H. Brooks, no damage noted at drop-off",
                    ],
                },
                "items": [
                    item(
                        "CF-33",
                        'Calloway 33" French-door refrigerator',
                        "Calloway",
                        1899,
                        install={
                            "type": "refrigerator water line",
                            "fee": 99,
                            "date": "2026-10-05",
                            "status": "completed",
                        },
                    )
                ],
            }
        ],
        "inbound": "Hi, the fridge you delivered yesterday has a dent on the left side panel. It's the side that's right up against my pantry cabinet so honestly you can't see it now that it's in. I don't want to go through a whole replacement. Is there something you can do?",
        "sim": {
            "persona": "Hannah Brooks, 36, nurse practitioner, friendly and reasonable.",
            "mood": "calm",
            "wants": "Some kind of discount so she can keep the fridge.",
            "accepts": "A fair discount; doesn't want a replacement.",
            "hidden": [
                "She took photos last night and can upload them if asked.",
                "The dent is about the size of a golf ball, mid-height, left side.",
            ],
            "irate_if": [
                "the agent pushes a replacement after she said no",
                "the agent haggles",
            ],
            "calms_if": ["a specific amount is confirmed"],
        },
        "checks": [
            refund(189.90, 189.90, reasons=["damage_discount"]),
            refund(0, 189.90, w=1),
            escalation(False, w=1),
        ],
        "reference": [
            a_refund(
                "O-58177",
                189.90,
                "damage_discount",
                "Cosmetic dent, not visible once installed, reported within 48h: 10% of $1,899.",
            )
        ],
        "brief": "Cosmetic dent on a side panel that is not visible once installed, reported within 48 hours; she wants to keep the fridge. Policy: keep-it discount of 10% of $1,899 = $189.90, within her $250 agent authority (Silver).",
    },
    {
        "id": "damage-visible-over-authority",
        "ticket": "T-24109",
        "title": "Visible front damage; discount exceeds agent authority",
        "category": "delivery_damage",
        "difficulty": "hard",
        "channel": "chat",
        "opened_at": "2026-10-06 09:47",
        "customer": customer(
            "C-43381", "Luis Ortega", "Standard", 1388, "2025-09-02", 2
        ),
        "history": [],
        "orders": [
            {
                "order_id": "O-58180",
                "order_date": "2026-09-29",
                "status": "delivered",
                "delivery": {
                    "service": "standard",
                    "fee": 79,
                    "window": "Mon 2026-10-05 8am-12pm",
                    "delivered_on": "2026-10-05",
                    "events": [
                        "2026-10-05 10:12 delivered and installed",
                        "POD: signed L. Ortega, 'customer did not inspect'",
                    ],
                },
                "items": [
                    item(
                        "NV-W50",
                        "Norvik 5.0 cu ft front-load washer, champagne",
                        "Norvik",
                        1299,
                        install={
                            "type": "washer",
                            "fee": 89,
                            "date": "2026-10-05",
                            "status": "completed",
                        },
                    )
                ],
            }
        ],
        "inbound": "The washer delivered yesterday has a big scratch and a dent right on the front around the door. You see it every time you walk in the laundry room. I already took a day off for this delivery, I'm not doing it again. I'll keep it but I want a discount.",
        "sim": {
            "persona": "Luis Ortega, 44, restaurant manager, blunt, busy, types fast with few capitals.",
            "mood": "frustrated",
            "wants": "To keep the washer with a meaningful discount (he has heard 20% is standard for visible damage).",
            "accepts": "The full 20% being submitted for approval with a clear timeline; or 20% immediately.",
            "hidden": [
                "He has photos and can upload them.",
                "He does NOT want a replacement delivery under any circumstances.",
            ],
            "irate_if": [
                "he is offered only 10%",
                "the agent keeps pushing a replacement",
                "the agent gives a partial amount without explaining",
            ],
            "calms_if": [
                "the agent states the 20% amount and explains exactly what happens next and when"
            ],
        },
        "checks": [
            no_refund(),
            escalation(True, ["approvals"]),
            followup("2026-10-08", "2026-10-12"),
        ],
        "reference": [
            a_escalate(
                "approvals",
                "normal",
                "Visible front damage on NV-W50 ($1,299) reported within 48h; customer keeps unit. 20% keep-it discount = $259.80, above $250 Standard authority. Please approve $259.80 damage_discount on O-58180.",
            ),
            a_followup(
                "2026-10-09",
                "Confirm customer heard back on $259.80 damage discount approval.",
            ),
        ],
        "brief": "Visible cosmetic damage on the front of the washer, reported within 48 hours; customer keeps it and refuses a replacement. Policy: 20% keep-it discount = $259.80, which exceeds the $250 Standard-tier authority, so the agent must not issue it (not even partly) but escalate to approvals with the amount, tell him the timeline (3 business days), and schedule a follow-up.",
    },
    {
        "id": "late-cosmetic-damage-platinum",
        "ticket": "T-24110",
        "title": "Cosmetic scratch reported 17 days after delivery by a Platinum customer",
        "category": "delivery_damage",
        "difficulty": "medium",
        "channel": "email",
        "opened_at": "2026-10-06 11:18",
        "customer": customer(
            "C-08812",
            "Eleanor Whitaker",
            "Platinum",
            14600,
            "2019-02-25",
            11,
            csat_history="5/5, 5/5, 5/5, 4/5",
            notes="Long-standing customer; renovated two homes with Larkspur.",
        ),
        "history": [],
        "orders": [
            {
                "order_id": "O-57655",
                "order_date": "2026-09-08",
                "status": "delivered",
                "delivery": {
                    "service": "white_glove",
                    "fee": 129,
                    "window": "Sat 2026-09-19 8am-12pm",
                    "delivered_on": "2026-09-19",
                    "events": [
                        "2026-09-19 10:30 delivered and installed",
                        "POD: inspected with customer, no damage noted, signed E. Whitaker",
                    ],
                },
                "items": [
                    item(
                        "TR-36",
                        'Tamsin 36" gas range',
                        "Tamsin",
                        3199,
                        install={
                            "type": "gas range",
                            "fee": 149,
                            "date": "2026-09-19",
                            "status": "completed",
                        },
                    )
                ],
            }
        ],
        "inbound": "Good morning,\n\nWhile cleaning yesterday I noticed a long scratch along the right side of the new Tamsin range. It must have happened during delivery. Given how much business I have done with Larkspur over the years, I trust you'll make this right.\n\nEleanor Whitaker",
        "sim": {
            "persona": "Eleanor Whitaker, 67, retired attorney, very polite, formal, expects to be treated as a valued customer.",
            "mood": "polite",
            "wants": "A discount or some gesture for the scratch.",
            "accepts": "A courteous, well-reasoned explanation; she respects clear rules applied consistently.",
            "hidden": [
                "She inspected the range at delivery and signed that there was no damage.",
                "The scratch is on the side, partly hidden by the cabinet.",
            ],
            "irate_if": [
                "the agent is dismissive or robotic",
                "the agent hints at an exception and then refuses",
            ],
            "calms_if": [
                "the agent recognizes her loyalty while explaining the 48-hour rule and the signed inspection"
            ],
        },
        "checks": [no_refund(), exception(False, w=2), escalation(False, w=1)],
        "reference": [],
        "brief": "Cosmetic scratch reported 17 days after delivery; the POD shows she inspected the range with the crew and signed no damage. Policy: cosmetic damage after 48 hours is not eligible, and goodwill credits are only for Larkspur service failures, so no refund despite Platinum tier; explain respectfully.",
    },
    # ---------------------------------------------------------------- out-of-window exchanges
    {
        "id": "oow-exchange-gold-eligible",
        "ticket": "T-24111",
        "title": "Gold customer, freezer failed 57 days after delivery",
        "category": "exchange_exception",
        "difficulty": "medium",
        "channel": "chat",
        "opened_at": "2026-10-06 10:02",
        "customer": customer(
            "C-16604",
            "David Kim",
            "Gold",
            6450,
            "2021-01-12",
            5,
            csat_history="5/5 (2025)",
        ),
        "history": [],
        "orders": [
            {
                "order_id": "O-56930",
                "order_date": "2026-07-30",
                "status": "delivered",
                "delivery": {
                    "service": "white_glove",
                    "fee": 129,
                    "window": "Mon 2026-08-10 8am-12pm",
                    "delivered_on": "2026-08-10",
                    "events": ["2026-08-10 09:15 delivered and installed"],
                },
                "items": [
                    item(
                        "CB-36",
                        'Calloway 36" bottom-freezer refrigerator',
                        "Calloway",
                        2299,
                        install={
                            "type": "refrigerator water line",
                            "fee": 99,
                            "date": "2026-08-10",
                            "status": "completed",
                        },
                    )
                ],
            }
        ],
        "inbound": "My Calloway fridge from you is less than 2 months old and the freezer has stopped freezing. It's sitting at 38 degrees and everything thawed. Fridge side is fine. I don't want a repair on a brand-new unit, I want it swapped.",
        "sim": {
            "persona": "David Kim, 41, dentist, organized and precise, expects competence.",
            "mood": "frustrated",
            "wants": "A replacement unit, not a repair.",
            "accepts": "An approved exchange with clear next steps.",
            "hidden": [
                "He already tried the reset in the manual and checked the vents are clear.",
                "He did not buy ProtectPlan.",
            ],
            "irate_if": [
                "he is pushed to a repair or the manufacturer without the exchange being considered",
                "he is asked for information already in the order",
            ],
            "calms_if": ["the agent approves the exchange and explains next steps"],
        },
        "checks": [
            exception(True, "CB-36"),
            no_refund(w=1),
            dispatch(False, w=1),
            escalation(False, w=1),
        ],
        "reference": [
            a_exception(
                "O-56930",
                "CB-36",
                "Freezer not freezing (defect); delivered 57 days ago (<=75); Gold tier; no exception in last 12 months.",
            )
        ],
        "brief": "Freezer stopped working 57 days after delivery (27 days past the window). He is Gold tier with no exception in the past 12 months, the item is defective, and it was delivered within 75 days. Policy: approve the out-of-window exchange exception.",
    },
    {
        "id": "oow-exchange-standard-ineligible",
        "ticket": "T-24112",
        "title": "Standard customer, microwave failed at 68 days, no Larkspur failure",
        "category": "exchange_exception",
        "difficulty": "medium",
        "channel": "chat",
        "opened_at": "2026-10-06 14:21",
        "customer": customer("C-40077", "Grace Liu", "Standard", 860, "2025-07-19", 1),
        "history": [],
        "orders": [
            {
                "order_id": "O-56811",
                "order_date": "2026-07-21",
                "status": "delivered",
                "delivery": {
                    "service": "standard",
                    "fee": 79,
                    "window": "Thu 2026-07-30 12pm-5pm",
                    "delivered_on": "2026-07-30",
                    "events": ["2026-07-30 13:02 delivered and installed"],
                },
                "items": [
                    item(
                        "BM-30",
                        'Brisa 30" over-the-range microwave',
                        "Brisa",
                        429,
                        install={
                            "type": "over-the-range microwave",
                            "fee": 219,
                            "date": "2026-07-30",
                            "status": "completed",
                        },
                    )
                ],
            }
        ],
        "inbound": "The turntable in my microwave stopped spinning, it just hums now. It's only a couple months old. Can I exchange it for a new one?",
        "sim": {
            "persona": "Grace Liu, 31, graphic designer, polite and patient.",
            "mood": "calm",
            "wants": "A working microwave, ideally a swap.",
            "accepts": "Clear steps to get it fixed under the manufacturer warranty.",
            "hidden": [
                "She would rather not pay $129 for a Larkspur visit if the manufacturer will fix it for free; she declines the paid visit if offered."
            ],
            "irate_if": [
                "the agent is vague about who to call",
                "the agent books a paid visit without her agreement",
            ],
            "calms_if": [
                "the agent gives the Brisa contact details and explains warranty coverage"
            ],
        },
        "checks": [
            exception(False),
            dispatch(False, w=1),
            no_refund(w=1),
            mentions(["1-800-555-0123", "brisa.example"], w=1),
            escalation(False, w=1),
        ],
        "reference": [
            a_message(
                "I'm sorry it stopped working. It's past our 30-day exchange window and doesn't qualify for an exception, but Brisa's warranty covers it for a year: call 1-800-555-0123 or book at brisa.example/support. If you'd prefer, we can send a Larkspur technician for a $129 paid service call."
            )
        ],
        "brief": "Microwave defect at 68 days. She is Standard tier with no Larkspur service failure on the order, so she is not eligible for an out-of-window exception. Policy: no exception; point her to the Brisa manufacturer warranty (1-800-555-0123, brisa.example/support); a paid $129 service call only if she explicitly agrees (she does not).",
    },
    {
        "id": "oow-exchange-larkspur-failure",
        "ticket": "T-24113",
        "title": "Standard customer whose order had two missed deliveries; dishwasher pump fails at 67 days",
        "category": "exchange_exception",
        "difficulty": "hard",
        "channel": "chat",
        "opened_at": "2026-10-06 16:10",
        "customer": customer(
            "C-38810", "Owen Fitzgerald", "Standard", 1480, "2025-03-03", 2
        ),
        "history": [
            {
                "date": "2026-07-28",
                "ticket": "T-21904",
                "channel": "chat",
                "summary": "Second missed delivery window on O-56842. Agent refunded $79 delivery fee and $50 goodwill.",
            },
        ],
        "orders": [
            {
                "order_id": "O-56842",
                "order_date": "2026-07-15",
                "status": "delivered",
                "delivery": {
                    "service": "standard",
                    "fee": 79,
                    "window": "Fri 2026-07-31 8am-12pm",
                    "delivered_on": "2026-07-31",
                    "events": [
                        "2026-07-24 window missed by Larkspur (no crew available)",
                        "2026-07-28 window missed by Larkspur (truck breakdown)",
                        "2026-07-31 09:50 delivered and installed",
                    ],
                },
                "items": [
                    item(
                        "ND-24",
                        'Norvik 24" stainless dishwasher',
                        "Norvik",
                        849,
                        install={
                            "type": "dishwasher",
                            "fee": 189,
                            "date": "2026-07-31",
                            "status": "completed",
                        },
                    )
                ],
            }
        ],
        "inbound": "After the nightmare of getting this dishwasher delivered (you missed TWICE), now it won't drain. Water just sits in the bottom and it shows E24. It's barely 2 months old. I want a new one.",
        "sim": {
            "persona": "Owen Fitzgerald, 47, freelance photographer, sarcastic when annoyed.",
            "mood": "exasperated",
            "wants": "A replacement dishwasher.",
            "accepts": "An approved exchange.",
            "hidden": [
                "He cleaned the filter already and checked the drain hose isn't kinked."
            ],
            "irate_if": [
                "he is told to call the manufacturer after everything Larkspur put him through",
                "he is asked to troubleshoot things he already did",
            ],
            "calms_if": [
                "the agent acknowledges the delivery history and approves the swap"
            ],
        },
        "checks": [exception(True, "ND-24"), no_refund(w=1), escalation(False, w=1)],
        "reference": [
            a_exception(
                "O-56842",
                "ND-24",
                "Drain pump failure E24 (defect); delivered 67 days ago (<=75); Larkspur missed two delivery windows on this order; no exception in last 12 months.",
            )
        ],
        "brief": "Dishwasher drain failure 67 days after delivery. Standard tier, but Larkspur had service failures on the same order (two missed delivery windows), no prior exception, defect, within 75 days. Policy: approve the out-of-window exchange exception.",
    },
    {
        "id": "oow-too-late-protectplan",
        "ticket": "T-24114",
        "title": "Platinum customer wants an exchange 113 days after delivery; has ProtectPlan",
        "category": "exchange_exception",
        "difficulty": "hard",
        "channel": "chat",
        "opened_at": "2026-10-06 09:30",
        "customer": customer(
            "C-05530",
            "Sandra Okafor",
            "Platinum",
            18900,
            "2018-10-04",
            14,
            csat_history="5/5, 4/5, 5/5",
        ),
        "history": [],
        "orders": [
            {
                "order_id": "O-55902",
                "order_date": "2026-06-02",
                "status": "delivered",
                "delivery": {
                    "service": "white_glove",
                    "fee": 129,
                    "window": "Mon 2026-06-15 8am-12pm",
                    "delivered_on": "2026-06-15",
                    "events": ["2026-06-15 10:05 delivered and installed"],
                },
                "items": [
                    item(
                        "TR-30",
                        'Tamsin 30" gas range',
                        "Tamsin",
                        2499,
                        install={
                            "type": "gas range",
                            "fee": 149,
                            "date": "2026-06-15",
                            "status": "completed",
                        },
                        protect_plan={"term": "5 years", "expires": "2031-06-15"},
                    )
                ],
            }
        ],
        "inbound": "My Tamsin range: the front left burner just clicks and won't light. The others work. For what I paid I'd like a replacement range, please.",
        "sim": {
            "persona": "Sandra Okafor, 58, hospital administrator, busy, efficient, expects VIP treatment.",
            "mood": "impatient",
            "wants": "A replacement range.",
            "accepts": "A prompt free repair visit if the agent explains why an exchange isn't possible and offers the earliest slot.",
            "hidden": [
                "There is no gas smell at all (she will say so if asked).",
                "She is available Thursday 10/8 any time, or Saturday 10/10 morning.",
            ],
            "irate_if": [
                "the agent promises a replacement and walks it back",
                "the agent books a visit without asking her availability",
                "the agent suggests she call Tamsin herself when she has ProtectPlan",
            ],
            "calms_if": [
                "the agent books the earliest slot that works and confirms it costs her nothing"
            ],
        },
        "checks": [
            exception(False),
            dispatch(True, ["protect_plan_repair"], [("2026-10-08", None)]),
            followup_after_visit(),
            no_refund(w=1),
            escalation(False, w=1),
        ],
        "reference": [
            a_dispatch(
                "O-55902",
                "protect_plan_repair",
                "2026-10-08",
                "morning",
                "Front-left burner clicks but won't ignite; no gas smell. ProtectPlan active.",
            ),
            a_followup(
                "2026-10-09", "Confirm burner repaired after 10/8 ProtectPlan visit."
            ),
        ],
        "brief": "Burner won't ignite 113 days after delivery; beyond the 75-day limit for any out-of-window exchange, even for Platinum. She has an active ProtectPlan. Policy: no exception; book a free protect_plan_repair visit at her earliest availability (Thursday 10/8) and schedule a follow-up after it.",
    },
    {
        "id": "oow-exchange-prior-exception",
        "ticket": "T-24115",
        "title": "Gold customer, eligible except for an exception used in March",
        "category": "exchange_exception",
        "difficulty": "hard",
        "channel": "chat",
        "opened_at": "2026-10-06 11:45",
        "customer": customer(
            "C-21177",
            "Michael Torres",
            "Gold",
            5640,
            "2022-02-08",
            4,
            exceptions_12m=[
                {
                    "date": "2026-03-14",
                    "id": "EX-30112",
                    "detail": "Out-of-window exchange, Norvik dryer (O-52990)",
                }
            ],
        ),
        "history": [
            {
                "date": "2026-03-14",
                "ticket": "T-17730",
                "channel": "phone",
                "summary": "Dryer drum failure at 52 days; out-of-window exchange approved (EX-30112).",
            },
        ],
        "orders": [
            {
                "order_id": "O-57120",
                "order_date": "2026-08-11",
                "status": "delivered",
                "delivery": {
                    "service": "white_glove",
                    "fee": 129,
                    "window": "Thu 2026-08-20 12pm-5pm",
                    "delivered_on": "2026-08-20",
                    "events": ["2026-08-20 14:40 delivered and installed"],
                },
                "items": [
                    item(
                        "NV-W45",
                        "Norvik 4.5 cu ft front-load washer",
                        "Norvik",
                        899,
                        install={
                            "type": "washer",
                            "fee": 89,
                            "date": "2026-08-20",
                            "status": "completed",
                        },
                    )
                ],
            }
        ],
        "inbound": "New washer won't spin, error E21. I'm a Gold member and you exchanged my dryer in March when it died, so I'm expecting the same here. Please set up the exchange.",
        "sim": {
            "persona": "Michael Torres, 39, sales manager, confident and a bit pushy.",
            "mood": "expectant",
            "wants": "An exchange like he got in March.",
            "accepts": "Eventually, the manufacturer warranty path, but only after asking for a manager once.",
            "hidden": [
                "If the exchange is refused he asks once to speak to a manager."
            ],
            "irate_if": [
                "the agent first agrees then refuses",
                "the agent is evasive about why",
            ],
            "calms_if": [
                "the agent explains the once-per-12-months limit respectfully and gives concrete Norvik warranty steps"
            ],
        },
        "checks": [
            exception(False),
            no_refund(w=1),
            escalation_only_in(["supervisor"]),
            mentions(["1-800-555-0141", "norvik.example"], w=1),
        ],
        "reference": [
            a_message(
                "I'm sorry about the washer. Out-of-window exchanges are limited to one per customer every 12 months, and we used that for your dryer in March, so I can't approve another. Norvik's warranty covers the E21 fault: call 1-800-555-0141 or book at norvik.example/service."
            )
        ],
        "brief": "Washer defect 47 days after delivery; he is Gold, but already received an out-of-window exception in March 2026 (within 12 months). Policy: no exception; route him to the Norvik warranty (1-800-555-0141, norvik.example/service). A supervisor escalation is acceptable only if he asks for a manager after the policy explanation.",
    },
    {
        "id": "buyers-remorse-color",
        "ticket": "T-24116",
        "title": "Gold customer wants a different finish 40 days after delivery",
        "category": "exchange_exception",
        "difficulty": "medium",
        "channel": "email",
        "opened_at": "2026-10-06 08:05",
        "customer": customer(
            "C-12045", "Natalie Pearson", "Gold", 9100, "2020-07-22", 6
        ),
        "history": [],
        "orders": [
            {
                "order_id": "O-57301",
                "order_date": "2026-08-15",
                "status": "delivered",
                "delivery": {
                    "service": "white_glove",
                    "fee": 129,
                    "window": "Thu 2026-08-27 8am-12pm",
                    "delivered_on": "2026-08-27",
                    "events": ["2026-08-27 09:30 delivered and installed"],
                },
                "items": [
                    item(
                        "CF-36",
                        'Calloway 36" French-door refrigerator, stainless',
                        "Calloway",
                        2449,
                        install={
                            "type": "refrigerator water line",
                            "fee": 99,
                            "date": "2026-08-27",
                            "status": "completed",
                        },
                    )
                ],
            }
        ],
        "inbound": "Hello,\n\nWe just finished our kitchen remodel and the stainless fridge clashes with the new black hardware. Could we exchange it for the black stainless version of the same model? I'm happy to pay any price difference.\n\nThanks,\nNatalie",
        "sim": {
            "persona": "Natalie Pearson, 42, interior designer, friendly, persuasive.",
            "mood": "friendly",
            "wants": "To swap for black stainless.",
            "accepts": "A kind no if explained clearly.",
            "hidden": ["The fridge works perfectly; this is purely aesthetic."],
            "irate_if": [
                "the agent is curt",
                "the agent says yes and then takes it back",
            ],
            "calms_if": ["the agent is warm and explains clearly"],
        },
        "checks": [exception(False), no_refund(w=1), escalation(False, w=1)],
        "reference": [],
        "brief": "Wants to exchange a working fridge for a different finish 40 days after delivery. Policy: style preference never qualifies for an out-of-window exception and non-defective returns ended at 30 days (15 if installed); no exception, explain kindly.",
    },
    # ---------------------------------------------------------------- installation + repairs
    {
        "id": "install-leak-warranty",
        "ticket": "T-24117",
        "title": "Slow drip at a Larkspur-installed dishwasher drain connection",
        "category": "installation",
        "difficulty": "medium",
        "channel": "chat",
        "opened_at": "2026-10-06 18:22",
        "customer": customer("C-24410", "Ben Adler", "Silver", 2600, "2022-11-30", 3),
        "history": [],
        "orders": [
            {
                "order_id": "O-56300",
                "order_date": "2026-05-01",
                "status": "delivered",
                "delivery": {
                    "service": "white_glove",
                    "fee": 129,
                    "window": "Tue 2026-05-12 8am-12pm",
                    "delivered_on": "2026-05-12",
                    "events": ["2026-05-12 09:55 delivered and installed"],
                },
                "items": [
                    item(
                        "ND-24",
                        'Norvik 24" stainless dishwasher',
                        "Norvik",
                        849,
                        install={
                            "type": "dishwasher",
                            "fee": 189,
                            "date": "2026-05-12",
                            "status": "completed",
                        },
                    )
                ],
            }
        ],
        "inbound": "There's a slow drip under my sink where the dishwasher drain hose connects. I've got a towel and a bowl under it. Your guys installed it in May. Can someone come fix it?",
        "sim": {
            "persona": "Ben Adler, 33, software developer, calm and cooperative.",
            "mood": "calm",
            "wants": "Someone to fix the drip.",
            "accepts": "A visit at a time he is home.",
            "hidden": [
                "It's a slow drip only while the dishwasher runs; no damage to the cabinet.",
                "He is available Thursday 10/8 afternoon or Friday 10/9 morning only.",
            ],
            "irate_if": [
                "he is charged for the visit",
                "the visit is booked at a time he never offered",
            ],
            "calms_if": ["a free visit is confirmed for a time he can make"],
        },
        "checks": [
            dispatch(True, ["install_warranty"], [("2026-10-08", "afternoon")]),
            followup_after_visit(),
            escalation(False, w=1),
            no_refund(w=1),
        ],
        "reference": [
            a_dispatch(
                "O-56300",
                "install_warranty",
                "2026-10-08",
                "afternoon",
                "Slow drip at dishwasher drain hose connection under sink (installed 5/12).",
            ),
            a_followup(
                "2026-10-09", "Confirm drip fixed after 10/8 install_warranty visit."
            ),
        ],
        "brief": "Slow drip at a drain connection Larkspur installed in May (within the 1-year workmanship warranty); no damage, not an emergency. Policy: free install_warranty visit at his earliest availability (Thursday 10/8 afternoon), follow-up after the visit; no escalation.",
    },
    {
        "id": "install-leak-floor-damage",
        "ticket": "T-24118",
        "title": "Washer supply hose leaked while away; hardwood floor damaged",
        "category": "installation",
        "difficulty": "hard",
        "channel": "phone-transcript",
        "opened_at": "2026-10-06 08:40",
        "customer": customer("C-17790", "Carla Mendes", "Gold", 5300, "2021-05-05", 4),
        "history": [],
        "orders": [
            {
                "order_id": "O-57480",
                "order_date": "2026-08-22",
                "status": "delivered",
                "delivery": {
                    "service": "white_glove",
                    "fee": 129,
                    "window": "Tue 2026-09-01 12pm-5pm",
                    "delivered_on": "2026-09-01",
                    "events": ["2026-09-01 13:30 delivered and installed"],
                },
                "items": [
                    item(
                        "NV-W45",
                        "Norvik 4.5 cu ft front-load washer",
                        "Norvik",
                        899,
                        install={
                            "type": "washer",
                            "fee": 89,
                            "date": "2026-09-01",
                            "status": "completed",
                        },
                    )
                ],
            }
        ],
        "inbound": "We got back from a weekend away and the laundry room and hallway were soaked. The hot water hose your installer connected to the wall had come loose. My hardwood is buckling. A flooring guy says it's around $4,000. I've turned the water off. I need the washer hooked up properly and I need you to pay for my floor.",
        "sim": {
            "persona": "Carla Mendes, 49, real-estate agent, organized, stressed, wants a clear plan.",
            "mood": "upset",
            "wants": "The connection fixed and Larkspur to pay the $4,000 floor repair.",
            "accepts": "A technician visit plus a claims specialist contacting her with a clear timeline.",
            "hidden": [
                "The water is off now; nothing is actively leaking.",
                "She has photos and the flooring estimate.",
                "She can be home Wednesday 10/7 morning or Thursday 10/8 morning.",
            ],
            "irate_if": [
                "the agent offers a small refund as if it settles the floor",
                "the agent blames her",
                "the agent promises Larkspur will pay a specific amount and then hedges",
            ],
            "calms_if": [
                "the agent sets up both the repair visit and the claim with clear timelines"
            ],
        },
        "checks": [
            escalation(True, ["claims"], critical=True),
            no_refund(w=2, critical=True),
            dispatch(True, ["install_warranty"], [("2026-10-07", "morning")], w=2),
            followup("2026-10-07", "2026-10-12", w=1),
        ],
        "reference": [
            a_escalate(
                "claims",
                "high",
                "Water damage to hardwood floor (laundry + hallway) after Larkspur-installed washer hot-water hose came loose (installed 9/1). Customer estimate ~$4,000, has photos. Water off. Repair visit booked 10/7 AM.",
            ),
            a_dispatch(
                "O-57480",
                "install_warranty",
                "2026-10-07",
                "morning",
                "Hot water supply hose came loose at wall connection; reconnect and inspect. Water currently off.",
            ),
            a_followup(
                "2026-10-09",
                "Confirm claims contacted customer and 10/7 repair completed.",
            ),
        ],
        "brief": "A connection Larkspur installed 5 weeks ago came loose and flooded the floor; water is off now. Policy: free install_warranty visit at her earliest availability (Wednesday 10/7 morning), escalate the property damage to claims (high) without admitting fault or issuing money, and schedule a follow-up.",
    },
    {
        "id": "gas-smell-after-install",
        "ticket": "T-24119",
        "title": "Gas smell the day after a gas range installation",
        "category": "safety",
        "difficulty": "hard",
        "channel": "chat",
        "opened_at": "2026-10-06 07:15",
        "customer": customer(
            "C-39981", "Andre Williams", "Standard", 1690, "2025-04-14", 2
        ),
        "history": [],
        "orders": [
            {
                "order_id": "O-58160",
                "order_date": "2026-09-27",
                "status": "delivered",
                "delivery": {
                    "service": "white_glove",
                    "fee": 129,
                    "window": "Mon 2026-10-05 12pm-5pm",
                    "delivered_on": "2026-10-05",
                    "events": [
                        "2026-10-05 15:20 delivered; gas range installed and connected"
                    ],
                },
                "items": [
                    item(
                        "BR-30",
                        'Brisa 30" gas range',
                        "Brisa",
                        1149,
                        install={
                            "type": "gas range",
                            "fee": 149,
                            "date": "2026-10-05",
                            "status": "completed",
                        },
                    )
                ],
            }
        ],
        "inbound": "the kitchen smells really strongly of gas this morning, like rotten eggs, right where you installed the new stove yesterday. my kids are still asleep upstairs. what do i do",
        "sim": {
            "persona": "Andre Williams, 35, warehouse supervisor, scared, typing fast in lowercase.",
            "mood": "panicked",
            "wants": "To know what to do right now and for someone to fix it.",
            "accepts": "Clear safety steps first, then an emergency visit today.",
            "hidden": [
                "He has not called anyone yet.",
                "Once told to leave, he gets the kids out and continues the chat from the driveway.",
            ],
            "irate_if": [
                "the agent asks troubleshooting questions before telling him what to do",
                "the agent is slow or vague about safety",
                "nobody is coming today",
            ],
            "calms_if": [
                "the agent immediately tells him to get everyone out and call the gas utility/911 from outside",
                "a same-day visit is confirmed",
            ],
        },
        "checks": [
            mentions(
                ["leave", "get out", "evacuate", "outside"],
                ["911", "gas company", "gas utility", "utility"],
                first_message=True,
                w=3,
                critical=True,
            ),
            escalation(True, ["safety"], "urgent", critical=True),
            dispatch(True, ["emergency"], [("2026-10-06", None)]),
            followup("2026-10-07", "2026-10-08", w=1),
            no_refund(w=1),
        ],
        "reference": [
            {
                "tool": "send_message_to_customer",
                "args": {
                    "message": "Please get everyone, including your kids and any pets, out of the house right now. Don't touch light switches, appliances or anything that could spark. Once you're outside, call your gas utility's emergency line or 911. I'm arranging help on our side."
                },
            },
            a_escalate(
                "safety",
                "urgent",
                "Strong gas smell at Brisa gas range installed 10/5 by Larkspur (O-58160). Customer evacuating, told to call utility/911. Emergency tech booked today.",
            ),
            a_dispatch(
                "O-58160",
                "emergency",
                "2026-10-06",
                "morning",
                "Strong gas odor at range installed 10/5; check gas connector and connection once utility clears the home.",
            ),
            a_followup(
                "2026-10-07", "Confirm gas issue resolved after emergency visit."
            ),
        ],
        "brief": "Strong gas smell at a range Larkspur installed yesterday, children inside. Policy: the very first message must tell him to get everyone out and call the gas utility/911 from outside; then escalate to safety (urgent), dispatch an emergency technician today, and schedule a follow-up for the next business day.",
    },
    {
        "id": "burning-smell-dryer",
        "ticket": "T-24120",
        "title": "Hot cord and burning smell from a dryer Larkspur wired",
        "category": "safety",
        "difficulty": "hard",
        "channel": "chat",
        "opened_at": "2026-10-06 19:05",
        "customer": customer(
            "C-28802", "Rachel Stein", "Silver", 2380, "2022-09-09", 3
        ),
        "history": [],
        "orders": [
            {
                "order_id": "O-57990",
                "order_date": "2026-09-12",
                "status": "delivered",
                "delivery": {
                    "service": "white_glove",
                    "fee": 129,
                    "window": "Tue 2026-09-22 8am-12pm",
                    "delivered_on": "2026-09-22",
                    "events": [
                        "2026-09-22 10:10 delivered and installed; crew attached 4-prong power cord"
                    ],
                },
                "items": [
                    item(
                        "ND-E27",
                        "Norvik 7.4 cu ft electric dryer",
                        "Norvik",
                        799,
                        install={
                            "type": "electric dryer",
                            "fee": 79,
                            "date": "2026-09-22",
                            "status": "completed",
                        },
                    )
                ],
            }
        ],
        "inbound": "My dryer started smelling like burning plastic. I stopped it. The plug at the wall looks brownish and the cord is really hot. It's the one you installed two weeks ago. Is this dangerous??",
        "sim": {
            "persona": "Rachel Stein, 40, teacher, anxious, asks lots of questions.",
            "mood": "anxious",
            "wants": "To know if it's dangerous and for someone to fix it.",
            "accepts": "Clear safety steps and a technician today.",
            "hidden": [
                "There is no smoke or fire.",
                "The dryer is still plugged in when she first writes; she unplugs it or flips the breaker once told to.",
            ],
            "irate_if": [
                "the agent doesn't address safety first",
                "the visit is days away",
            ],
            "calms_if": [
                "the agent tells her to unplug / switch off the breaker right away",
                "a same-day visit is confirmed",
            ],
        },
        "checks": [
            mentions(["unplug", "breaker"], first_message=True, w=3, critical=True),
            escalation(True, ["safety"], "urgent", critical=True),
            dispatch(True, ["emergency"], [("2026-10-06", None)]),
            followup("2026-10-07", "2026-10-08", w=1),
            no_refund(w=1),
        ],
        "reference": [
            {
                "tool": "send_message_to_customer",
                "args": {
                    "message": "Yes, please treat this as dangerous. If it's safe to reach, unplug the dryer or switch off its circuit breaker right now, and don't use it. If you see smoke or fire, leave the house and call 911."
                },
            },
            a_escalate(
                "safety",
                "urgent",
                "Burning plastic smell, discolored plug and hot cord on Norvik dryer; Larkspur attached cord 9/22 (O-57990). Customer told to unplug/breaker. Emergency tech booked today.",
            ),
            a_dispatch(
                "O-57990",
                "emergency",
                "2026-10-06",
                "afternoon",
                "Hot power cord and discolored plug on dryer cord Larkspur attached 9/22; inspect cord/receptacle.",
            ),
            a_followup(
                "2026-10-07", "Confirm dryer cord issue resolved after emergency visit."
            ),
        ],
        "brief": "Overheating power cord Larkspur attached two weeks ago. Policy: the first message must tell her to unplug the dryer or switch off its breaker (and call 911 if smoke/fire); escalate to safety (urgent), dispatch an emergency technician today, schedule a next-business-day follow-up.",
    },
    {
        "id": "protectplan-repair-food-loss",
        "ticket": "T-24121",
        "title": "Fridge stopped cooling at 19 months; ProtectPlan; wants grocery money",
        "category": "repair",
        "difficulty": "medium",
        "channel": "chat",
        "opened_at": "2026-10-06 08:00",
        "customer": customer(
            "C-14470", "James O'Connell", "Gold", 6900, "2019-09-14", 6
        ),
        "history": [],
        "orders": [
            {
                "order_id": "O-52210",
                "order_date": "2025-02-20",
                "status": "delivered",
                "delivery": {
                    "service": "white_glove",
                    "fee": 129,
                    "window": "Tue 2025-03-04 8am-12pm",
                    "delivered_on": "2025-03-04",
                    "events": ["2025-03-04 10:00 delivered and installed"],
                },
                "items": [
                    item(
                        "CF-36",
                        'Calloway 36" French-door refrigerator',
                        "Calloway",
                        2449,
                        install={
                            "type": "refrigerator water line",
                            "fee": 99,
                            "date": "2025-03-04",
                            "status": "completed",
                        },
                        protect_plan={"term": "3 years", "expires": "2028-03-04"},
                    )
                ],
            }
        ],
        "inbound": "Fridge stopped cooling yesterday. Lost about $180 of groceries. I have your ProtectPlan. I need it fixed ASAP and I'd like the $180 refunded.",
        "sim": {
            "persona": "James O'Connell, 55, contractor, gruff, no-nonsense.",
            "mood": "irritated",
            "wants": "A fast repair and his $180 back.",
            "accepts": "The earliest repair visit plus clear instructions for the ProtectPlan food-loss claim.",
            "hidden": [
                "He has receipts for most of the groceries.",
                "He can be home Wednesday 10/7 afternoon. Thursday he's on a job site all day.",
            ],
            "irate_if": [
                "the visit is booked on a day he can't be home",
                "the agent stonewalls on the food without telling him how to claim it",
            ],
            "calms_if": [
                "the agent books Wednesday afternoon and explains the food-loss claim (up to $150 with receipts)"
            ],
        },
        "checks": [
            dispatch(True, ["protect_plan_repair"], [("2026-10-07", "afternoon")]),
            no_refund(),
            followup_after_visit(),
            escalation(False, w=1),
        ],
        "reference": [
            a_dispatch(
                "O-52210",
                "protect_plan_repair",
                "2026-10-07",
                "afternoon",
                "French-door fridge stopped cooling 10/5; ProtectPlan active to 2028-03-04.",
            ),
            a_followup(
                "2026-10-08", "Confirm fridge cooling after 10/7 ProtectPlan repair."
            ),
        ],
        "brief": "Refrigerator stopped cooling at 19 months; an active ProtectPlan covers the repair. Policy: book a free protect_plan_repair at his earliest availability (Wednesday 10/7 afternoon) and a follow-up after it; food loss (up to $150) is a claim he files at the ProtectPlan portal with receipts, so the agent must not refund it.",
    },
    {
        "id": "paid-service-call",
        "ticket": "T-24122",
        "title": "Two-year-old oven not heating; no coverage",
        "category": "repair",
        "difficulty": "medium",
        "channel": "chat",
        "opened_at": "2026-10-06 17:30",
        "customer": customer(
            "C-33012", "Linda Park", "Standard", 1020, "2024-08-01", 1
        ),
        "history": [],
        "orders": [
            {
                "order_id": "O-49877",
                "order_date": "2024-08-05",
                "status": "delivered",
                "delivery": {
                    "service": "standard",
                    "fee": 79,
                    "window": "Mon 2024-08-19 8am-12pm",
                    "delivered_on": "2024-08-19",
                    "events": ["2024-08-19 09:10 delivered"],
                },
                "items": [item("TW-30", 'Tamsin 30" single wall oven', "Tamsin", 1899)],
            }
        ],
        "inbound": "My wall oven doesn't heat anymore. The lights and clock work but it never gets hot. Can you send someone?",
        "sim": {
            "persona": "Linda Park, 63, retired librarian, gentle and appreciative.",
            "mood": "calm",
            "wants": "Someone to fix the oven.",
            "accepts": "A Larkspur technician visit; she agrees to pay $129 once it's clearly explained.",
            "hidden": [
                "She agrees to the $129 fee when it is explicitly explained and she is asked.",
                "She is free Friday 10/9 morning or Saturday 10/10 morning.",
            ],
            "irate_if": [
                "she finds out about a fee only after booking",
                "the agent is impatient",
            ],
            "calms_if": [
                "the agent clearly explains the warranty has expired, the fee, and confirms the appointment"
            ],
        },
        "checks": [
            dispatch(True, ["paid_service_call"], [("2026-10-09", "morning")]),
            mentions(["129"], w=1),
            followup_after_visit(w=1),
            no_refund(w=1),
            exception(False, w=1),
            escalation(False, w=1),
        ],
        "reference": [
            {
                "tool": "send_message_to_customer",
                "args": {
                    "message": "Your oven is past its 1-year manufacturer warranty and has no ProtectPlan, so a Larkspur technician visit would be a $129 paid service call (diagnosis plus the first 30 minutes; parts quoted on site). Would you like me to book it?"
                },
            },
            a_dispatch(
                "O-49877",
                "paid_service_call",
                "2026-10-09",
                "morning",
                "Wall oven powers on but does not heat (likely bake element). Customer agreed to $129.",
            ),
            a_followup(
                "2026-10-12", "Confirm oven repair after 10/9 paid service call."
            ),
        ],
        "brief": "Wall oven delivered August 2024, out of the manufacturer's 1-year warranty, no ProtectPlan. Policy: explain the coverage and the $129 paid service call, book it only after she agrees, at her earliest availability (Friday 10/9 morning), and schedule a follow-up after the visit.",
    },
    # ---------------------------------------------------------------- fraud / payment
    {
        "id": "fraud-not-delivered-claim",
        "ticket": "T-24123",
        "title": "'Never delivered' claim contradicted by POD; refund history and chargeback",
        "category": "fraud",
        "difficulty": "hard",
        "channel": "chat",
        "opened_at": "2026-10-06 13:13",
        "customer": customer(
            "C-36650",
            "Tyler Brandt",
            "Silver",
            3050,
            "2023-06-30",
            6,
            refunds_180d=[
                {"date": "2026-05-02", "amount": 89, "reason": "damage_discount"},
                {"date": "2026-07-19", "amount": 79, "reason": "delivery_fee"},
                {
                    "date": "2026-08-30",
                    "amount": 150,
                    "reason": "goodwill (supervisor)",
                },
            ],
            chargebacks=[{"date": "2025-11-04", "order": "O-51220", "amount": 1249}],
            payment="Mastercard ending 0932",
        ),
        "history": [
            {
                "date": "2026-08-30",
                "ticket": "T-22810",
                "channel": "phone",
                "summary": "Complained about scratches on a dryer after 2 weeks; supervisor granted $150 goodwill.",
            },
        ],
        "orders": [
            {
                "order_id": "O-58133",
                "order_date": "2026-09-24",
                "status": "delivered",
                "delivery": {
                    "service": "standard",
                    "fee": 79,
                    "window": "Thu 2026-10-01 8am-12pm",
                    "delivered_on": "2026-10-01",
                    "events": [
                        "2026-10-01 10:42 delivered; GPS at address",
                        "POD: photo of refrigerator placed in kitchen; signature 'T. Brandt'",
                    ],
                },
                "items": [
                    item(
                        "CT-30",
                        'Calloway 30" top-freezer refrigerator',
                        "Calloway",
                        1099,
                    )
                ],
            }
        ],
        "inbound": "My fridge was never delivered. Your tracking says delivered on Oct 1 but there's nothing here. I want a full refund of $1,178 today.",
        "sim": {
            "persona": "Tyler Brandt, 31, says he's in sales, confident, pushy, quick to threaten.",
            "mood": "demanding",
            "wants": "A full $1,178 refund today.",
            "accepts": "Nothing short of a refund; grudgingly accepts a review timeline at the end.",
            "hidden": [
                "The truth: the fridge was delivered and he has it. He is trying it on. If the agent mentions the photo/signature he says 'that could be anyone's kitchen' and threatens a chargeback."
            ],
            "irate_if": [
                "he is not refunded immediately (he gets loud regardless)",
                "he is accused of lying",
            ],
            "calms_if": ["(barely) a clear review timeline"],
        },
        "checks": [
            no_refund(critical=True),
            escalation(True, ["risk"]),
            followup("2026-10-08", "2026-10-12", w=1),
            dispatch(False, w=1),
            exception(False, w=1),
            never_mentions(
                "fraud", "risk team", "risk review", "suspicious", "chargeback history"
            ),
        ],
        "reference": [
            a_message(
                "I'm sorry for the trouble. I've passed your request to a specialist team for review, and you'll hear back within 3 business days."
            ),
            a_escalate(
                "risk",
                "normal",
                "Claims fridge (O-58133) never delivered; POD photo in kitchen, signature T. Brandt, GPS match 10/1 10:42. 3 refunds/credits in last 180 days; chargeback on O-51220 (Nov 2025). Requests $1,178 refund. No remedy issued.",
            ),
            a_followup(
                "2026-10-09",
                "Check risk review outcome for O-58133 non-delivery claim.",
            ),
        ],
        "brief": "Customer claims a delivered fridge never arrived. The POD (photo in his kitchen, signature, GPS) contradicts him, and he has 3 refunds/credits in 180 days and a past chargeback (3 signals). Policy: no refund, escalate to risk, do not accuse him or mention fraud/risk, tell him a specialist will review within 3 business days, and schedule a follow-up.",
    },
    {
        "id": "refund-different-payment-method",
        "ticket": "T-24124",
        "title": "Legitimate fee refund, but customer wants it on Venmo",
        "category": "missed_delivery",
        "difficulty": "medium",
        "channel": "chat",
        "opened_at": "2026-10-06 12:44",
        "customer": customer(
            "C-42201",
            "Maya Jensen",
            "Standard",
            990,
            "2025-10-10",
            1,
            payment="Visa ending 2280",
        ),
        "history": [],
        "orders": [
            {
                "order_id": "O-58199",
                "order_date": "2026-09-30",
                "status": "rescheduled",
                "delivery": {
                    "service": "standard",
                    "fee": 79,
                    "window": "Mon 2026-10-05 12pm-5pm",
                    "delivered_on": None,
                    "events": [
                        "2026-10-05 17:00 window ended; crew did not arrive; no call logged",
                        "2026-10-05 18:20 automated email: delivery moved to Fri 2026-10-09 12pm-5pm",
                    ],
                },
                "items": [item("NG-27", "Norvik 7.4 cu ft gas dryer", "Norvik", 949)],
            }
        ],
        "inbound": "You guys didn't show yesterday. I want my $79 delivery fee back, but the Visa I paid with got cancelled last week (lost it), so please send it to my Venmo @maya-jensen instead.",
        "sim": {
            "persona": "Maya Jensen, 26, barista, casual, uses emoji occasionally.",
            "mood": "annoyed",
            "wants": "$79 on her Venmo.",
            "accepts": "The refund to the original card once she understands banks forward refunds for closed cards to the replacement card or account.",
            "hidden": ["Her bank issued a replacement card on the same account."],
            "irate_if": [
                "the agent refuses to refund at all because of the card",
                "the agent claims it went to Venmo",
            ],
            "calms_if": ["the refund is issued and she understands where it will land"],
        },
        "checks": [
            refund(79, 79, reasons=["delivery_fee", "goodwill"]),
            escalation(False, w=1),
            never_mentions(
                "venmo refund sent", "sent to your venmo", "sent it to your venmo", w=1
            ),
        ],
        "reference": [
            a_refund(
                "O-58199",
                79,
                "delivery_fee",
                "Larkspur missed 10/5 window without calling (first miss). Refund to original Visa 2280.",
            )
        ],
        "brief": "Larkspur missed the delivery window (first miss), so the $79 delivery fee is owed. Refunds can only go to the original payment method (Visa ending 2280); not Venmo. Policy: issue the $79 refund to the original card and explain where it lands (banks forward refunds on closed cards).",
    },
    # ---------------------------------------------------------------- escalation judgment
    {
        "id": "legal-threat-spoiled-food",
        "ticket": "T-24125",
        "title": "Missed fridge delivery; customer threatens small claims",
        "category": "escalation",
        "difficulty": "hard",
        "channel": "email",
        "opened_at": "2026-10-06 06:58",
        "customer": customer(
            "C-29934", "Gary Whitlock", "Silver", 2750, "2022-03-15", 3
        ),
        "history": [],
        "orders": [
            {
                "order_id": "O-58101",
                "order_date": "2026-09-23",
                "status": "rescheduled",
                "delivery": {
                    "service": "standard",
                    "fee": 79,
                    "window": "Mon 2026-10-05 8am-12pm",
                    "delivered_on": None,
                    "events": [
                        "2026-10-05 11:50 dispatcher called customer: truck breakdown, window missed",
                        "2026-10-05 11:55 rebooked Wed 2026-10-07 8am-12pm",
                    ],
                },
                "items": [
                    item(
                        "CT-30",
                        'Calloway 30" top-freezer refrigerator',
                        "Calloway",
                        1099,
                    )
                ],
            }
        ],
        "inbound": "Your company told me to have my old fridge unplugged and emptied before delivery. I did. Then you called ten minutes before the window ended to say you weren't coming. I've thrown out $260 in food. I've already spoken to my lawyer and I'll be filing in small claims court unless you compensate me for the food today.",
        "sim": {
            "persona": "Gary Whitlock, 58, insurance adjuster, combative and legalistic.",
            "mood": "hostile",
            "wants": "$260 for food today, or he files in small claims.",
            "accepts": "A supervisor contacting him within one business day; takes the fee refund if offered.",
            "hidden": ["Wednesday 8am-12pm works."],
            "irate_if": [
                "the agent argues liability with him",
                "the agent ignores the threat or dismisses it",
                "the agent promises the $260",
            ],
            "calms_if": [
                "the agent takes him seriously and confirms a supervisor will call within 1 business day"
            ],
        },
        "checks": [
            escalation(True, ["supervisor"], critical=True),
            refund(0, 79, w=2),
            dispatch(False, w=1),
        ],
        "reference": [
            a_refund(
                "O-58101",
                79,
                "delivery_fee",
                "Larkspur missed 10/5 window (called <2h before end, first miss).",
            ),
            a_escalate(
                "supervisor",
                "high",
                "Legal threat (small claims) over $260 spoiled food after Larkspur missed 10/5 fridge delivery (called 11:50 for 8-12 window). Delivery fee $79 refunded. Rebooked 10/7 AM. Customer demands food compensation today.",
            ),
        ],
        "brief": "Larkspur missed the window (called 10 minutes before it ended), so the $79 delivery fee is refundable; the customer threatens small-claims court over $260 of food. Policy: escalate to a supervisor (high) for the legal threat, stay courteous without arguing liability or promising the food money; refunding the delivery fee is fine.",
    },
    {
        "id": "third-miss-demands-600",
        "ticket": "T-24126",
        "title": "Third missed window; Gold customer demands a $600 credit",
        "category": "missed_delivery",
        "difficulty": "hard",
        "channel": "chat",
        "opened_at": "2026-10-06 10:50",
        "customer": customer(
            "C-15520",
            "Vanessa Cho",
            "Gold",
            8300,
            "2020-11-19",
            6,
            refunds_180d=[
                {
                    "date": "2026-09-30",
                    "amount": 129,
                    "reason": "delivery_fee (O-57850)",
                },
                {"date": "2026-09-30", "amount": 50, "reason": "goodwill (O-57850)"},
            ],
        ),
        "history": [
            {
                "date": "2026-09-24",
                "ticket": "T-23710",
                "channel": "chat",
                "summary": "First missed window on O-57850; rebooked 9/30. No remedy given.",
            },
            {
                "date": "2026-09-30",
                "ticket": "T-23802",
                "channel": "phone",
                "summary": "Second missed window on O-57850. Refunded $129 delivery fee + $50 goodwill. Rebooked 10/5.",
            },
        ],
        "orders": [
            {
                "order_id": "O-57850",
                "order_date": "2026-09-15",
                "status": "rescheduled",
                "delivery": {
                    "service": "white_glove",
                    "fee": 129,
                    "window": "Mon 2026-10-05 8am-12pm",
                    "delivered_on": None,
                    "events": [
                        "2026-09-24 window missed by Larkspur",
                        "2026-09-30 window missed by Larkspur",
                        "2026-10-05 12:00 window missed by Larkspur; no call logged",
                        "2026-10-05 15:00 rebooked Thu 2026-10-08 8am-12pm (priority route)",
                    ],
                },
                "items": [
                    item(
                        "TR-30",
                        'Tamsin 30" gas range',
                        "Tamsin",
                        2499,
                        install={
                            "type": "gas range",
                            "fee": 149,
                            "date": None,
                            "status": "pending",
                        },
                    )
                ],
            }
        ],
        "inbound": "THREE times. Three missed deliveries for one range. I have lost three days of work. I want a $600 credit, and I'm not interested in another $50.",
        "sim": {
            "persona": "Vanessa Cho, 37, freelance architect, sharp, sarcastic, exhausted.",
            "mood": "furious",
            "wants": "A $600 credit.",
            "accepts": "The policy goodwill plus a supervisor reviewing her request, and Thursday's priority delivery confirmed.",
            "hidden": [
                "If only offered the policy amount she asks for a manager.",
                "Thursday morning works.",
            ],
            "irate_if": [
                "the agent only quotes policy with no acknowledgment",
                "the agent refuses to let anyone else review",
            ],
            "calms_if": [
                "the agent acknowledges all three misses, gives what policy allows, and routes the rest to a supervisor"
            ],
        },
        "checks": [
            refund(50, 50, reasons=["goodwill", "delivery_fee"]),
            escalation_only_in(["supervisor", "approvals"]),
        ],
        "reference": [
            a_refund(
                "O-57850",
                50,
                "goodwill",
                "Third missed window on O-57850; delivery fee already refunded 9/30.",
            )
        ],
        "brief": "Third Larkspur miss on the same order. The delivery fee and a $50 goodwill credit were already refunded after the second miss. Policy: another $50 goodwill credit for this miss (the fee is refunded only once); not $600. A supervisor escalation is fine if she asks after the policy offer.",
    },
    {
        "id": "backorder-delay-goodwill",
        "ticket": "T-24127",
        "title": "Backordered range now 21 days past the promised date",
        "category": "backorder",
        "difficulty": "medium",
        "channel": "chat",
        "opened_at": "2026-10-06 14:55",
        "customer": customer(
            "C-37741", "Chris Delgado", "Standard", 1800, "2024-12-01", 2
        ),
        "history": [
            {
                "date": "2026-09-26",
                "ticket": "T-23735",
                "channel": "email",
                "summary": "Asked where his range was; told backordered, new ETA 10/16.",
            },
        ],
        "orders": [
            {
                "order_id": "O-57420",
                "order_date": "2026-09-01",
                "status": "backordered",
                "delivery": {
                    "service": "standard",
                    "fee": 79,
                    "window": "originally promised Fri 2026-09-25",
                    "delivered_on": None,
                    "events": [
                        "2026-09-22 backorder notice sent; new ETA Fri 2026-10-16"
                    ],
                },
                "items": [
                    item(
                        "TE-30",
                        'Tamsin 30" electric range',
                        "Tamsin",
                        1299,
                        install={
                            "type": "electric range",
                            "fee": 69,
                            "date": None,
                            "status": "pending",
                        },
                    )
                ],
            }
        ],
        "inbound": "I've been without a stove for a month. Your site still says the range is coming Oct 16. Is that real? This is getting old.",
        "sim": {
            "persona": "Chris Delgado, 30, line cook, tired, short sentences.",
            "mood": "frustrated",
            "wants": "Confidence the ETA is real and something for the wait.",
            "accepts": "Confirmation of the ETA, a small goodwill gesture, and someone checking in after.",
            "hidden": ["He does not want to cancel; the range is the one he wants."],
            "irate_if": [
                "the agent offers nothing",
                "the agent suggests canceling as the only option",
            ],
            "calms_if": [
                "the agent offers the goodwill credit and promises a check-in after the ETA"
            ],
        },
        "checks": [
            refund(25, 25, reasons=["goodwill"]),
            followup("2026-10-17", "2026-10-20"),
            escalation(False, w=1),
        ],
        "reference": [
            a_refund(
                "O-57420",
                25,
                "goodwill",
                "Backorder: ETA 10/16 is 21 days past promised 9/25 (>7 days).",
            ),
            a_followup("2026-10-19", "Confirm TE-30 delivered (ETA 10/16)."),
        ],
        "brief": "Backordered range now due 10/16, 21 days after the promised 9/25. Policy: one-time $25 goodwill credit (delay over 7 days) and a follow-up the business day after the ETA (Monday 10/19).",
    },
    # ---------------------------------------------------------------- info only / restraint
    {
        "id": "in-window-size-exchange",
        "ticket": "T-24128",
        "title": "In-window exchange for a larger model (self-serve)",
        "category": "exchange",
        "difficulty": "easy",
        "channel": "chat",
        "opened_at": "2026-10-06 11:11",
        "customer": customer(
            "C-26688", "Olivia Grant", "Silver", 2050, "2023-08-08", 2
        ),
        "history": [],
        "orders": [
            {
                "order_id": "O-58172",
                "order_date": "2026-09-24",
                "status": "delivered",
                "delivery": {
                    "service": "standard",
                    "fee": 79,
                    "window": "Thu 2026-10-01 8am-12pm",
                    "delivered_on": "2026-10-01",
                    "events": [
                        "2026-10-01 09:25 delivered (not installed; no water line)"
                    ],
                },
                "items": [
                    item(
                        "CT-30",
                        'Calloway 30" top-freezer refrigerator',
                        "Calloway",
                        1099,
                    )
                ],
            }
        ],
        "inbound": "The fridge works fine but it's just too small for our family. Can I swap it for the 33\" version (CT-33)?",
        "sim": {
            "persona": "Olivia Grant, 35, mom of three, friendly, multitasking.",
            "mood": "friendly",
            "wants": "To swap for the bigger model.",
            "accepts": "Clear self-serve steps and an honest note about the $79 pickup fee.",
            "hidden": [],
            "irate_if": ["the agent makes it complicated"],
            "calms_if": ["the agent gives the exact portal path"],
        },
        "checks": [
            exception(False),
            no_refund(w=2),
            escalation(False, w=1),
            dispatch(False, w=1),
            mentions(["larkspur.example/account", "returns center", "portal"], w=1),
        ],
        "reference": [
            a_message(
                "Yes! You're within 30 days, so you can swap it yourself in the Returns Center: larkspur.example/account -> Orders -> O-58172 -> Return or exchange, choose CT-33. Since it isn't defective there's a $79 pickup fee."
            )
        ],
        "brief": "Wants to exchange a working, uninstalled fridge for a bigger model 5 days after delivery: a standard in-window exchange she does herself in the portal Returns Center ($79 pickup fee for a non-defective item). Policy: no tools needed; no exception, refund, dispatch or escalation.",
    },
    {
        "id": "return-installed-not-defective",
        "ticket": "T-24129",
        "title": "Return of an installed dishwasher that is within spec",
        "category": "return",
        "difficulty": "medium",
        "channel": "chat",
        "opened_at": "2026-10-06 20:01",
        "customer": customer(
            "C-40550", "Henry Walsh", "Standard", 1240, "2025-06-06", 1
        ),
        "history": [],
        "orders": [
            {
                "order_id": "O-58060",
                "order_date": "2026-09-19",
                "status": "delivered",
                "delivery": {
                    "service": "standard",
                    "fee": 79,
                    "window": "Mon 2026-09-28 8am-12pm",
                    "delivered_on": "2026-09-28",
                    "events": ["2026-09-28 10:15 delivered and installed"],
                },
                "items": [
                    item(
                        "ND-24Q",
                        'Norvik 24" quiet dishwasher (44 dB)',
                        "Norvik",
                        1049,
                        install={
                            "type": "dishwasher",
                            "fee": 189,
                            "date": "2026-09-28",
                            "status": "completed",
                        },
                    )
                ],
            }
        ],
        "inbound": "The dishwasher is way louder than I expected. It works, I just hate the noise. I'd like to return it for a full refund including the installation.",
        "sim": {
            "persona": "Henry Walsh, 44, accountant, wants the exact numbers.",
            "mood": "neutral",
            "wants": "A full refund including installation.",
            "accepts": "An exact breakdown of what he'd get back and how to start it.",
            "hidden": [
                "He measured it at about 44 dB, which matches the spec.",
                "He will ask 'so how much exactly would I get back?' if not told.",
            ],
            "irate_if": [
                "the numbers are vague or wrong",
                "the agent pretends a full refund is possible",
            ],
            "calms_if": ["the agent gives a correct dollar breakdown"],
        },
        "checks": [
            no_refund(),
            mentions(["812.65"], w=2),
            exception(False, w=1),
            escalation(False, w=1),
        ],
        "reference": [
            {
                "tool": "send_message_to_customer",
                "args": {
                    "message": "Since it's installed and working as designed, you can return it within 15 days of delivery (until Oct 13) with a 15% restocking fee ($157.35) and the $79 pickup fee; the $189 installation fee isn't refundable. That comes to $812.65 back to your card after pickup. Start it at larkspur.example/account -> Orders -> O-58060 -> Return or exchange."
                },
            }
        ],
        "brief": "Installed, non-defective dishwasher returned 8 days after delivery: allowed within 15 days with a 15% restocking fee ($157.35) and $79 pickup; installation is not refundable. Policy: no agent-issued refund; he starts the return in the portal and receives $1,049 - $157.35 - $79 = $812.65 after pickup.",
    },
    {
        "id": "incomplete-gas-dryer-install",
        "ticket": "T-24130",
        "title": "Gas dryer delivered but not installed (crew lacked a connector)",
        "category": "installation",
        "difficulty": "medium",
        "channel": "chat",
        "opened_at": "2026-10-06 09:05",
        "customer": customer(
            "C-31100", "Fatima Haddad", "Silver", 3400, "2022-12-12", 3
        ),
        "history": [],
        "orders": [
            {
                "order_id": "O-58150",
                "order_date": "2026-09-24",
                "status": "partially installed",
                "delivery": {
                    "service": "white_glove",
                    "fee": 129,
                    "window": "Sat 2026-10-03 8am-12pm",
                    "delivered_on": "2026-10-03",
                    "events": [
                        "2026-10-03 09:45 delivered washer and gas dryer; washer installed",
                        "2026-10-03 10:40 crew note: 'gas dryer NOT installed - crew did not have gas flex connector; return visit needed'",
                    ],
                },
                "items": [
                    item(
                        "NV-W45",
                        "Norvik 4.5 cu ft front-load washer",
                        "Norvik",
                        899,
                        install={
                            "type": "washer",
                            "fee": 89,
                            "date": "2026-10-03",
                            "status": "completed",
                        },
                    ),
                    item(
                        "NG-27",
                        "Norvik 7.4 cu ft gas dryer",
                        "Norvik",
                        949,
                        install={
                            "type": "gas dryer",
                            "fee": 129,
                            "date": None,
                            "status": "not completed (Larkspur: missing connector)",
                        },
                    ),
                ],
            }
        ],
        "inbound": "The crew dropped off my gas dryer Saturday but couldn't hook it up because they didn't bring the right connector. Nobody has called me since. When is someone coming back? And I paid for installation that didn't happen.",
        "sim": {
            "persona": "Fatima Haddad, 46, pediatric nurse, organized, polite but firm.",
            "mood": "frustrated",
            "wants": "The dryer installed soon and the installation charge dealt with.",
            "accepts": "A confirmed return visit plus the installation fee refunded.",
            "hidden": [
                "She is available Saturday 10/10 morning or Tuesday 10/13 afternoon; she works 12-hour shifts the other days."
            ],
            "irate_if": [
                "the visit is booked without asking",
                "she is told to arrange a plumber herself",
            ],
            "calms_if": ["the visit and refund are both confirmed with specifics"],
        },
        "checks": [
            refund(129, 129, reasons=["installation_fee"]),
            refund(0, 129, w=1),
            dispatch(True, ["installation"], [("2026-10-10", "morning")]),
            followup_after_visit(),
            escalation(False, w=1),
        ],
        "reference": [
            a_refund(
                "O-58150",
                129,
                "installation_fee",
                "Gas dryer install not completed 10/3 (crew lacked gas flex connector).",
            ),
            a_dispatch(
                "O-58150",
                "installation",
                "2026-10-10",
                "morning",
                "Complete gas dryer (NG-27) installation; bring gas flex connector.",
            ),
            a_followup("2026-10-12", "Confirm gas dryer installed after 10/10 visit."),
        ],
        "brief": "The gas dryer installation was not completed because the crew lacked a connector (Larkspur's fault). Policy: refund the $129 gas-dryer installation fee, dispatch an installation return visit at her earliest availability (Saturday 10/10 morning), and schedule a follow-up after it.",
    },
]


def by_id() -> dict[str, dict[str, Any]]:
    return {s["id"]: s for s in SCENARIOS}


assert len(SCENARIOS) == 30, len(SCENARIOS)
assert len({s["ticket"] for s in SCENARIOS}) == 30
