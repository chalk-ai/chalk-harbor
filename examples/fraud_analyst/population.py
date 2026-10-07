"""A small, seeded population of Kestrel Pay accounts: history with known outcomes, and a review
queue of flagged payouts whose outcome is hidden.

``generate()`` returns everything the example needs, deterministically:

* ``tables`` -- the warehouse the analyst can query with SQL (accounts, logins, transactions,
  review_queue). Pending accounts' statuses read ``pending_review``; nothing in it is a label.
* ``features`` -- per-account feature values served by the Chalk query tool.
* ``sealed`` -- per-account results of the two paid tools (identity deep verification, $5; the
  consortium social-network search, $2). They live root-only in the sandbox.
* ``cases`` -- the review queue with each case's archetype and true label, which reach the
  sandbox only with the rubric at grading time.

Archetypes are built so cheap evidence settles some cases and only paid evidence settles others:
ring members share devices with closed fraud accounts (visible in SQL); account takeovers show a
new device and country, a password reset and a new payout destination; synthetic identities look
clean until deep verification; risky-looking legitimate customers (travellers, thin files, shared
households) clear on deep verification and network search.

A second, hard tier of 12 cases (K-5041 on) is built so the most visible signal points the wrong
way and the decision rests on joining or weighing evidence: a new account paying out to a bank
account that closed fraud accounts cashed out to; a domestic takeover the risk model scores low; a
dormant, verified identity cashing out a burst of deposits from new cards; and, on the legitimate
side, a phone-loss recovery that looks like a takeover, a second-hand device last used by a
fraudster years ago, and a long-standing customer who always logs in through a VPN abroad. The
hard tier draws from its own random stream, so it leaves the first 40 cases unchanged.
"""

from __future__ import annotations

import datetime as dt
import random
from typing import Any

TODAY = dt.date(2026, 10, 7)
SEED = 20261007
HISTORICAL_ACCOUNTS = 320
RINGS = 8

HARD_FRAUD_ARCHETYPES = ("payout_mule_link", "ato_quiet", "bust_out")
HARD_LEGIT_ARCHETYPES = (
    "legit_account_recovery",
    "legit_resold_device",
    "legit_vpn_privacy",
)
HARD_ARCHETYPES = (*HARD_FRAUD_ARCHETYPES, *HARD_LEGIT_ARCHETYPES)
FRAUD_ARCHETYPES = (
    "ring_member",
    "synthetic_identity",
    "account_takeover",
    *HARD_FRAUD_ARCHETYPES,
)
LEGIT_ARCHETYPES = (
    "legit_clean",
    "legit_traveler",
    "legit_thin_file",
    "legit_household",
    *HARD_LEGIT_ARCHETYPES,
)
# The review queue, in order: 40 cases, 20 fraud and 20 legitimate.
QUEUE = [
    *(["ring_member"] * 7), *(["synthetic_identity"] * 7), *(["account_takeover"] * 6),
    *(["legit_clean"] * 6), *(["legit_traveler"] * 5), *(["legit_thin_file"] * 5), *(["legit_household"] * 4),
]  # fmt: skip
# The hard tier, after it: 12 cases, two of each hard archetype.
HARD_QUEUE = [archetype for archetype in HARD_ARCHETYPES for _ in range(2)]
# The production model's score for the hard tier: low on the frauds, high on the legitimate cases.
HARD_RISK = {
    "payout_mule_link": (0.25, 0.45), "ato_quiet": (0.12, 0.3), "bust_out": (0.15, 0.35),
    "legit_account_recovery": (0.6, 0.82), "legit_resold_device": (0.55, 0.78), "legit_vpn_privacy": (0.65, 0.88),
}  # fmt: skip

FIRST = [
    "Ava",
    "Liam",
    "Noah",
    "Emma",
    "Mia",
    "Lucas",
    "Zoe",
    "Omar",
    "Priya",
    "Chen",
    "Diego",
    "Sara",
    "Ken",
    "Nia",
    "Ivan",
    "Lena",
    "Theo",
    "Rosa",
    "Amir",
    "Kai",
    "Hana",
    "Joel",
    "Ines",
    "Rui",
]
LAST = [
    "Park",
    "Silva",
    "Okafor",
    "Novak",
    "Haddad",
    "Brooks",
    "Tanaka",
    "Rossi",
    "Mendes",
    "Clarke",
    "Ibrahim",
    "Kowalski",
    "Nguyen",
    "Fischer",
    "Bauer",
    "Costa",
    "Ali",
    "Moreau",
]
CITIES = [
    "Austin TX",
    "Denver CO",
    "Columbus OH",
    "Tampa FL",
    "Portland OR",
    "Raleigh NC",
    "Phoenix AZ",
    "Madison WI",
]
COUNTRIES_ABROAD = ["PT", "MX", "JP", "FR", "TH", "GB"]
RISKY_COUNTRIES = ["RO", "NG", "VN", "UA", "BR"]


def _ts(rng: random.Random, days_ago_lo: int, days_ago_hi: int) -> str:
    day = TODAY - dt.timedelta(days=rng.randint(days_ago_lo, days_ago_hi))
    return f"{day.isoformat()} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}"


class _Builder:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.standard_rng = self.rng
        self.first_hard_account: int | None = None
        self.accounts: list[dict[str, Any]] = []
        self.logins: list[dict[str, Any]] = []
        self.transactions: list[dict[str, Any]] = []
        self.features: dict[str, dict[str, Any]] = {}
        self.sealed: dict[str, dict[str, Any]] = {}
        self.latent: dict[str, dict[str, Any]] = {}
        self.ring_devices: list[list[str]] = []
        self.ring_accounts: list[list[str]] = []
        self._next_account = 10001
        self._next_device = 1
        self._next_txn = 1

    # -- primitives ---------------------------------------------------------------------------

    def device(self) -> str:
        self._next_device += 1
        return f"D-{self._next_device:05d}"

    def new_account(self, *, age_days: int, status: str, email_domain: str | None = None,
                    city: str | None = None, last: str | None = None) -> str:  # fmt: skip
        rng = self.rng
        account_id = f"A-{self._next_account}"
        self._next_account += 1
        first, last = rng.choice(FIRST), last or rng.choice(LAST)
        domain = email_domain or rng.choice(
            ["gmail.com", "outlook.com", "icloud.com", "yahoo.com"]
        )
        self.accounts.append({
            "account_id": account_id,
            "created_at": (TODAY - dt.timedelta(days=age_days)).isoformat(),
            "full_name": f"{first} {last}",
            "email": f"{first.lower()}.{last.lower()}{rng.randint(1, 999)}@{domain}",
            "phone": f"+1-{rng.randint(200, 989)}-555-{rng.randint(1000, 9999)}",
            "city": city or rng.choice(CITIES),
            "status": status,
        })  # fmt: skip
        return account_id

    def login(
        self,
        account_id: str,
        device: str,
        country: str,
        vpn: bool,
        days: tuple[int, int],
        n: int,
    ) -> None:
        for _ in range(n):
            self.logins.append({"account_id": account_id, "ts": _ts(self.rng, *days), "device_id": device,
                                "ip_country": country, "vpn": int(vpn)})  # fmt: skip

    def txn(self, account_id: str, kind: str, amount: float, days: tuple[int, int], status: str = "settled",
            destination: str | None = None) -> None:  # fmt: skip
        self.transactions.append({"txn_id": f"T-{self._next_txn:06d}", "account_id": account_id,
                                  "ts": _ts(self.rng, *days), "type": kind, "amount": round(amount, 2),
                                  "destination": destination, "status": status})  # fmt: skip
        self._next_txn += 1

    def normal_activity(
        self, account_id: str, device: str, age_days: int, country: str = "US"
    ) -> None:
        rng = self.rng
        span = (1, max(2, min(age_days, 60)))
        self.login(account_id, device, country, False, span, rng.randint(4, 14))
        for _ in range(rng.randint(3, 12)):
            self.txn(
                account_id,
                rng.choice(["purchase", "purchase", "deposit"]),
                rng.uniform(12, 380),
                span,
            )
        if age_days > 30:
            self.txn(
                account_id,
                "payout",
                rng.uniform(80, 900),
                span,
                destination=f"bank-{rng.randint(1000, 9999)}",
            )

    # -- historical population --------------------------------------------------------------------

    def history(self) -> None:
        rng = self.rng
        for _ in range(RINGS):
            devices = [self.device() for _ in range(rng.randint(1, 2))]
            members = []
            for _ in range(rng.randint(3, 5)):
                age = rng.randint(20, 200)
                status = "closed_fraud" if rng.random() < 0.8 else "active"
                account = self.new_account(
                    age_days=age,
                    status=status,
                    email_domain=rng.choice(["mail.tm", "proton.me"]),
                )
                # Every member uses the ring's first device, so a new member shares it with all of them.
                for device in devices[: rng.randint(1, len(devices))]:
                    self.login(
                        account,
                        device,
                        "US",
                        rng.random() < 0.6,
                        (1, min(age, 60)),
                        rng.randint(2, 6),
                    )
                for _ in range(rng.randint(2, 6)):
                    self.txn(account, "purchase", rng.uniform(200, 1500), (1, min(age, 60)),
                             status="chargeback" if rng.random() < 0.5 else "settled")  # fmt: skip
                members.append(account)
                self.latent[account] = {"fraud": True, "archetype": "ring_member"}
            self.ring_devices.append(devices)
            self.ring_accounts.append(members)
        while len(self.accounts) < HISTORICAL_ACCOUNTS:
            fraud = rng.random() < 0.08
            age = rng.randint(30, 2400)
            status = (
                "closed_fraud"
                if fraud
                else rng.choice(["active"] * 9 + ["closed_voluntary"])
            )
            account = self.new_account(age_days=age, status=status)
            device = self.device()
            self.normal_activity(account, device, age)
            if fraud:
                self.txn(
                    account,
                    "purchase",
                    rng.uniform(600, 2000),
                    (1, 60),
                    status="chargeback",
                )
            self.latent[account] = {"fraud": fraud, "archetype": "historical"}

    # -- the review queue -------------------------------------------------------------------------

    def pending(self, archetype: str) -> tuple[str, str, float]:
        """Create one flagged account; return (account_id, trigger, amount)."""
        rng = self.rng
        fraud = archetype in FRAUD_ARCHETYPES
        if archetype == "ring_member":
            ring = rng.randrange(RINGS)
            age = rng.randint(3, 25)
            account = self.new_account(
                age_days=age,
                status="pending_review",
                email_domain=rng.choice(["mail.tm", "proton.me", "gmail.com"]),
            )
            self.login(
                account,
                self.ring_devices[ring][0],
                "US",
                rng.random() < 0.7,
                (0, age),
                rng.randint(3, 8),
            )
            for _ in range(rng.randint(2, 5)):
                self.txn(account, "deposit", rng.uniform(300, 900), (1, age))
            amount = round(rng.uniform(1800, 4800), 2)
            trigger = f"Payout of ${amount:,.2f} to a bank account added {rng.randint(1, 3)} days ago, {age} days after sign-up"
        elif archetype == "synthetic_identity":
            age = rng.randint(40, 140)
            account = self.new_account(age_days=age, status="pending_review")
            device = self.device()
            self.normal_activity(account, device, age)
            amount = round(rng.uniform(2500, 6000), 2)
            trigger = f"Payout of ${amount:,.2f}, {rng.randint(3, 6)}x this account's largest previous payout"
        elif archetype == "account_takeover":
            age = rng.randint(600, 2400)
            account = self.new_account(age_days=age, status="pending_review")
            self.normal_activity(account, self.device(), age)
            country = rng.choice(RISKY_COUNTRIES)
            self.login(account, self.device(), country, True, (0, 3), rng.randint(2, 5))
            amount = round(rng.uniform(1500, 5000), 2)
            trigger = f"Payout of ${amount:,.2f} to a new bank account, requested after a password reset"
        elif archetype == "legit_clean":
            age = rng.randint(400, 2400)
            account = self.new_account(age_days=age, status="pending_review")
            self.normal_activity(account, self.device(), age)
            amount = round(rng.uniform(1200, 3200), 2)
            trigger = (
                f"Payout of ${amount:,.2f} exceeds the account's 90-day average by 4x"
            )
        elif archetype == "legit_traveler":
            age = rng.randint(500, 2400)
            account = self.new_account(age_days=age, status="pending_review")
            device = self.device()
            self.normal_activity(account, device, age)
            country = rng.choice(COUNTRIES_ABROAD)
            self.login(
                account, device, country, rng.random() < 0.6, (0, 6), rng.randint(3, 6)
            )
            amount = round(rng.uniform(900, 2600), 2)
            trigger = f"Payout of ${amount:,.2f} requested from {country}, outside the account's usual country"
        elif archetype == "legit_thin_file":
            age = rng.randint(4, 20)
            account = self.new_account(age_days=age, status="pending_review")
            device = self.device()
            self.login(account, device, "US", False, (0, age), rng.randint(2, 5))
            self.txn(account, "deposit", rng.uniform(1500, 4000), (1, age))
            amount = round(rng.uniform(1400, 3500), 2)
            trigger = f"Payout of ${amount:,.2f} from an account opened {age} days ago"
        elif archetype == "legit_household":
            partner = next(a for a in self.accounts if a["status"] == "active" and self.latent[a["account_id"]]["archetype"] == "historical"
                           and not self.latent[a["account_id"]]["fraud"] and a["account_id"] not in self._used_partners)  # fmt: skip
            self._used_partners.add(partner["account_id"])
            partner_device = next(
                r["device_id"]
                for r in self.logins
                if r["account_id"] == partner["account_id"]
            )
            age = rng.randint(60, 600)
            account = self.new_account(age_days=age, status="pending_review", city=partner["city"],
                                       last=partner["full_name"].split()[1])  # fmt: skip
            self.normal_activity(account, partner_device, age)
            amount = round(rng.uniform(1100, 2800), 2)
            trigger = f"Payout of ${amount:,.2f}; the login device is shared with another customer"
        else:
            raise ValueError(archetype)
        self.txn(
            account,
            "payout",
            amount,
            (0, 0),
            status="pending",
            destination=f"bank-{rng.randint(1000, 9999)}",
        )
        self.latent[account] = {"fraud": fraud, "archetype": archetype}
        return account, trigger, amount

    def start_hard_phase(self) -> None:
        """Switch to the hard tier's own random stream, so it leaves the earlier cases unchanged."""
        self.standard_rng = self.rng
        self.rng = random.Random(self.seed + 1)
        self.first_hard_account = len(self.accounts)

    def pending_hard(self, archetype: str) -> tuple[str, str, float]:
        """Create one flagged account in the hard tier; return (account_id, trigger, amount)."""
        rng = self.rng
        home = self.device()
        latent: dict[str, Any] = {
            "fraud": archetype in FRAUD_ARCHETYPES,
            "archetype": archetype,
        }
        destination = f"bank-{rng.randint(1000, 9999)}"
        if archetype == "payout_mule_link":
            # Closed fraud accounts cashed out to this bank account months ago; the new account
            # has its own device and a quiet history, and pays out to the same account.
            for _ in range(rng.randint(2, 3)):
                prior_age = rng.randint(150, 400)
                prior = self.new_account(age_days=prior_age, status="closed_fraud")
                self.latent[prior] = {"fraud": True, "archetype": "historical"}
                self.login(
                    prior,
                    self.device(),
                    "US",
                    rng.random() < 0.5,
                    (90, prior_age),
                    rng.randint(3, 6),
                )
                for _ in range(rng.randint(2, 4)):
                    self.txn(
                        prior,
                        "deposit",
                        rng.uniform(400, 1500),
                        (90, prior_age),
                        status="chargeback",
                    )
                self.txn(
                    prior,
                    "payout",
                    rng.uniform(1500, 4000),
                    (90, min(prior_age, 140)),
                    destination=destination,
                )
            age = rng.randint(25, 80)
            account = self.new_account(age_days=age, status="pending_review")
            self.login(account, home, "US", False, (0, age), rng.randint(4, 9))
            deposits = [rng.uniform(300, 1100) for _ in range(rng.randint(3, 6))]
            for deposit in deposits:
                self.txn(account, "deposit", deposit, (1, age))
            amount = round(sum(deposits) * rng.uniform(0.85, 0.95), 2)
            trigger = f"First payout from this account (${amount:,.2f}), {age} days after sign-up"
        elif archetype == "ato_quiet":
            # A domestic takeover: no VPN, no new country. The owner's own device is still active
            # while a second device appears, resets the password and adds a bank account.
            age = rng.randint(900, 2400)
            account = self.new_account(age_days=age, status="pending_review")
            self.normal_activity(account, home, age)
            self.login(account, home, "US", False, (0, 2), rng.randint(1, 3))
            self.login(account, self.device(), "US", False, (0, 2), rng.randint(2, 4))
            amount = round(rng.uniform(1600, 4200), 2)
            trigger = f"Payout of ${amount:,.2f} to a bank account added {rng.choice(['yesterday', '2 days ago'])}"
        elif archetype == "bust_out":
            # A real, verified identity: a long-quiet account takes a burst of deposits from
            # several new cards (reported stolen in the consortium) and cashes them out.
            age = rng.randint(400, 900)
            account = self.new_account(age_days=age, status="pending_review")
            self.login(account, home, "US", False, (20, 300), rng.randint(5, 10))
            for _ in range(rng.randint(3, 6)):
                self.txn(account, "purchase", rng.uniform(15, 120), (20, 300))
            self.login(account, home, "US", False, (0, 9), rng.randint(4, 8))
            deposits = [rng.uniform(400, 1200) for _ in range(rng.randint(5, 8))]
            for deposit in deposits:
                self.txn(
                    account,
                    "deposit",
                    deposit,
                    (1, 9),
                    destination=f"card-{rng.randint(1000, 9999)}",
                )
            amount = round(sum(deposits) * rng.uniform(0.85, 0.95), 2)
            trigger = f"Payout of ${amount:,.2f}; the account has never paid out before"
        elif archetype == "legit_account_recovery":
            # A lost phone: the old device goes silent, a new one logs in from home, the
            # customer resets the password and the carrier issues a new SIM.
            age = rng.randint(700, 2400)
            account = self.new_account(age_days=age, status="pending_review")
            self.login(account, home, "US", False, (12, 60), rng.randint(5, 12))
            for _ in range(rng.randint(3, 10)):
                self.txn(
                    account,
                    rng.choice(["purchase", "purchase", "deposit"]),
                    rng.uniform(12, 380),
                    (12, 60),
                )
            self.txn(
                account,
                "payout",
                rng.uniform(300, 1200),
                (12, 60),
                destination=f"bank-{rng.randint(1000, 9999)}",
            )
            self.login(account, self.device(), "US", False, (0, 6), rng.randint(3, 6))
            amount = round(rng.uniform(900, 2600), 2)
            trigger = f"Payout of ${amount:,.2f} to a new bank account, requested from a new device after a password reset"
        elif archetype == "legit_resold_device":
            # The device's previous owner was closed for fraud long before this account existed.
            prior_age = rng.randint(800, 1200)
            prior = self.new_account(age_days=prior_age, status="closed_fraud")
            self.latent[prior] = {"fraud": True, "archetype": "historical"}
            self.login(
                prior, home, "US", rng.random() < 0.5, (500, 750), rng.randint(4, 8)
            )
            for _ in range(rng.randint(2, 4)):
                self.txn(
                    prior,
                    "purchase",
                    rng.uniform(300, 1200),
                    (500, 750),
                    status="chargeback",
                )
            latent["prior_owner"] = prior
            age = rng.randint(90, 300)
            account = self.new_account(age_days=age, status="pending_review")
            self.normal_activity(account, home, age)
            amount = round(rng.uniform(800, 2400), 2)
            trigger = f"Payout of ${amount:,.2f}; the login device was previously used by an account closed for fraud"
        elif archetype == "legit_vpn_privacy":
            # Years of logins through one VPN exit abroad on one device, paying out to the same
            # bank account as always.
            age = rng.randint(900, 2400)
            account = self.new_account(age_days=age, status="pending_review")
            country = rng.choice(RISKY_COUNTRIES)
            self.login(account, home, country, True, (1, 400), rng.randint(12, 20))
            self.login(account, home, country, True, (0, 1), rng.randint(1, 2))
            for _ in range(rng.randint(5, 12)):
                self.txn(
                    account,
                    rng.choice(["purchase", "purchase", "deposit"]),
                    rng.uniform(12, 380),
                    (1, 400),
                )
            for _ in range(rng.randint(3, 5)):
                self.txn(
                    account,
                    "payout",
                    rng.uniform(600, 2000),
                    (20, 400),
                    destination=destination,
                )
            amount = round(rng.uniform(1200, 2600), 2)
            trigger = (
                f"Payout of ${amount:,.2f} requested through a VPN exiting in {country}"
            )
        else:
            raise ValueError(archetype)
        self.txn(
            account, "payout", amount, (0, 0), status="pending", destination=destination
        )
        self.latent[account] = latent
        return account, trigger, amount

    # -- derived data -----------------------------------------------------------------------------

    def derive(self) -> None:
        by_device: dict[str, set[str]] = {}
        for row in self.logins:
            by_device.setdefault(row["device_id"], set()).add(row["account_id"])
        status = {a["account_id"]: a["status"] for a in self.accounts}
        logins_by_account: dict[str, list[dict[str, Any]]] = {}
        for row in self.logins:
            logins_by_account.setdefault(row["account_id"], []).append(row)
        for index, account in enumerate(self.accounts):
            hard = (
                self.first_hard_account is not None and index >= self.first_hard_account
            )
            rng = self.rng if hard else self.standard_rng
            account_id, latent = (
                account["account_id"],
                self.latent[account["account_id"]],
            )
            archetype, fraud = latent["archetype"], latent["fraud"]
            age = (TODAY - dt.date.fromisoformat(account["created_at"])).days
            logins = [r for r in self.logins if r["account_id"] == account_id]
            recent = [
                r
                for r in logins
                if (TODAY - dt.date.fromisoformat(r["ts"][:10])).days <= 30
            ]
            txns = [t for t in self.transactions if t["account_id"] == account_id]
            devices = {r["device_id"] for r in recent}
            shared = set().union(
                *(by_device.get(d, set()) for d in {r["device_id"] for r in logins})
            ) - {account_id}
            risk = rng.uniform(*HARD_RISK[archetype]) if archetype in HARD_RISK else {
                "ring_member": rng.uniform(0.62, 0.9), "synthetic_identity": rng.uniform(0.18, 0.42),
                "account_takeover": rng.uniform(0.55, 0.85), "legit_clean": rng.uniform(0.05, 0.25),
                "legit_traveler": rng.uniform(0.5, 0.78), "legit_thin_file": rng.uniform(0.45, 0.7),
                "legit_household": rng.uniform(0.35, 0.6), "historical": rng.uniform(0.5, 0.9) if fraud else rng.uniform(0.02, 0.3),
            }[archetype]  # fmt: skip
            self.features[account_id] = {
                "account.age_days": age,
                "account.email_age_days": rng.randint(1, 40) if archetype in ("ring_member", "synthetic_identity", "payout_mule_link") else rng.randint(age, age + 2000),
                "account.risk_score": round(risk, 3),
                "account.txn_count_30d": sum(1 for t in txns if (TODAY - dt.date.fromisoformat(t["ts"][:10])).days <= 30),
                "account.chargeback_count_180d": sum(1 for t in txns if t["status"] == "chargeback"),
                "account.distinct_devices_30d": len(devices),
                "account.distinct_countries_30d": len({r["ip_country"] for r in recent}),
                "account.vpn_login_ratio_30d": round(sum(r["vpn"] for r in recent) / len(recent), 2) if recent else 0.0,
                "account.password_reset_7d": archetype in ("account_takeover", "ato_quiet", "legit_account_recovery"),
                "account.new_payout_destination_7d": archetype in ("ring_member", "account_takeover", "payout_mule_link", "ato_quiet",
                                                                  "bust_out", "legit_account_recovery")
                                                     or (archetype not in HARD_ARCHETYPES and rng.random() < 0.15),
                "account.device_shared_account_count": len(shared),
                "account.max_prior_payout_usd": round(max([t["amount"] for t in txns if t["type"] == "payout" and t["status"] == "settled"] or [0.0]), 2),
            }  # fmt: skip
            deep = {"document_match": True, "ssn_name_dob_match": True, "liveness_passed": True,
                    "synthetic_identity_score": round(rng.uniform(0.02, 0.15), 2), "phone_tenure_months": rng.randint(24, 140),
                    "sim_swap_last_7d": False, "address_history_years": rng.randint(3, 15)}  # fmt: skip
            if archetype == "synthetic_identity":
                deep.update(ssn_name_dob_match=False, synthetic_identity_score=round(rng.uniform(0.84, 0.97), 2),
                            phone_tenure_months=rng.randint(1, 4), address_history_years=0)  # fmt: skip
            elif archetype == "ring_member":
                deep.update(synthetic_identity_score=round(rng.uniform(0.35, 0.65), 2), phone_tenure_months=rng.randint(0, 3),
                            liveness_passed=rng.random() < 0.5)  # fmt: skip
            elif archetype == "account_takeover":
                deep.update(sim_swap_last_7d=True)
            elif archetype == "payout_mule_link":
                deep.update(
                    synthetic_identity_score=round(rng.uniform(0.3, 0.5), 2),
                    phone_tenure_months=rng.randint(2, 8),
                )
            elif archetype == "ato_quiet":
                deep.update(sim_swap_last_7d=True, liveness_passed=False)
            elif archetype == "legit_account_recovery":
                deep.update(sim_swap_last_7d=True)
            elif archetype == "legit_thin_file":
                deep.update(
                    phone_tenure_months=rng.randint(30, 90),
                    address_history_years=rng.randint(1, 4),
                )
            links: list[dict[str, Any]] = []
            if archetype == "ring_member" or (
                archetype == "historical"
                and fraud
                and account_id in {a for r in self.ring_accounts for a in r}
            ):
                links = [{"link_type": rng.choice(["phone", "device", "email_pattern"]), "institution": f"member-{rng.randint(10, 99)}",
                          "status": "confirmed_fraud"} for _ in range(rng.randint(2, 4))]  # fmt: skip
            elif archetype == "synthetic_identity":
                links = [{"link_type": "ssn_with_different_name", "institution": f"member-{rng.randint(10, 99)}",
                          "status": rng.choice(["confirmed_fraud", "under_investigation"])} for _ in range(rng.randint(1, 3))]  # fmt: skip
            elif archetype == "account_takeover":
                links = [
                    {
                        "link_type": "device",
                        "institution": f"member-{rng.randint(10, 99)}",
                        "status": "confirmed_fraud",
                    }
                ]
            elif archetype == "payout_mule_link":
                links = [{"link_type": "payout_destination", "institution": f"member-{rng.randint(10, 99)}",
                          "status": "confirmed_fraud"} for _ in range(rng.randint(1, 2))]  # fmt: skip
            elif archetype == "ato_quiet":
                links = [
                    {
                        "link_type": "device",
                        "institution": f"member-{rng.randint(10, 99)}",
                        "status": "confirmed_fraud",
                    }
                ]
            elif archetype == "bust_out":
                links = [{"link_type": "funding_card", "institution": f"member-{rng.randint(10, 99)}",
                          "status": "confirmed_fraud"} for _ in range(rng.randint(2, 3))]  # fmt: skip
            elif archetype == "legit_resold_device":
                prior = latent["prior_owner"]
                last_seen = max(r["ts"] for r in logins_by_account[prior])[:10]
                links = [{"link_type": "device", "institution": "kestrel", "account_id": prior, "status": "closed_fraud",
                          "last_seen_on_device": last_seen}]  # fmt: skip
            elif archetype == "legit_household":
                partners = sorted(shared)
                links = [
                    {
                        "link_type": "device",
                        "institution": "kestrel",
                        "account_id": p,
                        "status": "good_standing",
                    }
                    for p in partners
                    if status.get(p) == "active"
                ]
            self.sealed[account_id] = {
                "deep_verification": deep,
                "social_network_search": {"consortium_fraud_reports": sum(1 for x in links if x["status"] == "confirmed_fraud"),
                                          "linked_identities": links},
            }  # fmt: skip


def generate(seed: int = SEED) -> dict[str, Any]:
    builder = _Builder(seed)
    builder._used_partners = set()  # type: ignore[attr-defined]
    builder.history()
    cases = []
    for index, archetype in enumerate(QUEUE):
        account, trigger, amount = builder.pending(archetype)
        cases.append({"case_id": f"K-{5001 + index}", "account_id": account, "trigger": trigger, "amount_usd": amount,
                      "opened_at": f"{TODAY.isoformat()} 09:{index:02d}", "archetype": archetype,
                      "label": "fraud" if archetype in FRAUD_ARCHETYPES else "legit"})  # fmt: skip
    builder.start_hard_phase()
    for archetype in HARD_QUEUE:
        index = len(cases)
        account, trigger, amount = builder.pending_hard(archetype)
        cases.append({"case_id": f"K-{5001 + index}", "account_id": account, "trigger": trigger, "amount_usd": amount,
                      "opened_at": f"{TODAY.isoformat()} 09:{index:02d}", "archetype": archetype,
                      "label": "fraud" if archetype in FRAUD_ARCHETYPES else "legit"})  # fmt: skip
    builder.derive()
    review_queue = [
        {
            k: c[k]
            for k in ("case_id", "account_id", "opened_at", "trigger", "amount_usd")
        }
        for c in cases
    ]
    return {
        "today": TODAY.isoformat(),
        "tables": {
            "accounts": builder.accounts,
            "logins": builder.logins,
            "transactions": builder.transactions,
            "review_queue": review_queue,
        },
        "features": builder.features,
        "sealed": builder.sealed,
        "cases": cases,
    }
