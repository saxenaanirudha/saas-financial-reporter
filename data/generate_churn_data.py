"""
generate_churn_data.py
======================
Deterministic synthetic customer-level churn dataset generator.

Run once to produce data/customer_churn_data.csv.
The dataset is intentionally synthetic and is used solely for portfolio
demonstration of an end-to-end ML workflow.

Usage:
    python data/generate_churn_data.py
"""

import random
import csv
from pathlib import Path
from datetime import date, timedelta

RANDOM_SEED = 42
NUM_CUSTOMERS = 500
OUTPUT_FILE = Path(__file__).parent / "customer_churn_data.csv"

PLANS = ["Basic", "Pro", "Enterprise"]
PLAN_FEES = {"Basic": 29.99, "Pro": 79.99, "Enterprise": 199.99}
CONTRACT_TYPES = ["Monthly", "Annual"]

REFERENCE_DATE = date(2025, 6, 30)
SIGNUP_START = date(2022, 1, 1)
SIGNUP_END = date(2024, 6, 30)


def random_date(rng, start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=rng.randint(0, delta))


def generate_churn_label(
    last_login_days: int,
    payment_failures: int,
    usage_hours: float,
    support_tickets: int,
    auto_renew: int,
    contract_type: str,
    tenure_months: int,
    rng: random.Random,
) -> int:
    """
    Synthetic rule-based churn signal.
    Higher last_login_days, more payment_failures, low usage → higher churn risk.
    Not a real business model — purely for portfolio ML demonstration.
    """
    score = 0.0

    # Recency signal: long time since last login
    if last_login_days > 75:
        score += 0.30
    elif last_login_days > 45:
        score += 0.15

    # Payment failures
    if payment_failures >= 3:
        score += 0.25
    elif payment_failures >= 1:
        score += 0.10

    # Low usage
    if usage_hours < 15:
        score += 0.25
    elif usage_hours < 40:
        score += 0.10

    # No auto-renew
    if auto_renew == 0:
        score += 0.10

    # Monthly contract (easier to leave)
    if contract_type == "Monthly":
        score += 0.08

    # Short tenure
    if tenure_months < 6:
        score += 0.08

    # Support tickets (frustration signal)
    if support_tickets >= 6:
        score += 0.12
    elif support_tickets >= 3:
        score += 0.05

    # Add noise
    score += rng.uniform(-0.08, 0.08)
    score = max(0.0, min(1.0, score))

    # Convert to binary with threshold
    return 1 if score >= 0.65 else 0


def generate_dataset(seed: int = RANDOM_SEED, n: int = NUM_CUSTOMERS) -> list:
    rng = random.Random(seed)
    rows = []

    for i in range(1, n + 1):
        customer_id = f"CUST-{i:04d}"
        signup_date = random_date(rng, SIGNUP_START, SIGNUP_END)
        tenure_months = max(
            1, (REFERENCE_DATE.year - signup_date.year) * 12
            + (REFERENCE_DATE.month - signup_date.month)
        )

        plan = rng.choice(PLANS)
        monthly_fee = round(PLAN_FEES[plan] + rng.uniform(-5.0, 5.0), 2)
        contract_type = rng.choice(CONTRACT_TYPES)
        auto_renew = rng.randint(0, 1)

        last_login_days_ago = rng.randint(1, 120)
        support_tickets = rng.randint(0, 10)
        monthly_usage_hours = round(rng.uniform(0.5, 200.0), 2)
        payment_failures = rng.randint(0, 5)
        discount_percent = round(rng.uniform(0.0, 30.0), 2)

        churned = generate_churn_label(
            last_login_days_ago,
            payment_failures,
            monthly_usage_hours,
            support_tickets,
            auto_renew,
            contract_type,
            tenure_months,
            rng,
        )

        rows.append({
            "CustomerID": customer_id,
            "SignupDate": signup_date.isoformat(),
            "Plan": plan,
            "MonthlyFee": monthly_fee,
            "TenureMonths": tenure_months,
            "LastLoginDaysAgo": last_login_days_ago,
            "SupportTickets": support_tickets,
            "MonthlyUsageHours": monthly_usage_hours,
            "PaymentFailures": payment_failures,
            "DiscountPercent": discount_percent,
            "ContractType": contract_type,
            "AutoRenew": auto_renew,
            "Churned": churned,
        })

    return rows


def main():
    rows = generate_dataset()
    churned_count = sum(r["Churned"] for r in rows)
    active_count = len(rows) - churned_count
    churn_rate = churned_count / len(rows) * 100

    fieldnames = [
        "CustomerID", "SignupDate", "Plan", "MonthlyFee", "TenureMonths",
        "LastLoginDaysAgo", "SupportTickets", "MonthlyUsageHours",
        "PaymentFailures", "DiscountPercent", "ContractType", "AutoRenew", "Churned",
    ]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated: {OUTPUT_FILE}")
    print(f"Total customers : {len(rows)}")
    print(f"Active (0)      : {active_count}")
    print(f"Churned (1)     : {churned_count}")
    print(f"Churn rate      : {churn_rate:.1f}%")


if __name__ == "__main__":
    main()
