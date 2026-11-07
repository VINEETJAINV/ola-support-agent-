"""
Dataset generator for Ola support tickets.
Uses a fixed seed for deterministic generation.
"""
import os
import random
import json
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"


# SEED: 42 - chosen for deterministic generation
SEED = 42

# Category weights - sum to 1.0
CATEGORIES = ["Billing", "Technical Issue", "Account Access", "Product Defect", "General Inquiry"]
CATEGORY_WEIGHTS = [0.25, 0.20, 0.20, 0.15, 0.20]

# Status weights - sum to 1.0
STATUSES = ["Open", "In Progress", "Escalated", "Resolved", "Closed"]
STATUS_WEIGHTS = [0.15, 0.25, 0.10, 0.30, 0.20]

# Escalated probability (target 10-30%)
ESCALATED_PROB = 0.20

# Resolution time range: 0.5 to 72 hours
# Reasoning: Covers quick fixes (30 min) to complex multi-day investigations (3 days)
RESOLUTION_TIME_MIN = 0.5
RESOLUTION_TIME_MAX = 72.0

# Synthetic fare/transaction amount range, in INR
AMOUNT_MIN_INR = 50.0
AMOUNT_MAX_INR = 5000.0

# Days since created: 0-30 days
DAYS_SINCE_CREATED_MIN = 0
DAYS_SINCE_CREATED_MAX = 30

# Minimum records per category
MIN_PER_CATEGORY = 3
# Minimum records per status
MIN_PER_STATUS = 1

# Total records to generate
TOTAL_RECORDS = 50


@dataclass
class SupportTicket:
    record_id: str
    category: str
    status: str
    resolution_time_hours: float
    days_since_created: int
    escalated: bool
    amount_inr: float = AMOUNT_MIN_INR


def generate_tickets(
    num_records: int = TOTAL_RECORDS,
    seed: int = SEED
) -> List[SupportTicket]:
    """Generate support tickets with deterministic randomness."""
    rng = random.Random(seed)
    amount_rng = random.Random(seed + 1)
    tickets = []
    
    for i in range(num_records):
        category = rng.choices(CATEGORIES, weights=CATEGORY_WEIGHTS, k=1)[0]
        status = rng.choices(STATUSES, weights=STATUS_WEIGHTS, k=1)[0]
        escalated = rng.random() < ESCALATED_PROB
        resolution_time = round(rng.uniform(RESOLUTION_TIME_MIN, RESOLUTION_TIME_MAX), 2)
        days_since_created = rng.randint(DAYS_SINCE_CREATED_MIN, DAYS_SINCE_CREATED_MAX)
        amount_inr = round(amount_rng.uniform(AMOUNT_MIN_INR, AMOUNT_MAX_INR), 2)
        
        ticket = SupportTicket(
            record_id=f"TKT-{i+1:04d}",
            category=category,
            status=status,
            resolution_time_hours=resolution_time,
            days_since_created=days_since_created,
            escalated=escalated,
            amount_inr=amount_inr,
        )
        tickets.append(ticket)
    
    return tickets


SUPPORT_TICKETS = generate_tickets()


def validate_tickets(tickets: List[SupportTicket]) -> Dict[str, Any]:
    """Validate the generated tickets meet all constraints."""
    category_counts = {c: 0 for c in CATEGORIES}
    status_counts = {s: 0 for s in STATUSES}
    escalated_count = 0
    
    for t in tickets:
        category_counts[t.category] += 1
        status_counts[t.status] += 1
        if t.escalated:
            escalated_count += 1
    
    total = len(tickets)
    escalated_pct = (escalated_count / total) * 100
    
    return {
        "total": total,
        "category_counts": category_counts,
        "status_counts": status_counts,
        "escalated_count": escalated_count,
        "escalated_percentage": round(escalated_pct, 2),
        "categories_ok": all(v >= MIN_PER_CATEGORY for v in category_counts.values()),
        "statuses_ok": all(v >= MIN_PER_STATUS for v in status_counts.values()),
        "escalated_in_range": 10 <= escalated_pct <= 30,
    }


def print_validation_report(validation: Dict[str, Any]) -> None:
    """Print a formatted validation report."""
    print("=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"Total records: {validation['total']}")
    print(f"\nCategory counts (min {MIN_PER_CATEGORY} each):")
    for cat, count in validation['category_counts'].items():
        status = "OK" if count >= MIN_PER_CATEGORY else "FAIL"
        print(f"  {cat}: {count} {status}")
    
    print(f"\nStatus counts (min {MIN_PER_STATUS} each):")
    for status, count in validation['status_counts'].items():
        status_mark = "OK" if count >= MIN_PER_STATUS else "FAIL"
        print(f"  {status}: {count} {status_mark}")
    
    print(f"\nEscalated: {validation['escalated_count']} ({validation['escalated_percentage']}%) - ", end="")
    print("OK (10-30%)" if validation['escalated_in_range'] else "FAIL (outside 10-30%)")
    
    all_ok = (validation['categories_ok'] and 
              validation['statuses_ok'] and 
              validation['escalated_in_range'])
    print(f"\nAll constraints satisfied: {'YES' if all_ok else 'NO'}")


def main():
    """Main function to generate and validate tickets."""
    os.makedirs("transcripts", exist_ok=True)
    
    tickets = generate_tickets()
    validation = validate_tickets(tickets)
    print_validation_report(validation)
    
    # Save tickets to JSON
    tickets_data = [asdict(t) for t in tickets]
    with open("transcripts/dataset.json", "w", encoding="utf-8") as f:
        json.dump(tickets_data, f, indent=2)
    
    # Save validation report
    with open("transcripts/dataset_validation.txt", "w", encoding="utf-8") as f:
        f.write("DATASET VALIDATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Total records: {validation['total']}\n")
        f.write(f"Category weights: {dict(zip(CATEGORIES, CATEGORY_WEIGHTS))}\n")
        f.write(f"Status weights: {dict(zip(STATUSES, STATUS_WEIGHTS))}\n")
        f.write(f"Escalated probability: {ESCALATED_PROB}\n")
        f.write(f"Resolution time range: {RESOLUTION_TIME_MIN}-{RESOLUTION_TIME_MAX} hours\n")
        f.write(f"Reasoning: Covers quick fixes (30 min) to complex multi-day investigations (3 days)\n")
        f.write(f"Synthetic amount range: INR {AMOUNT_MIN_INR}-{AMOUNT_MAX_INR}\n")
        f.write(f"Days since created range: {DAYS_SINCE_CREATED_MIN}-{DAYS_SINCE_CREATED_MAX} days\n")
        f.write("\nCategory counts:\n")
        for cat, count in validation['category_counts'].items():
            f.write(f"  {cat}: {count}\n")
        f.write("\nStatus counts:\n")
        for status, count in validation['status_counts'].items():
            f.write(f"  {status}: {count}\n")
        f.write(f"\nEscalated: {validation['escalated_count']} ({validation['escalated_percentage']}%)\n")
        f.write(f"Escalated in range (10-30%): {validation['escalated_in_range']}\n")
    
    print("\nTranscripts saved to transcripts/")
    
    if not (validation['categories_ok'] and validation['statuses_ok'] and validation['escalated_in_range']):
        raise SystemExit("Validation failed!")


if __name__ == "__main__":
    main()