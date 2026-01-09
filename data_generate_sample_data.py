"""Generate synthetic flight fares dataset.

Usage:
  python data/generate_sample_data.py --out data/flights.csv --n 5000
"""
import argparse
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

AIRPORTS = ["JFK", "LAX", "SFO", "ORD", "ATL", "MIA", "SEA", "BOS", "DEN", "LAS"]
AIRLINES = ["Delta", "United", "American", "Spirit", "JetBlue", "Southwest"]

def random_date(start_days=1, max_days=365):
    dt = datetime.utcnow() + timedelta(days=random.randint(start_days, max_days))
    return dt.date().isoformat()

def generate_row():
    origin = random.choice(AIRPORTS)
    dest = random.choice([a for a in AIRPORTS if a != origin])
    depart_date = random_date(1, 120)
    days_to_departure = random.randint(0, 120)
    airline = random.choice(AIRLINES)
    duration = random.randint(60, 720)  # minutes
    stops = random.choices([0,1,2], weights=[0.6, 0.3, 0.1])[0]
    # Base price by distance proxy (airport index difference)
    base_distance = abs(AIRPORTS.index(origin) - AIRPORTS.index(dest))
    base_price = 50 + base_distance * 40 + duration * 0.1
    # Seasonality: random monthly effect
    month = random.randint(1,12)
    month_factor = 1 + 0.1 * np.sin(month / 12 * 2 * np.pi)
    # urgency increases price
    urgency = 1.0 + 0.02 * max(0, 30 - days_to_departure)
    # stops lower price slightly
    stops_penalty = 1 - 0.1 * stops
    price = base_price * month_factor * urgency * stops_penalty
    # Airline bias
    if airline == "Spirit":
        price *= 0.8
    if airline == "Delta":
        price *= 1.05
    # add noise
    price = max(20, price + random.gauss(0, 25))
    return {
        "origin": origin,
        "destination": dest,
        "depart_date": depart_date,
        "days_to_departure": days_to_departure,
        "airline": airline,
        "duration": duration,
        "stops": stops,
        "price": round(price, 2),
    }

def main(out, n):
    rows = [generate_row() for _ in range(n)]
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/flights.csv")
    parser.add_argument("--n", type=int, default=5000)
    args = parser.parse_args()
    main(args.out, args.n)