"""Generate a small SYNTHETIC review file for tests, CI and the dashboard demo.

These reviews are invented from templates. They are NOT real customer
reviews and the numbers they produce say nothing about real couriers.
Use the real Kaggle data (see README) for any actual findings.

    python scripts/make_demo_data.py
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

SEED = 7
N = 900
OUT = Path(__file__).resolve().parents[1] / "data" / "sample" / "demo_reviews_synthetic.csv"

COURIERS = ["Evri", "DPD", "Royal Mail", "Yodel", "Amazon delivery", "DHL", "InPost"]
WEIGHTS = [0.27, 0.22, 0.2, 0.12, 0.09, 0.06, 0.04]
SHOPS = ["an online clothes shop", "a homeware store", "a pet supplies site", "a tech retailer",
         "a garden centre", "a bookshop", "a sports shop"]

POSITIVE = [
    "{c} delivered right on time and the driver was really friendly.",
    "Parcel arrived a day early with {c}, brilliant service.",
    "Great tracking from {c}, I got a one hour slot and they stuck to it.",
    "The {c} courier was polite and handed it to me at the door. Box was in perfect condition.",
    "Quick delivery by {c}, kept me informed the whole way.",
    "Really impressed, {c} turned up exactly when the app said.",
    "Lovely {c} driver, smiled and even waited while I got to the door.",
    "Fast and reliable as always with {c}, no complaints at all.",
]
NEGATIVE = [
    "{c} marked my parcel as delivered but it never arrived.",
    "The {c} driver left it outside in the rain and the box was soaked.",
    "Item arrived damaged, {c} must have thrown it over the fence.",
    "Waited in all day and {c} said they attempted delivery. Total lie, nobody knocked.",
    "Three days late with {c} and customer service just gave me a chatbot.",
    "Parcel lost by {c}. Tried to call them, impossible to speak to a person.",
    "{c} dumped it on the doorstep and walked off, no photo, no knock.",
    "Tracking with {c} said out for delivery for a week. Useless.",
    "Box was crushed and the contents broken. {c} is a joke.",
    "Delivered to the wrong address by {c}, the neighbour had to bring it round.",
    "Still waiting for a refund after {c} lost my order. Avoid.",
]
NEUTRAL = [
    "Delivery came with {c}, took about the expected time.",
    "{c} delivered it. Took a couple of days.",
    "It came via {c}, the parcel was fine but tracking was a bit vague.",
]
# Retailer reviews that blame the courier (common in real Trustpilot data)
BLAME = [
    "Product is great but the delivery with {c} was awful.",
    "Lovely item from {shop}. Shame they use {c}, it turned up two days late.",
    "Good quality, but please stop using {c}. Mine was left in the bin.",
]
OPENERS = ["", "Ordered on Monday.", "Second order this month.", "Bought a gift for my mum.",
           "Ordered some shoes.", "Needed this for work.", "First time ordering.", "Ordered a new kettle.",
           "Got a birthday present delivered.", "Ordered dog food."]
CLOSERS = ["", "Thanks.", "Will see next time.", "Just saying.", "Five minutes from the depot too.",
           "That's my experience anyway.", "Hope this helps someone."]
TITLES = {"positive": ["Great delivery", "Spot on", "Very happy", "Excellent"],
          "negative": ["Terrible", "Never again", "Awful service", "Not happy"],
          "neutral": ["OK", "Fine", "Average"]}

# Per-courier chance of a negative review, so the demo has some contrast
P_NEG = {"Evri": 0.55, "DPD": 0.22, "Royal Mail": 0.4, "Yodel": 0.6, "Amazon delivery": 0.35,
         "DHL": 0.3, "InPost": 0.25}


def main():
    rng = random.Random(SEED)
    rows = []
    start = pd.Timestamp("2025-06-01")
    for i in range(N):
        c = rng.choices(COURIERS, WEIGHTS)[0]
        r = rng.random()
        if r < P_NEG[c]:
            tone = "negative"
            text = rng.choice(NEGATIVE + BLAME).format(c=c, shop=rng.choice(SHOPS))
            if rng.random() < 0.4:
                text += " " + rng.choice(NEGATIVE).format(c=c)
            rating = rng.choices([1, 2, 3], [0.7, 0.22, 0.08])[0]
        elif r < P_NEG[c] + 0.1:
            tone = "neutral"
            text = rng.choice(NEUTRAL).format(c=c)
            rating = rng.choices([2, 3, 4], [0.2, 0.6, 0.2])[0]
        else:
            tone = "positive"
            text = rng.choice(POSITIVE).format(c=c)
            if rng.random() < 0.3:
                text += " " + rng.choice(POSITIVE).format(c=c)
            rating = rng.choices([3, 4, 5], [0.05, 0.25, 0.7])[0]
        if c == "Evri" and rng.random() < 0.15:
            text = text.replace("Evri", "Hermes")
        # Openers/closers add variety so de-duplication keeps most rows
        text = f"{rng.choice(OPENERS)} {text} {rng.choice(CLOSERS)}".strip()
        company = c.lower().replace(" ", "") + ".co.uk" if rng.random() < 0.6 else rng.choice(SHOPS)
        rows.append({
            "Company": company,
            "Title": rng.choice(TITLES[tone]),
            "Review": text,
            "Rating": rating,
            "Date": (start + pd.Timedelta(days=rng.randint(0, 540))).date().isoformat(),
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Wrote {len(rows)} synthetic reviews to {OUT}")


if __name__ == "__main__":
    main()
