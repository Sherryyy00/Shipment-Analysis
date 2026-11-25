"""Aspect-based sentiment: *what* about the delivery people liked or hated.

Approach (simple, explainable, no labelled data needed):
1. Split each review into sentences.
2. Tag a sentence with an aspect when it matches that aspect's keywords.
3. Score that sentence with the courier-tuned VADER.

So "Driver was lovely but the box was crushed" gives
driver = positive, parcel condition = negative, instead of one blended score.
"""
from __future__ import annotations

import re

import pandas as pd

from .sentiment import _vader_courier, to_label

ASPECTS: dict[str, re.Pattern] = {
    "Delivery speed": re.compile(
        r"\b(late|early|on time|delay\w*|quick\w*|fast\w*|slow\w*|took \w+ (days|weeks)|next day|same day|"
        r"waiting|waited|days? later|weeks?|eta)\b", re.I),
    "Lost / not delivered": re.compile(
        r"\b(lost|missing|never (arrived|came|received|turned up|delivered)|not (arrived|received|delivered)|"
        r"didn'?t (arrive|receive|turn up)|marked as delivered|says? delivered|undelivered|stolen)\b", re.I),
    "Parcel condition": re.compile(
        r"\b(damaged|damage|broken|crushed|smashed|soaked|wet|torn|dented|ripped|condition|intact|well packaged)\b", re.I),
    "Driver behaviour": re.compile(
        r"\b(driver|courier|delivery (man|guy|lady|person|woman)|postman|postie|polite|rude|friendly|helpful|"
        r"courteous|threw|thrown|dumped|chucked|knock\w*|doorbell|smil\w*)\b", re.I),
    "Safe place / doorstep": re.compile(
        r"\b(safe ?place|doorstep|left (it )?(outside|on the step|in the bin|behind)|bin|neighbou?r|porch|"
        r"front door|unattended|photo)\b", re.I),
    "Tracking & updates": re.compile(
        r"\b(track\w*|updates?|notif\w*|text message|texts?|emails?|app|time slot|one hour|1 hour|hour slot|kept me informed)\b", re.I),
    "Customer service": re.compile(
        r"\b(customer service|customer services|support|chat ?bot|chatbot|helpline|phone(d)?|call(ed|ing)?|"
        r"complain\w*|refund\w*|claim|compensation|reply|replied|response|ticket|contact\w*)\b", re.I),
    "Price & value": re.compile(
        r"\b(price|cheap\w*|expensive|cost\w*|value|money|fee|charged?|surcharge|customs)\b", re.I),
}

ASPECT_ORDER = list(ASPECTS)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+|\s+but\s+", re.I)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT_RE.split(str(text)) if s and len(s.strip()) > 2]


def aspect_mentions(df: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per (review, aspect, sentence) with a sentiment score."""
    rows = []
    for rid, text in zip(df["review_id"], df["text_model"]):
        for sent in split_sentences(text):
            hits = [name for name, pat in ASPECTS.items() if pat.search(sent)]
            if not hits:
                continue
            score = _vader_courier.polarity_scores(sent)["compound"]
            for name in hits:
                rows.append({"review_id": rid, "aspect": name, "sentence": sent, "aspect_score": score})
    out = pd.DataFrame(rows, columns=["review_id", "aspect", "sentence", "aspect_score"])
    out["aspect_label"] = out["aspect_score"].map(to_label)
    return out


def aspect_review_level(mentions: pd.DataFrame) -> pd.DataFrame:
    """Average the sentence scores so each review counts once per aspect."""
    if mentions.empty:
        return mentions.assign(aspect_score=pd.Series(dtype=float))
    out = mentions.groupby(["review_id", "aspect"], as_index=False)["aspect_score"].mean()
    out["aspect_label"] = out["aspect_score"].map(to_label)
    return out
