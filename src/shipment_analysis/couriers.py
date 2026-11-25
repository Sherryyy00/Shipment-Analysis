"""Find which UK couriers a review mentions."""
from __future__ import annotations

import re

import pandas as pd

# Order here is also the display order in charts.
COURIER_PATTERNS: dict[str, re.Pattern] = {
    "Evri": re.compile(r"\bevri\b|\bhermes\b|\bmyhermes\b", re.IGNORECASE),
    "DPD": re.compile(r"\bdpd\b", re.IGNORECASE),
    "Royal Mail": re.compile(r"\broyal\s*mail\b", re.IGNORECASE),
    "Yodel": re.compile(r"\byodel\b", re.IGNORECASE),
    "Amazon Logistics": re.compile(r"\bamazon\s+(logistics|driver|delivery|courier|van)s?\b", re.IGNORECASE),
    "DHL": re.compile(r"\bdhl\b", re.IGNORECASE),
    "Parcelforce": re.compile(r"\bparcel\s*force\b", re.IGNORECASE),
    "InPost": re.compile(r"\bin\s?post\b(?!\s+(code|office))", re.IGNORECASE),
    # Case-sensitive so "ups and downs" / "mix-ups" don't count
    "UPS": re.compile(r"\bUPS\b"),
    "FedEx": re.compile(r"\bfed\s?ex\b", re.IGNORECASE),
}

COURIER_ORDER = list(COURIER_PATTERNS)


def find_couriers(text: str) -> list[str]:
    return [name for name, pat in COURIER_PATTERNS.items() if pat.search(str(text))]


def courier_from_company(company: str) -> str | None:
    """If the review is *about* a courier (e.g. Trustpilot page for evri.com)."""
    hits = find_couriers(str(company).replace(".", " "))
    return hits[0] if hits else None


def tag_couriers(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``couriers`` (list) and ``n_couriers`` columns."""
    df = df.copy()

    def _tag(row) -> list[str]:
        found = find_couriers(row["text_model"])
        own = courier_from_company(row.get("company", ""))
        if own and own not in found:
            found.insert(0, own)
        return found

    df["couriers"] = df.apply(_tag, axis=1)
    df["n_couriers"] = df["couriers"].str.len()
    return df


def explode_couriers(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (review, courier). Reviews with no courier are dropped."""
    out = df[df["n_couriers"] > 0].explode("couriers").rename(columns={"couriers": "courier"})
    return out.reset_index(drop=True)


def courier_snippets(df: pd.DataFrame, window: int = 1) -> pd.DataFrame:
    """Text about each courier only: sentences naming it plus the next ``window`` sentences.

    Most real reviews are on a *retailer's* page ("Lovely dress, shame they use
    Evri. It arrived late."). Scoring the whole review mixes the shop's
    product and service into the courier's score, so courier-level results use
    these snippets instead. If the review is on the courier's own page (company
    column), the whole review is about that courier.
    """
    from .aspects import split_sentences  # local import avoids a cycle

    rows = []
    for rec in df[df["n_couriers"] > 0].itertuples():
        own = courier_from_company(getattr(rec, "company", ""))
        sents = split_sentences(rec.text_model)
        for courier in rec.couriers:
            if courier == own:
                snippet, scope = rec.text_model, "whole review"
            else:
                pat = COURIER_PATTERNS[courier]
                keep: set[int] = set()
                for i, s in enumerate(sents):
                    if pat.search(s):
                        keep.update(range(i, min(len(sents), i + window + 1)))
                snippet = " ".join(sents[i] for i in sorted(keep)) or rec.text_model
                scope = "courier sentences"
            rows.append({"review_id": rec.review_id, "courier": courier, "courier_text": snippet,
                         "courier_scope": scope})
    return pd.DataFrame(rows, columns=["review_id", "courier", "courier_text", "courier_scope"])
