"""Text cleaning.

Two versions of each review are kept on purpose:

* ``text_model``: light clean only (URLs, emails, whitespace). VADER and
  transformer models use capitals, punctuation and emoji as signal
  ("NEVER again!!!" is more negative than "never again"), so we keep them.
* ``text_clean``: lowercased letters only, for TF-IDF and topic modelling.

The original project ran sentiment on the heavily stripped text and
overwrote the raw text in place, which lost both of these.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
# "20th", "1st", "3rd" -> removed, otherwise "th"/"st" survive as junk tokens
ORDINAL_RE = re.compile(r"\b\d+(st|nd|rd|th)\b")
WS_RE = re.compile(r"\s+")
# Trustpilot-style truncation marker
SEE_MORE_RE = re.compile(r"\.{3}\s*see more\s*$", re.IGNORECASE)


def light_clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = SEE_MORE_RE.sub("", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip()


def deep_clean(text: str) -> str:
    text = light_clean(text).lower()
    text = text.replace("'", "")  # don't -> dont, keeps the token together
    text = ORDINAL_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip()


def clean_reviews(df: pd.DataFrame, min_words: int = 3) -> pd.DataFrame:
    """Combine title + body, clean, drop empties and exact duplicates."""
    df = df.copy()
    title = df["title"].fillna("").astype(str).str.strip()
    body = df["text"].fillna("").astype(str).str.strip()
    # Add the title only when it isn't just the start of the body
    title_is_prefix = [b.lower().startswith(t.lower().rstrip(". ")) for t, b in zip(title, body)]
    use_title = (title != "") & ~pd.Series(title_is_prefix, index=df.index, dtype=bool)
    full = body.where(~use_title, title + ". " + body)

    df["text_model"] = full.map(light_clean)
    df["text_clean"] = df["text_model"].map(deep_clean)
    df["n_words"] = df["text_clean"].str.split().str.len().fillna(0).astype(int)

    before = len(df)
    df = df[df["n_words"] >= min_words]
    df = df.drop_duplicates(subset=["text_clean"])
    df.attrs["dropped"] = before - len(df)
    return df.reset_index(drop=True)
