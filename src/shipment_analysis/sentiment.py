"""Sentiment scoring with three models.

* VADER: rule-based, built for short social text, uses caps and punctuation.
* TextBlob: lexicon average, gives polarity and subjectivity.
* Transformer (optional): RoBERTa fine-tuned on ~124M tweets
  (cardiffnlp/twitter-roberta-base-sentiment-latest). Needs
  ``pip install -r requirements-extra.txt`` and an internet connection
  the first time, to download the model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

LABELS = ["negative", "neutral", "positive"]
TRANSFORMER_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

_vader = SentimentIntensityAnalyzer()  # build once, not once per row

# Delivery words that stock VADER treats as neutral. "Parcel never arrived,
# marked as delivered" scores 0.0 with plain VADER, which is plainly wrong for
# a courier review. Values use VADER's -4..+4 scale.
COURIER_LEXICON: dict[str, float] = {
    "lost": -2.0, "missing": -1.8, "damaged": -2.2, "smashed": -2.5, "crushed": -2.0,
    "broken": -1.8, "stolen": -2.6, "late": -1.5, "delayed": -1.6, "delay": -1.4,
    "delays": -1.4, "dumped": -2.2, "thrown": -1.6, "chucked": -1.8, "abandoned": -2.0,
    "unattended": -1.2, "liar": -2.6, "lied": -2.4, "lying": -2.2, "scam": -3.0,
    "fake": -2.0, "falsely": -2.0, "refund": -0.6, "complaint": -1.6, "chasing": -1.2,
    "rude": -2.2, "careless": -2.0, "unreliable": -2.2, "unacceptable": -2.6,
    "nightmare": -2.8, "shambles": -2.6, "joke": -1.2, "rubbish": -2.4, "useless": -2.4,
    "avoid": -2.0, "undelivered": -2.0, "redelivery": -0.8,
    "early": 1.4, "punctual": 2.0, "prompt": 1.6, "speedy": 1.8, "courteous": 2.2,
    "efficient": 1.8, "seamless": 2.0, "reliable": 2.0, "informed": 1.0,
}

_vader_courier = SentimentIntensityAnalyzer()
_vader_courier.lexicon.update(COURIER_LEXICON)


def vader_compound(text: str) -> float:
    return _vader.polarity_scores(str(text))["compound"]


def to_label(score: float, threshold: float = 0.05) -> str:
    """Standard VADER cut-offs: <= -0.05 negative, >= 0.05 positive."""
    if pd.isna(score):
        return "neutral"
    if score >= threshold:
        return "positive"
    if score <= -threshold:
        return "negative"
    return "neutral"


def stars_to_label(rating: float) -> str | None:
    """Star rating as a weak ground-truth label: 1-2 neg, 3 neutral, 4-5 pos."""
    if pd.isna(rating):
        return None
    if rating <= 2:
        return "negative"
    if rating >= 4:
        return "positive"
    return "neutral"


def add_lexicon_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    scores = df["text_model"].map(_vader.polarity_scores).apply(pd.Series)
    df["vader_neg"] = scores["neg"]
    df["vader_neu"] = scores["neu"]
    df["vader_pos"] = scores["pos"]
    df["vader_compound"] = scores["compound"]
    df["vader_label"] = df["vader_compound"].map(to_label)

    df["vaderx_compound"] = df["text_model"].map(lambda t: _vader_courier.polarity_scores(t)["compound"])
    df["vaderx_label"] = df["vaderx_compound"].map(to_label)

    blobs = df["text_model"].map(lambda t: TextBlob(t).sentiment)
    df["tb_polarity"] = blobs.map(lambda s: s.polarity)
    df["tb_subjectivity"] = blobs.map(lambda s: s.subjectivity)
    df["tb_label"] = df["tb_polarity"].map(lambda s: to_label(s, threshold=0.05))

    df["star_label"] = df["rating"].map(stars_to_label)
    return df


def add_transformer_sentiment(df: pd.DataFrame, batch_size: int = 32, device: int | None = None) -> pd.DataFrame:
    """Adds ``tf_label`` and ``tf_score`` (P(pos) - P(neg), range -1..1)."""
    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install the extras first: pip install -r requirements-extra.txt") from exc

    if device is None:
        try:
            import torch

            device = 0 if torch.cuda.is_available() else -1
        except ImportError:  # pragma: no cover
            device = -1

    clf = pipeline("sentiment-analysis", model=TRANSFORMER_MODEL, top_k=None, device=device)
    outputs = clf(df["text_model"].tolist(), batch_size=batch_size, truncation=True, max_length=512)

    df = df.copy()
    probs = [{d["label"].lower(): d["score"] for d in row} for row in outputs]
    df["tf_label"] = [max(p, key=p.get) for p in probs]
    df["tf_score"] = [p.get("positive", 0) - p.get("negative", 0) for p in probs]
    return df


def sentiment_columns(df: pd.DataFrame) -> list[str]:
    """Which model label columns exist in this frame."""
    return [c for c in ["vader_label", "vaderx_label", "tb_label", "tf_label"] if c in df]


MODEL_NAMES = {
    "vader_label": "VADER",
    "vaderx_label": "VADER + courier lexicon",
    "tb_label": "TextBlob",
    "tf_label": "RoBERTa (transformer)",
}

SCORE_COLUMNS = {
    "vader_label": "vader_compound",
    "vaderx_label": "vaderx_compound",
    "tb_label": "tb_polarity",
    "tf_label": "tf_score",
}


def best_score_column(df: pd.DataFrame) -> str:
    """Transformer score if it was run, otherwise the courier-tuned VADER."""
    return "tf_score" if "tf_score" in df else "vaderx_compound"


def net_sentiment(labels: pd.Series) -> float:
    """% positive minus % negative, like an NPS-style headline number."""
    if len(labels) == 0:
        return np.nan
    return 100 * ((labels == "positive").mean() - (labels == "negative").mean())
