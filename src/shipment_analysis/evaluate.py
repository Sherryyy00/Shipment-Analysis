"""How good is each sentiment model? Star ratings act as the reference label.

Stars aren't perfect ground truth (people give 1 star to a retailer for a
courier's fault, or 3 stars with glowing text), so treat these scores as
"agreement with the reviewer's own rating", which is still the fairest
comparison available without hand-labelling.
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .sentiment import LABELS, MODEL_NAMES, sentiment_columns


def model_scores(df: pd.DataFrame) -> pd.DataFrame:
    rated = df.dropna(subset=["star_label"])
    rows = []
    for col in sentiment_columns(rated):
        rows.append({
            "model": MODEL_NAMES[col],
            "accuracy": accuracy_score(rated["star_label"], rated[col]),
            "macro_f1": f1_score(rated["star_label"], rated[col], labels=LABELS, average="macro", zero_division=0),
            "negative_f1": f1_score(rated["star_label"], rated[col], labels=["negative"], average="macro",
                                    zero_division=0),
            "n": len(rated),
        })
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)


def confusion(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rated = df.dropna(subset=["star_label"])
    cm = confusion_matrix(rated["star_label"], rated[col], labels=LABELS)
    return pd.DataFrame(cm, index=[f"stars: {l}" for l in LABELS], columns=[f"model: {l}" for l in LABELS])


def disagreements(df: pd.DataFrame, col: str, n: int = 10) -> pd.DataFrame:
    """Most confident model mistakes, useful for spotting sarcasm and blame-shifting."""
    rated = df.dropna(subset=["star_label"])
    wrong = rated[(rated["star_label"] == "negative") & (rated[col] == "positive")]
    score_col = {"vader_label": "vader_compound", "vaderx_label": "vaderx_compound",
                 "tb_label": "tb_polarity", "tf_label": "tf_score"}[col]
    return wrong.sort_values(score_col, ascending=False).head(n)[["rating", score_col, "text_model"]]
