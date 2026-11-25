"""Topic modelling: the recurring themes in complaints and in praise.

Default is TF-IDF + NMF (fast, deterministic, runs anywhere). BERTopic is
available as an option for larger datasets (``--bertopic``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

# Words that appear everywhere in courier reviews and would dominate every
# topic without saying anything. Courier names are removed so topics describe
# *what happened*, not *who*.
DOMAIN_STOP_WORDS = {
    "evri", "hermes", "dpd", "royal", "mail", "yodel", "dhl", "ups", "fedex", "inpost", "parcelforce",
    "amazon", "parcel", "parcels", "package", "packages", "order", "ordered", "delivery", "delivered",
    "deliver", "service", "company", "courier", "couriers", "item", "items", "just", "got", "get",
    "im", "ive", "dont", "didnt", "wasnt", "youre", "theyre", "cant", "wont", "said", "told", "time",
    "day", "days", "today", "yesterday", "week", "really", "use", "used", "using", "thank", "thanks",
}
STOP_WORDS = sorted(ENGLISH_STOP_WORDS | DOMAIN_STOP_WORDS)


def fit_topics(texts: pd.Series, n_topics: int = 6, n_words: int = 8, random_state: int = 42):
    """Return (topic_table, doc_topic_ids). Scales min_df to the corpus size."""
    texts = texts.fillna("")
    n_docs = len(texts)
    if n_docs < n_topics * 3:
        return pd.DataFrame(columns=["topic", "top_words", "n_reviews", "share"]), np.full(n_docs, -1)

    vec = TfidfVectorizer(
        stop_words=STOP_WORDS,
        ngram_range=(1, 2),
        min_df=max(2, min(10, n_docs // 200)),
        max_df=0.6,
        max_features=5000,
        sublinear_tf=True,
    )
    X = vec.fit_transform(texts)
    n_topics = min(n_topics, X.shape[1] - 1)
    model = NMF(n_components=n_topics, init="nndsvda", random_state=random_state, max_iter=400)
    W = model.fit_transform(X)
    vocab = np.array(vec.get_feature_names_out())

    doc_topic = np.where(W.max(axis=1) > 0, W.argmax(axis=1), -1)
    rows = []
    for k, weights in enumerate(model.components_):
        top = vocab[np.argsort(weights)[::-1][:n_words]]
        n = int((doc_topic == k).sum())
        rows.append({"topic": k, "top_words": ", ".join(top), "n_reviews": n, "share": n / n_docs})
    table = pd.DataFrame(rows).sort_values("n_reviews", ascending=False).reset_index(drop=True)
    return table, doc_topic


def fit_bertopic(texts: pd.Series):  # pragma: no cover - optional, heavy
    try:
        from bertopic import BERTopic
    except ImportError as exc:
        raise ImportError("Install the extras first: pip install -r requirements-extra.txt") from exc
    model = BERTopic(min_topic_size=15, verbose=False)
    topics, _ = model.fit_transform(texts.fillna("").tolist())
    info = model.get_topic_info().rename(columns={"Topic": "topic", "Count": "n_reviews", "Name": "top_words"})
    info = info[info["topic"] != -1]
    info["share"] = info["n_reviews"] / len(texts)
    return info[["topic", "top_words", "n_reviews", "share"]].reset_index(drop=True), np.array(topics)
