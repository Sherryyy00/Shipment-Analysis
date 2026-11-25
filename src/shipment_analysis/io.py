"""Loading review data and mapping it onto one standard schema.

Every downstream step works on these columns:
    review_id, company, category, title, text, rating, date
Only ``text`` is required. The others are used when present.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Common column names seen in review datasets (Kaggle / Trustpilot exports).
# The first match wins, case-insensitive.
COLUMN_CANDIDATES: dict[str, list[str]] = {
    "text": ["text", "review", "review_text", "reviewbody", "review_body", "content", "body", "comment"],
    "title": ["title", "review_title", "headline", "summary"],
    "rating": ["rating", "stars", "star_rating", "score", "review_rating", "rating_value"],
    "date": ["date", "review_date", "published", "published_date", "date_published", "created_at", "time"],
    "company": ["company", "company_name", "business", "business_name", "domain", "brand"],
    "category": ["category", "company_category", "industry", "sector"],
}


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {c.lower().strip(): c for c in df.columns}
    for name in candidates:
        if name in lookup:
            return lookup[name]
    return None


def read_any(path: str | Path) -> pd.DataFrame:
    """Read CSV, Excel, JSON(L) or Parquet based on the file extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def standardise(df: pd.DataFrame, mapping: dict[str, str] | None = None) -> pd.DataFrame:
    """Rename columns to the standard schema.

    ``mapping`` overrides auto-detection, e.g. ``{"text": "reviewBody"}``.
    """
    mapping = dict(mapping or {})
    for target, candidates in COLUMN_CANDIDATES.items():
        if target not in mapping:
            found = _find_column(df, candidates)
            if found is not None:
                mapping[target] = found

    if "text" not in mapping:
        raise KeyError(
            "Could not find a review text column. Pass it explicitly, e.g. --text-col review_text. "
            f"Columns available: {list(df.columns)}"
        )

    out = pd.DataFrame({target: df[source] for target, source in mapping.items()})
    for col in ["title", "company", "category"]:
        if col not in out:
            out[col] = ""
    out["title"] = out["title"].fillna("").astype(str)
    out["text"] = out["text"].fillna("").astype(str)

    if "rating" in out:
        # Handles "5", 5.0 and "Rated 5 out of 5 stars"
        out["rating"] = pd.to_numeric(
            out["rating"].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce"
        )
    else:
        out["rating"] = pd.NA

    if "date" in out:
        out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True).dt.tz_localize(None)
    else:
        out["date"] = pd.NaT

    out.insert(0, "review_id", range(len(out)))
    return out[["review_id", "company", "category", "title", "text", "rating", "date"]]


def load_reviews(path: str | Path, mapping: dict[str, str] | None = None) -> pd.DataFrame:
    return standardise(read_any(path), mapping)
