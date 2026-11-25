"""Run the full analysis end to end.

    python -m shipment_analysis.pipeline --input data/raw/trustpilot_reviews.csv

Writes CSVs to <out>/processed, charts to <out>/figures and a short
<out>/summary.md with the headline numbers.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from . import plots
from .aspects import ASPECT_ORDER, aspect_mentions
from .clean import clean_reviews, deep_clean
from .couriers import COURIER_ORDER, courier_snippets, explode_couriers, tag_couriers
from .evaluate import model_scores
from .io import load_reviews
from .sentiment import (MODEL_NAMES, _vader_courier, to_label, add_lexicon_sentiment, add_transformer_sentiment, best_score_column,
                        net_sentiment, sentiment_columns)
from .topics import fit_bertopic, fit_topics

log = logging.getLogger("shipment_analysis")


def best_label_column(df: pd.DataFrame) -> str:
    return "tf_label" if "tf_label" in df else "vaderx_label"


def courier_summary(mentions: pd.DataFrame, label_col: str, score_col: str, min_n: int) -> pd.DataFrame:
    g = mentions.groupby("courier")
    out = pd.DataFrame({
        "n_reviews": g.size(),
        "avg_rating": g["rating"].mean(),
        "mean_score": g[score_col].mean(),
        "pct_negative": g[label_col].apply(lambda s: 100 * (s == "negative").mean()),
        "pct_positive": g[label_col].apply(lambda s: 100 * (s == "positive").mean()),
        "net_sentiment": g[label_col].apply(net_sentiment),
    }).reset_index()
    out = out[out["n_reviews"] >= min_n]
    order = {c: i for i, c in enumerate(COURIER_ORDER)}
    return out.sort_values("courier", key=lambda s: s.map(order)).reset_index(drop=True)


def run(args: argparse.Namespace) -> dict:
    t0 = time.time()
    out = Path(args.out)
    (out / "processed").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    mapping = {k: v for k, v in {"text": args.text_col, "rating": args.rating_col, "date": args.date_col,
                                  "title": args.title_col, "company": args.company_col}.items() if v}
    df = load_reviews(args.input, mapping)
    log.info("Loaded %s reviews from %s", f"{len(df):,}", args.input)

    df = clean_reviews(df, min_words=args.min_words)
    log.info("After cleaning: %s reviews (%s dropped as empty/short/duplicate)",
             f"{len(df):,}", f"{df.attrs.get('dropped', 0):,}")

    df = tag_couriers(df)
    if not args.all_reviews:
        df = df[df["n_couriers"] > 0].reset_index(drop=True)
        log.info("Kept %s reviews that mention a courier", f"{len(df):,}")
    if args.sample and len(df) > args.sample:
        df = df.sample(args.sample, random_state=42).reset_index(drop=True)
        log.info("Sampled down to %s reviews", f"{len(df):,}")
    if df.empty:
        raise SystemExit("No reviews left to analyse. Try --all-reviews or check the text column.")

    df = add_lexicon_sentiment(df)
    if args.transformer:
        log.info("Running transformer model (first run downloads ~500MB)...")
        df = add_transformer_sentiment(df)

    label_col, score_col = best_label_column(df), best_score_column(df)
    log.info("Headline model: %s", MODEL_NAMES[label_col])

    # Courier-focused sentiment: score only the text about each courier
    snips = courier_snippets(df)
    snips["courier_score"] = snips["courier_text"].map(lambda t: _vader_courier.polarity_scores(t)["compound"])
    snips["courier_label"] = snips["courier_score"].map(to_label)
    mentions = explode_couriers(df).merge(snips, on=["review_id", "courier"], how="left")
    summary = courier_summary(mentions, "courier_label", "courier_score", args.min_courier_n)
    review_summary = courier_summary(mentions, label_col, score_col, args.min_courier_n)
    summary["net_whole_review"] = summary["courier"].map(review_summary.set_index("courier")["net_sentiment"])
    keep = summary["courier"].tolist()

    # Aspects, also on courier text only, so the shop's own customer service
    # or product quality isn't blamed on the courier
    snip_in = snips.reset_index().rename(columns={"index": "snip_id"})
    asp = aspect_mentions(snip_in.rename(columns={"review_id": "orig_review_id", "snip_id": "review_id",
                                                  "courier_text": "text_model"}))
    asp = asp.rename(columns={"review_id": "snip_id"}).merge(
        snip_in[["snip_id", "review_id", "courier", "courier_label"]], on="snip_id").drop(columns="snip_id")
    asp_c = (asp.groupby(["review_id", "courier", "aspect"], as_index=False)
             .agg(aspect_score=("aspect_score", "mean"), courier_label=("courier_label", "first")))
    asp_c["aspect_label"] = asp_c["aspect_score"].map(to_label)
    asp_c = asp_c[asp_c["courier"].isin(keep)]
    heat = asp_c.pivot_table(index="courier", columns="aspect", values="aspect_score", aggfunc="mean")
    heat_n = asp_c.pivot_table(index="courier", columns="aspect", values="aspect_score", aggfunc="count")
    cols = [a for a in ASPECT_ORDER if a in heat.columns]
    rows = [c for c in keep if c in heat.index]
    heat, heat_n = heat.reindex(index=rows, columns=cols), heat_n.reindex(index=rows, columns=cols).fillna(0)

    neg_pairs = asp_c[asp_c["courier_label"] == "negative"]
    n_neg = max(1, int((mentions["courier_label"] == "negative").sum()))
    neg_share = (neg_pairs.groupby("aspect").size() / n_neg).reindex(ASPECT_ORDER).dropna()

    # Topics
    topic_fn = fit_bertopic if args.bertopic else (lambda t: fit_topics(t, n_topics=args.n_topics))
    mentions["courier_text_clean"] = mentions["courier_text"].fillna("").map(deep_clean)
    neg_m = mentions[mentions["courier_label"] == "negative"].drop_duplicates("review_id")
    pos_m = mentions[mentions["courier_label"] == "positive"].drop_duplicates("review_id")
    topics_neg, neg_ids = topic_fn(neg_m["courier_text_clean"])
    topics_pos, pos_ids = topic_fn(pos_m["courier_text_clean"])
    df["topic_negative"] = df["review_id"].map(pd.Series(neg_ids, index=neg_m["review_id"].values))
    df["topic_positive"] = df["review_id"].map(pd.Series(pos_ids, index=pos_m["review_id"].values))

    # Model evaluation
    scores = model_scores(df) if df["star_label"].notna().any() else pd.DataFrame()

    # Monthly trend
    monthly = pd.DataFrame()
    if mentions["date"].notna().any():
        m = mentions.dropna(subset=["date"]).assign(month=lambda d: d["date"].dt.to_period("M").dt.to_timestamp())
        monthly = (m.groupby(["month", "courier"])
                   .agg(n=("courier_label", "size"), net_sentiment=("courier_label", net_sentiment))
                   .reset_index())
        monthly = monthly[(monthly["n"] >= args.min_month_n) & monthly["courier"].isin(keep)]

    # Save tables
    p = out / "processed"
    save_cols = [c for c in df.columns if c != "text_clean"]
    df.assign(couriers=df["couriers"].str.join("|"))[save_cols].to_csv(p / "reviews_scored.csv", index=False)
    mentions.drop(columns=["text_clean", "courier_text_clean"]).to_csv(p / "courier_mentions.csv", index=False)
    asp.to_csv(p / "aspect_sentences.csv", index=False)
    asp_c.to_csv(p / "aspect_by_courier_long.csv", index=False)
    summary.to_csv(p / "courier_summary.csv", index=False)
    heat.round(3).to_csv(p / "aspect_by_courier.csv")
    topics_neg.to_csv(p / "topics_negative.csv", index=False)
    topics_pos.to_csv(p / "topics_positive.csv", index=False)
    if not scores.empty:
        scores.to_csv(p / "model_scores.csv", index=False)
    if not monthly.empty:
        monthly.to_csv(p / "monthly_trend.csv", index=False)

    # Charts
    f = out / "figures"
    if not summary.empty:
        plots.courier_net_sentiment(summary, f / "01_courier_net_sentiment.png")
    if not heat.empty:
        plots.aspect_heatmap(heat, heat_n, f / "02_aspect_heatmap.png", min_n=args.min_cell_n)
    if not neg_share.empty:
        plots.aspect_share_negative(neg_share, f / "03_negative_aspects.png")
    if not scores.empty:
        plots.model_comparison(scores, f / "04_model_comparison.png")
        plots.label_share_by_stars(df, label_col, MODEL_NAMES[label_col], f / "05_label_vs_stars.png")
    if not monthly.empty and monthly["month"].nunique() >= 3:
        plots.monthly_trend(monthly, f / "06_monthly_trend.png")

    results = dict(n_reviews=len(df), label_col=label_col, summary=summary, scores=scores,
                   topics_neg=topics_neg, topics_pos=topics_pos, neg_share=neg_share, heat=heat,
                   seconds=time.time() - t0)
    write_summary(results, out / "summary.md", args)
    log.info("Done in %.1fs. Outputs in %s", results["seconds"], out)
    return results


def write_summary(r: dict, path: Path, args) -> None:
    lines = ["# Results summary", "", f"- Input: `{args.input}`",
             f"- Reviews analysed: **{r['n_reviews']:,}**",
             f"- Headline sentiment model: **{MODEL_NAMES[r['label_col']]}**", ""]
    s = r["summary"]
    if not s.empty:
        lines += ["## Couriers", "",
                  "Net sentiment is scored on the sentences about each courier. "
                  "'Whole review' scores the full review, which includes the shop and product.", "",
                  "| Courier | Reviews | Avg ★ | % negative | Net sentiment | Whole review |",
                  "|---|---:|---:|---:|---:|---:|"]
        for _, row in s.sort_values("net_sentiment").iterrows():
            avg = "" if pd.isna(row["avg_rating"]) else f"{row['avg_rating']:.2f}"
            lines.append(f"| {row['courier']} | {row['n_reviews']:,} | {avg} | {row['pct_negative']:.0f}% | "
                         f"{row['net_sentiment']:+.0f} | {row['net_whole_review']:+.0f} |")
        lines.append("")
    if not r["neg_share"].empty:
        top = r["neg_share"].sort_values(ascending=False).head(3)
        lines += ["## Top complaint aspects", ""] + [f"- {a}: {v:.0%} of negative reviews" for a, v in top.items()]
        lines.append("")
    if not r["scores"].empty:
        lines += ["## Model agreement with star ratings", "", "| Model | Accuracy | Macro F1 |", "|---|---:|---:|"]
        lines += [f"| {m.model} | {m.accuracy:.0%} | {m.macro_f1:.2f} |" for m in r["scores"].itertuples()]
        lines.append("")
    for name, t in [("Complaint themes", r["topics_neg"]), ("Praise themes", r["topics_pos"])]:
        if not t.empty:
            lines += [f"## {name}", ""] + [f"- ({row.share:.0%}) {row.top_words}" for row in t.itertuples()]
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="UK courier review sentiment analysis")
    p.add_argument("--input", required=True, help="CSV / XLSX / JSON(L) / Parquet file of reviews")
    p.add_argument("--out", default="reports", help="Output folder (default: reports)")
    p.add_argument("--text-col"), p.add_argument("--rating-col"), p.add_argument("--date-col")
    p.add_argument("--title-col"), p.add_argument("--company-col")
    p.add_argument("--all-reviews", action="store_true", help="Keep reviews that mention no courier")
    p.add_argument("--sample", type=int, default=0, help="Randomly sample N reviews (quick runs)")
    p.add_argument("--transformer", action="store_true", help="Also run the RoBERTa model (needs extras)")
    p.add_argument("--bertopic", action="store_true", help="Use BERTopic instead of NMF (needs extras)")
    p.add_argument("--n-topics", type=int, default=6)
    p.add_argument("--min-words", type=int, default=3)
    p.add_argument("--min-courier-n", type=int, default=20, help="Min reviews for a courier to be reported")
    p.add_argument("--min-cell-n", type=int, default=10, help="Min mentions per heatmap cell")
    p.add_argument("--min-month-n", type=int, default=10)
    p.add_argument("-q", "--quiet", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s")
    run(args)


if __name__ == "__main__":
    main()
