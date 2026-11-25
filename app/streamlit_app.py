"""Interactive dashboard for the courier review analysis.

    streamlit run app/streamlit_app.py

Reads the CSVs written by the pipeline. Pick the results folder in the sidebar
(reports/full after running on the real data, reports/demo for the synthetic demo).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
BLUE, RED, GREY = "#257de0", "#ec4647", "#c9c9c9"
SENT = {"negative": RED, "neutral": GREY, "positive": BLUE}
DIVERGING = [[0, "#c0302f"], [0.25, RED], [0.5, "#efefef"], [0.75, "#519ced"], [1, "#1b64be"]]
COURIER_COLOURS = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]

st.set_page_config(page_title="UK Courier Review Analysis", page_icon="📦", layout="wide")


@st.cache_data
def load(folder: str) -> dict[str, pd.DataFrame]:
    p = Path(folder) / "processed"
    out = {}
    for name in ["reviews_scored", "courier_mentions", "aspect_by_courier_long", "courier_summary",
                 "topics_negative", "topics_positive", "model_scores", "monthly_trend"]:
        f = p / f"{name}.csv"
        if f.exists():
            out[name] = pd.read_csv(f, parse_dates=["date"] if name in {"reviews_scored", "courier_mentions"} else None)
    return out


folders = sorted([str(d.relative_to(ROOT)) for d in (ROOT / "reports").glob("*") if (d / "processed").exists()])
if not folders:
    st.error("No results yet. Run the pipeline first, e.g.\n\n"
             "`python -m shipment_analysis.pipeline --input data/sample/demo_reviews_synthetic.csv --out reports/demo`")
    st.stop()

with st.sidebar:
    st.header("📦 Courier reviews")
    default = folders.index("reports/full") if "reports/full" in folders else 0
    folder = st.selectbox("Results folder", folders, index=default)
    data = load(str(ROOT / folder))
    reviews, mentions = data["reviews_scored"], data["courier_mentions"]
    label_col = "tf_label" if "tf_label" in reviews else "vaderx_label"
    score_col = "tf_score" if "tf_score" in reviews else "vaderx_compound"

    couriers_all = data["courier_summary"]["courier"].tolist()
    picked = st.multiselect("Couriers", couriers_all, default=couriers_all)
    if mentions["date"].notna().any():
        dmin, dmax = mentions["date"].min().date(), mentions["date"].max().date()
        date_range = st.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
    else:
        date_range = None
    st.caption("Model: " + ("RoBERTa transformer" if label_col == "tf_label" else "VADER + courier lexicon"))

if "demo" in folder:
    st.warning("You're looking at the **synthetic demo data**. These numbers are made up and say nothing "
               "about real couriers. Run the pipeline on the Kaggle data for real results.", icon="⚠️")

# Stable colour per courier, so filtering never repaints the survivors
colour_map = {c: COURIER_COLOURS[i % len(COURIER_COLOURS)] for i, c in enumerate(couriers_all)}

m = mentions[mentions["courier"].isin(picked)]
if date_range and len(date_range) == 2:
    m = m[(m["date"].isna()) | m["date"].between(pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]))]
ids = set(m["review_id"])
r = reviews[reviews["review_id"].isin(ids)]

st.title("What UK shoppers say about their couriers")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Reviews mentioning a courier", f"{len(r):,}")
k2.metric("Negative", f"{(r[label_col] == 'negative').mean():.0%}")
k3.metric("Positive", f"{(r[label_col] == 'positive').mean():.0%}")
if r["rating"].notna().any():
    k4.metric("Average rating", f"{r['rating'].mean():.2f} ★")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Couriers", "Aspects", "Themes", "Models", "Read the reviews"])

with tab1:
    g = m.groupby("courier")
    summ = pd.DataFrame({
        "reviews": g.size(),
        "net": g[label_col].apply(lambda s: 100 * ((s == "positive").mean() - (s == "negative").mean())),
        "avg_rating": g["rating"].mean(),
    }).reset_index().sort_values("net")
    fig = go.Figure(go.Bar(
        x=summ["net"], y=summ["courier"], orientation="h",
        marker_color=[BLUE if v >= 0 else RED for v in summ["net"]],
        customdata=summ[["reviews", "avg_rating"]],
        hovertemplate="<b>%{y}</b><br>Net sentiment %{x:+.0f}<br>%{customdata[0]:,} reviews"
                      "<br>Avg rating %{customdata[1]:.2f}★<extra></extra>",
        text=[f"{v:+.0f}" for v in summ["net"]], textposition="outside"))
    fig.update_layout(title="Net sentiment (% positive − % negative)", height=90 + 45 * len(summ),
                      xaxis=dict(zeroline=True, zerolinecolor="#666"), margin=dict(l=10, r=40, t=50, b=10))
    st.plotly_chart(fig, width="stretch")

    if "monthly_trend" in data:
        t = data["monthly_trend"].copy()
        t["month"] = pd.to_datetime(t["month"])
        t = t[t["courier"].isin(picked)]
        fig = px.line(t, x="month", y="net_sentiment", color="courier", markers=True,
                      color_discrete_map=colour_map, hover_data={"n": True},
                      labels={"net_sentiment": "Net sentiment", "month": ""}, title="Net sentiment by month")
        fig.add_hline(y=0, line_color="#666", line_width=1)
        st.plotly_chart(fig, width="stretch")

with tab2:
    if "aspect_by_courier_long" in data:
        a = data["aspect_by_courier_long"]
        a = a[a["review_id"].isin(ids) & a["courier"].isin(picked)]
        min_n = st.slider("Hide cells with fewer mentions than", 1, 50, 10)
        piv = a.pivot_table(index="courier", columns="aspect", values="aspect_score", aggfunc="mean")
        cnt = a.pivot_table(index="courier", columns="aspect", values="aspect_score", aggfunc="count")
        piv = piv.where(cnt >= min_n)
        fig = px.imshow(piv, color_continuous_scale=DIVERGING, zmin=-1, zmax=1, text_auto="+.2f", aspect="auto",
                        title="Mean sentence sentiment per aspect (red = negative, blue = positive)")
        fig.update_traces(hovertemplate="<b>%{y}</b> · %{x}<br>Mean sentiment %{z:+.2f}<extra></extra>")
        st.plotly_chart(fig, width="stretch")

        neg_ids = set(r.loc[r[label_col] == "negative", "review_id"])
        share = (a[a["review_id"].isin(neg_ids)].groupby("aspect")["review_id"].nunique() / max(1, len(neg_ids)))
        share = share.sort_values().reset_index(name="share")
        fig = px.bar(share, x="share", y="aspect", orientation="h", title="Share of negative reviews mentioning…",
                     color_discrete_sequence=[BLUE], text_auto=".0%")
        fig.update_layout(xaxis_tickformat=".0%", yaxis_title="", xaxis_title="")
        st.plotly_chart(fig, width="stretch")

        aspect = st.selectbox("Show example sentences for", sorted(a["aspect"].unique()))
        st.dataframe(pd.read_csv(Path(ROOT / folder / "processed" / "aspect_sentences.csv"))
                     .query("aspect == @aspect and review_id in @ids")
                     .sort_values("aspect_score")[["sentence", "aspect_score"]].head(50),
                     width="stretch", hide_index=True)

with tab3:
    c1, c2 = st.columns(2)
    for col, key, title in [(c1, "topics_negative", "Complaint themes"), (c2, "topics_positive", "Praise themes")]:
        with col:
            st.subheader(title)
            if key in data and not data[key].empty:
                t = data[key].copy()
                t["share"] = (t["share"] * 100).round(1)
                st.dataframe(t[["share", "top_words"]].rename(columns={"share": "% of reviews"}),
                             width="stretch", hide_index=True)
            else:
                st.info("Not enough reviews to model themes.")
    st.caption("Topics are found on the full dataset (not re-fitted when you filter).")

with tab4:
    if "model_scores" in data:
        st.dataframe(data["model_scores"].style.format({"accuracy": "{:.0%}", "macro_f1": "{:.2f}",
                                                        "negative_f1": "{:.2f}", "n": "{:,}"}),
                     width="stretch", hide_index=True)
        st.caption("Agreement with the reviewer's own star rating (1–2 negative, 3 neutral, 4–5 positive).")
        rated = r.dropna(subset=["rating"]).assign(stars=lambda d: d["rating"].round().astype(int).astype(str) + "★")
        share = (rated.groupby("stars")[label_col].value_counts(normalize=True).rename("share").reset_index())
        fig = px.bar(share, x="share", y="stars", color=label_col, orientation="h", color_discrete_map=SENT,
                     category_orders={label_col: ["negative", "neutral", "positive"]},
                     title="Model label by star rating", text_auto=".0%")
        fig.update_layout(xaxis_tickformat=".0%", legend_title="", yaxis_title="", xaxis_title="")
        st.plotly_chart(fig, width="stretch")

with tab5:
    q = st.text_input("Search review text")
    sent = st.multiselect("Sentiment", ["negative", "neutral", "positive"], default=["negative"])
    view = r[r[label_col].isin(sent)]
    if q:
        view = view[view["text_model"].str.contains(q, case=False, na=False)]
    st.caption(f"{len(view):,} reviews")
    st.dataframe(view[["date", "couriers", "rating", label_col, score_col, "text_model"]]
                 .rename(columns={"text_model": "review", label_col: "sentiment", score_col: "score"})
                 .sort_values("score").head(500), width="stretch", hide_index=True)
