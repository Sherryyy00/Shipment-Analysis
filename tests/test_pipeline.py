import pandas as pd
import pytest

from shipment_analysis.aspects import aspect_mentions, split_sentences
from shipment_analysis.clean import clean_reviews, deep_clean, light_clean
from shipment_analysis.couriers import courier_from_company, find_couriers
from shipment_analysis.io import standardise
from shipment_analysis.pipeline import build_parser, run
from shipment_analysis.sentiment import add_lexicon_sentiment, net_sentiment, stars_to_label, to_label


# ---------- loading ----------
def test_standardise_detects_columns_and_parses_ratings():
    raw = pd.DataFrame({"Review": ["great"], "Stars": ["Rated 4 out of 5 stars"], "Date": ["2025-01-02"]})
    out = standardise(raw)
    assert out.loc[0, "text"] == "great"
    assert out.loc[0, "rating"] == 4
    assert out.loc[0, "date"] == pd.Timestamp("2025-01-02")


def test_standardise_needs_text():
    with pytest.raises(KeyError):
        standardise(pd.DataFrame({"foo": [1]}))


# ---------- cleaning ----------
def test_light_clean_keeps_case_and_punctuation():
    assert light_clean("NEVER   again!!! see www.x.com") == "NEVER again!!! see"


def test_deep_clean_lowercases_and_strips():
    # The old project called .lower() without assigning it, so this never happened
    assert deep_clean("Didn't ARRIVE, 3 days late!") == "didnt arrive days late"


def test_clean_reviews_keeps_raw_text_separate_and_dedupes():
    df = pd.DataFrame({"title": ["", "", ""], "text": ["Box was CRUSHED!", "box was crushed", "ok"]})
    out = clean_reviews(df)
    assert len(out) == 1  # duplicate removed, "ok" too short
    assert out.loc[0, "text_model"] != out.loc[0, "text_clean"]


def test_title_is_prepended_only_when_new():
    df = pd.DataFrame({"title": ["Awful", "Parcel lost"], "text": ["Parcel lost again by them", "Parcel lost again"]})
    out = clean_reviews(df)
    assert out.loc[0, "text_model"].startswith("Awful.")
    assert out.loc[1, "text_model"] == "Parcel lost again"


# ---------- couriers ----------
@pytest.mark.parametrize("text,expected", [
    ("Evri lost it", ["Evri"]),
    ("used to be Hermes", ["Evri"]),
    ("DPD were great, unlike Royal Mail", ["DPD", "Royal Mail"]),
    ("life has ups and downs", []),
    ("sent with UPS", ["UPS"]),
    ("the amazon driver was lovely", ["Amazon Logistics"]),
])
def test_find_couriers(text, expected):
    assert find_couriers(text) == expected


def test_courier_from_company_domain():
    assert courier_from_company("www.dpd.co.uk") == "DPD"
    assert courier_from_company("some-shop.com") is None


# ---------- sentiment ----------
def test_labels():
    assert to_label(0.5) == "positive" and to_label(-0.5) == "negative" and to_label(0.01) == "neutral"
    assert stars_to_label(1) == "negative" and stars_to_label(3) == "neutral" and stars_to_label(5) == "positive"
    assert stars_to_label(float("nan")) is None


def test_courier_lexicon_catches_delivery_complaints():
    df = pd.DataFrame({"text_model": ["My parcel was lost and arrived damaged."], "rating": [1]})
    out = add_lexicon_sentiment(df)
    assert out.loc[0, "vaderx_compound"] < out.loc[0, "vader_compound"]
    assert out.loc[0, "vaderx_label"] == "negative"


def test_net_sentiment():
    assert net_sentiment(pd.Series(["positive", "positive", "negative", "neutral"])) == 25


# ---------- aspects ----------
def test_split_on_but():
    assert split_sentences("Driver was lovely but the box was crushed") == ["Driver was lovely", "the box was crushed"]


def test_aspects_get_separate_scores():
    df = pd.DataFrame({"review_id": [0], "text_model": ["The driver was lovely but the box was crushed."]})
    m = aspect_mentions(df).set_index("aspect")["aspect_score"]
    assert m["Driver behaviour"] > 0
    assert m["Parcel condition"] < 0


# ---------- end to end ----------
def test_pipeline_runs_on_demo_data(tmp_path):
    args = build_parser().parse_args([
        "--input", "data/sample/demo_reviews_synthetic.csv", "--out", str(tmp_path), "--min-cell-n", "5", "-q"])
    res = run(args)
    assert res["n_reviews"] > 500
    assert (tmp_path / "processed" / "courier_summary.csv").exists()
    assert (tmp_path / "figures" / "02_aspect_heatmap.png").exists()
    assert set(res["summary"]["courier"]) >= {"Evri", "DPD", "Royal Mail"}


def test_dashboard_renders(tmp_path):
    """Runs the Streamlit script headlessly against the demo results."""
    pytest.importorskip("streamlit")
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    if not Path("reports/demo/processed/courier_summary.csv").exists():
        run(build_parser().parse_args(["--input", "data/sample/demo_reviews_synthetic.csv",
                                       "--out", "reports/demo", "--min-cell-n", "5", "-q"]))
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    assert len(at.tabs) == 5
