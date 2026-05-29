"""
Fast Streamlit dashboard for Amazon Watches review data.

This version removes machine learning and word cloud code so deployment is faster.
It focuses only on data cleaning, filters, tables, and visual analytics.

Run locally:
    pip install -r requirements.txt
    streamlit run watches_streamlit_viz_only.py

Expected data file:
    Watches.txt.gz

Place Watches.txt.gz in the same folder as this script, or upload it from the sidebar.
"""

from __future__ import annotations

import gzip
import io
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Watches Review Dashboard",
    page_icon="⌚",
    layout="wide",
)


FIELD_MAP = {
    "product/productId": "product_id",
    "product/title": "product_title",
    "product/price": "price",
    "review/userId": "user_id",
    "review/profileName": "profile_name",
    "review/helpfulness": "helpfulness",
    "review/score": "score",
    "review/time": "unix_time",
    "review/summary": "summary",
    "review/text": "review_text",
}


@st.cache_data(show_spinner=False)
def parse_amazon_txt_gz(uploaded_bytes: bytes | None, default_path: str) -> pd.DataFrame:
    """Parse Amazon review txt.gz format into a DataFrame."""
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}

    if uploaded_bytes is not None:
        stream: Iterable[str] = gzip.open(io.BytesIO(uploaded_bytes), mode="rt", encoding="utf-8", errors="ignore")
    else:
        stream = gzip.open(default_path, mode="rt", encoding="utf-8", errors="ignore")

    with stream as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line.strip():
                if current:
                    records.append(current)
                    current = {}
                continue
            if ": " in line:
                key, value = line.split(": ", 1)
                current[FIELD_MAP.get(key, key.replace("/", "_"))] = value

    if current:
        records.append(current)

    return pd.DataFrame(records)


@st.cache_data(show_spinner=False)
def clean_and_engineer(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean raw review data and create dashboard features."""
    df = df_raw.copy()

    text_cols = ["product_title", "summary", "review_text", "profile_name", "user_id", "product_id"]
    for col in text_cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    df["score"] = pd.to_numeric(df.get("score"), errors="coerce")
    df = df.dropna(subset=["score"])
    df["score"] = df["score"].astype(float)

    price = df.get("price", pd.Series(index=df.index, dtype="object")).replace("unknown", np.nan)
    df["price"] = pd.to_numeric(price, errors="coerce")

    df["unix_time"] = pd.to_numeric(df.get("unix_time"), errors="coerce")
    df["review_date"] = pd.to_datetime(df["unix_time"], unit="s", errors="coerce")
    df["review_year"] = df["review_date"].dt.year
    df["review_month"] = df["review_date"].dt.to_period("M").astype(str)

    helpful = df.get("helpfulness", pd.Series("0/0", index=df.index)).fillna("0/0").astype(str)
    helpful_split = helpful.str.extract(r"(?P<helpful_votes>\d+)\s*/\s*(?P<total_votes>\d+)")
    df["helpful_votes"] = pd.to_numeric(helpful_split["helpful_votes"], errors="coerce").fillna(0).astype(int)
    df["total_votes"] = pd.to_numeric(helpful_split["total_votes"], errors="coerce").fillna(0).astype(int)
    df["helpfulness_ratio"] = np.where(df["total_votes"] > 0, df["helpful_votes"] / df["total_votes"], np.nan)

    df["review_length_chars"] = df["review_text"].str.len()
    df["review_length_words"] = df["review_text"].str.split().str.len().fillna(0).astype(int)
    df["summary_length_words"] = df["summary"].str.split().str.len().fillna(0).astype(int)

    df["sentiment"] = np.select(
        [df["score"] >= 4, df["score"] <= 2, df["score"] == 3],
        ["positive", "negative", "neutral"],
        default="unknown",
    )

    df["price_band"] = pd.cut(
        df["price"],
        bins=[0, 25, 50, 100, 200, 500, np.inf],
        labels=["$0-25", "$25-50", "$50-100", "$100-200", "$200-500", "$500+"],
    )

    return df


@st.cache_data(show_spinner=False)
def get_top_words(texts: pd.Series, n_words: int = 30) -> pd.DataFrame:
    """Simple fast word-frequency table without ML packages."""
    stop_words = {
        "the", "and", "for", "this", "that", "with", "was", "are", "you", "but", "not", "have", "has",
        "watch", "watches", "from", "they", "very", "just", "one", "all", "out", "can", "had", "its",
        "his", "her", "she", "him", "our", "your", "about", "would", "there", "their", "what", "when",
        "were", "been", "will", "more", "than", "like", "time", "good", "great",
    }
    text = " ".join(texts.dropna().astype(str).tolist()).lower()
    words = re.findall(r"[a-z]{3,}", text)
    words = [w for w in words if w not in stop_words]
    if not words:
        return pd.DataFrame(columns=["word", "count"])
    counts = pd.Series(words).value_counts().head(n_words).reset_index()
    counts.columns = ["word", "count"]
    return counts


def safe_filename_default() -> str:
    local = Path(__file__).with_name("Watches.txt.gz")
    return str(local) if local.exists() else "/mnt/data/Watches.txt.gz"


def apply_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    score_range = st.sidebar.slider("Score range", 1.0, 5.0, (1.0, 5.0), step=0.5)
    sentiments = st.sidebar.multiselect(
        "Sentiment classes",
        options=sorted(df["sentiment"].dropna().astype(str).unique()),
        default=sorted(df["sentiment"].dropna().astype(str).unique()),
    )

    years = sorted([int(y) for y in df["review_year"].dropna().unique()])
    if years:
        year_range = st.sidebar.slider("Review year range", min(years), max(years), (min(years), max(years)))
    else:
        year_range = (None, None)

    min_words = st.sidebar.slider(
        "Minimum review length, words",
        0,
        int(max(10, df["review_length_words"].quantile(0.99))),
        0,
    )
    search = st.sidebar.text_input("Search title, summary, or review text", "")

    filtered = df.copy()
    filtered = filtered[(filtered["score"] >= score_range[0]) & (filtered["score"] <= score_range[1])]
    if sentiments:
        filtered = filtered[filtered["sentiment"].astype(str).isin(sentiments)]
    if year_range[0] is not None:
        filtered = filtered[(filtered["review_year"] >= year_range[0]) & (filtered["review_year"] <= year_range[1])]
    filtered = filtered[filtered["review_length_words"] >= min_words]

    if search.strip():
        pattern = re.escape(search.strip())
        filtered = filtered[
            filtered["product_title"].str.contains(pattern, case=False, na=False)
            | filtered["review_text"].str.contains(pattern, case=False, na=False)
            | filtered["summary"].str.contains(pattern, case=False, na=False)
        ]

    return filtered


def main() -> None:
    st.title("⌚ Watches Review Visualization Dashboard")
    st.caption("Fast Streamlit app: no ML training, no transformers, no word cloud. Only cleaning, filters, charts, and tables.")

    uploaded = st.sidebar.file_uploader("Optional: upload Watches.txt.gz", type=["gz"])
    uploaded_bytes = uploaded.read() if uploaded is not None else None

    default_path = safe_filename_default()
    if uploaded_bytes is None and not Path(default_path).exists():
        st.error("Could not find Watches.txt.gz. Upload it in the sidebar or place it beside this script.")
        st.stop()

    with st.spinner("Loading and cleaning data..."):
        df_raw = parse_amazon_txt_gz(uploaded_bytes, default_path)
        df = clean_and_engineer(df_raw)

    filtered = apply_sidebar_filters(df)

    st.subheader("Key metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Reviews", f"{len(filtered):,}")
    c2.metric("Products", f"{filtered['product_id'].nunique():,}")
    c3.metric("Users", f"{filtered['user_id'].nunique():,}")
    c4.metric("Average score", f"{filtered['score'].mean():.2f}" if len(filtered) else "NA")
    c5.metric("Median words", f"{filtered['review_length_words'].median():.0f}" if len(filtered) else "NA")

    if filtered.empty:
        st.warning("No rows match the current filters.")
        st.stop()

    overview_tab, product_tab, time_tab, text_tab, table_tab = st.tabs(
        ["Overview", "Products", "Time trends", "Text summaries", "Data table"]
    )

    with overview_tab:
        st.subheader("Review distribution")
        col1, col2 = st.columns(2)

        score_counts = filtered["score"].value_counts().sort_index().reset_index()
        score_counts.columns = ["score", "count"]
        fig_score = px.bar(score_counts, x="score", y="count", title="Review count by star score")
        col1.plotly_chart(fig_score, use_container_width=True)

        sent_counts = filtered["sentiment"].astype(str).value_counts().reset_index()
        sent_counts.columns = ["sentiment", "count"]
        fig_sent = px.pie(sent_counts, names="sentiment", values="count", title="Sentiment mix from star scores")
        col2.plotly_chart(fig_sent, use_container_width=True)

        col3, col4 = st.columns(2)
        fig_len_hist = px.histogram(
            filtered,
            x="review_length_words",
            nbins=60,
            title="Histogram of review length",
            labels={"review_length_words": "Review length, words"},
        )
        col3.plotly_chart(fig_len_hist, use_container_width=True)

        fig_score_box = px.box(
            filtered,
            x="sentiment",
            y="review_length_words",
            points="outliers",
            title="Review length by sentiment class",
            labels={"review_length_words": "Review length, words"},
        )
        col4.plotly_chart(fig_score_box, use_container_width=True)

        if filtered["price"].notna().sum() > 0:
            fig_price = px.histogram(
                filtered.dropna(subset=["price"]),
                x="price",
                nbins=60,
                title="Histogram of product prices",
                labels={"price": "Price"},
            )
            st.plotly_chart(fig_price, use_container_width=True)

            price_score = filtered.dropna(subset=["price"]).copy()
            price_score = price_score[price_score["price"] <= price_score["price"].quantile(0.99)]
            fig_price_score = px.box(
                price_score,
                x="score",
                y="price",
                points="outliers",
                title="Price distribution by star score",
                labels={"price": "Price", "score": "Star score"},
            )
            st.plotly_chart(fig_price_score, use_container_width=True)

    with product_tab:
        st.subheader("Product-level analysis")
        min_reviews_product = st.slider("Minimum reviews per product", 1, 100, 5)

        product_summary = (
            filtered.groupby(["product_id", "product_title"], as_index=False)
            .agg(
                review_count=("score", "size"),
                average_score=("score", "mean"),
                median_score=("score", "median"),
                average_helpfulness=("helpfulness_ratio", "mean"),
                median_price=("price", "median"),
                median_review_words=("review_length_words", "median"),
            )
        )
        product_summary = product_summary[product_summary["review_count"] >= min_reviews_product]

        top_reviewed = product_summary.sort_values("review_count", ascending=False).head(20)
        fig_active = px.bar(
            top_reviewed.sort_values("review_count"),
            x="review_count",
            y="product_title",
            orientation="h",
            hover_data=["average_score", "median_price"],
            title="Most-reviewed products",
        )
        st.plotly_chart(fig_active, use_container_width=True)

        top_rated = product_summary.sort_values(["average_score", "review_count"], ascending=[False, False]).head(20)
        fig_top = px.bar(
            top_rated.sort_values("average_score"),
            x="average_score",
            y="product_title",
            orientation="h",
            hover_data=["review_count", "median_price"],
            title="Highest-rated products with enough reviews",
        )
        st.plotly_chart(fig_top, use_container_width=True)

        fig_scatter = px.scatter(
            product_summary,
            x="review_count",
            y="average_score",
            size="review_count",
            hover_name="product_title",
            hover_data=["median_price", "average_helpfulness"],
            title="Product popularity versus average rating",
            labels={"review_count": "Number of reviews", "average_score": "Average score"},
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.dataframe(product_summary.sort_values("review_count", ascending=False), use_container_width=True)

    with time_tab:
        st.subheader("Time trends")
        monthly = (
            filtered.dropna(subset=["review_date"])
            .groupby("review_month", as_index=False)
            .agg(review_count=("score", "size"), average_score=("score", "mean"), median_words=("review_length_words", "median"))
            .sort_values("review_month")
        )

        fig_month_count = px.line(monthly, x="review_month", y="review_count", title="Review volume over time")
        st.plotly_chart(fig_month_count, use_container_width=True)

        fig_month_score = px.line(monthly, x="review_month", y="average_score", title="Average star score over time")
        st.plotly_chart(fig_month_score, use_container_width=True)

        yearly_sentiment = (
            filtered.dropna(subset=["review_year"])
            .groupby(["review_year", "sentiment"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        fig_year_sent = px.bar(
            yearly_sentiment,
            x="review_year",
            y="count",
            color="sentiment",
            title="Sentiment counts by year",
            labels={"review_year": "Year", "count": "Reviews"},
        )
        st.plotly_chart(fig_year_sent, use_container_width=True)

    with text_tab:
        st.subheader("Text summaries without ML")
        sample_text = filtered["review_text"].dropna().sample(min(len(filtered), 8000), random_state=42)
        top_words = get_top_words(sample_text, n_words=30)
        if top_words.empty:
            st.info("No text available after filtering.")
        else:
            fig_words = px.bar(
                top_words.sort_values("count"),
                x="count",
                y="word",
                orientation="h",
                title="Most frequent review words",
            )
            st.plotly_chart(fig_words, use_container_width=True)

        col1, col2 = st.columns(2)
        fig_summary_len = px.histogram(
            filtered,
            x="summary_length_words",
            nbins=30,
            title="Summary length distribution",
            labels={"summary_length_words": "Summary length, words"},
        )
        col1.plotly_chart(fig_summary_len, use_container_width=True)

        fig_helpful = px.scatter(
            filtered[filtered["total_votes"] > 0],
            x="review_length_words",
            y="helpfulness_ratio",
            color="score",
            hover_data=["summary"],
            title="Review length versus helpfulness ratio",
            labels={"review_length_words": "Review length, words", "helpfulness_ratio": "Helpfulness ratio"},
        )
        col2.plotly_chart(fig_helpful, use_container_width=True)

        st.subheader("Example reviews")
        example_reviews = filtered.sample(min(10, len(filtered)), random_state=7)[
            ["product_title", "score", "sentiment", "summary", "review_text"]
        ]
        st.dataframe(example_reviews, use_container_width=True)

    with table_tab:
        st.subheader("Cleaned data preview")
        cols = [
            "product_id",
            "product_title",
            "price",
            "score",
            "sentiment",
            "review_date",
            "helpful_votes",
            "total_votes",
            "helpfulness_ratio",
            "review_length_words",
            "summary",
            "review_text",
        ]
        st.dataframe(filtered[cols].head(1000), use_container_width=True)

        csv = filtered[cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download filtered cleaned data as CSV",
            data=csv,
            file_name="filtered_watches_reviews.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
