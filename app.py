"""
Streamlit dashboard for Amazon Watches review data.

Run:
    pip install streamlit pandas numpy plotly scikit-learn wordcloud matplotlib
    streamlit run watches_streamlit_dashboard.py

Expected data file:
    Watches.txt.gz

Put Watches.txt.gz in the same folder as this script, or upload it from the sidebar.
"""

from __future__ import annotations

import gzip
import io
import re
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Watches Review Analytics Dashboard",
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
    """Clean raw review data and create useful dashboard features."""
    df = df_raw.copy()

    for col in ["product_title", "summary", "review_text", "profile_name", "user_id", "product_id"]:
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
    df["sentiment"] = df["sentiment"].astype("category")

    df["combined_text"] = (df["summary"] + " " + df["review_text"]).str.strip()
    return df


@st.cache_data(show_spinner=False)
def top_terms(texts: pd.Series, n_terms: int = 25) -> pd.DataFrame:
    """Return top TF-IDF terms from selected review texts."""
    cleaned = texts.dropna().astype(str)
    cleaned = cleaned[cleaned.str.len() > 10]
    if cleaned.empty:
        return pd.DataFrame(columns=["term", "score"])

    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    matrix = vectorizer.fit_transform(cleaned)
    scores = np.asarray(matrix.mean(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    out = pd.DataFrame({"term": terms, "score": scores}).sort_values("score", ascending=False).head(n_terms)
    return out


@st.cache_resource(show_spinner=False)
def train_sentiment_model(df: pd.DataFrame):
    """Train a simple TF-IDF + logistic regression sentiment classifier."""
    model_df = df[df["sentiment"].isin(["positive", "negative", "neutral"])].copy()
    model_df = model_df[model_df["combined_text"].str.len() > 20]

    if len(model_df) < 100 or model_df["sentiment"].nunique() < 2:
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        model_df["combined_text"],
        model_df["sentiment"].astype(str),
        test_size=0.2,
        random_state=42,
        stratify=model_df["sentiment"].astype(str),
    )

    pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english", max_features=12000, ngram_range=(1, 2), min_df=3)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=pipe.classes_)
    acc = accuracy_score(y_test, pred)
    return pipe, acc, report, cm, list(pipe.classes_)


@st.cache_data(show_spinner=False)
def make_wordcloud(texts: pd.Series) -> plt.Figure | None:
    """Create a word cloud figure."""
    text = " ".join(texts.dropna().astype(str).tolist())
    text = re.sub(r"[^A-Za-z\s]", " ", text).lower()
    if len(text.strip()) < 50:
        return None

    wc = WordCloud(
        width=1200,
        height=500,
        background_color="white",
        stopwords=set(ENGLISH_STOP_WORDS),
        max_words=150,
        collocations=True,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig


def safe_filename_default() -> str:
    local = Path(__file__).with_name("Watches.txt.gz")
    return str(local) if local.exists() else "/mnt/data/Watches.txt.gz"


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Dashboard controls")

    uploaded = st.sidebar.file_uploader("Optional: upload Watches.txt.gz", type=["gz"])
    st.sidebar.caption("If no file is uploaded, the app uses Watches.txt.gz from the app folder.")

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

    min_words = st.sidebar.slider("Minimum review length, words", 0, int(max(10, df["review_length_words"].quantile(0.99))), 0)
    search = st.sidebar.text_input("Search in title or review text", "")

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
    st.title("⌚ Watches Review Analytics Dashboard")
    st.markdown(
        "Interactive dashboard for cleaning, exploring, visualizing, and modeling Amazon watches review data."
    )

    uploaded = st.sidebar.file_uploader("Upload Watches.txt.gz", type=["gz"], key="data_uploader")
    uploaded_bytes = uploaded.read() if uploaded is not None else None

    default_path = safe_filename_default()
    if uploaded_bytes is None and not Path(default_path).exists():
        st.error("Could not find Watches.txt.gz. Upload it in the sidebar or place it beside this script.")
        st.stop()

    with st.spinner("Loading and cleaning data..."):
        df_raw = parse_amazon_txt_gz(uploaded_bytes, default_path)
        df = clean_and_engineer(df_raw)

    filtered = sidebar_filters(df)

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

    tabs = st.tabs([
        "Overview",
        "Products",
        "Text analytics",
        "Machine learning",
        "Data table",
    ])

    with tabs[0]:
        st.subheader("Review distribution")
        col1, col2 = st.columns(2)

        score_counts = filtered["score"].value_counts().sort_index().reset_index()
        score_counts.columns = ["score", "count"]
        fig_score = px.bar(score_counts, x="score", y="count", title="Review count by star score")
        col1.plotly_chart(fig_score, use_container_width=True)

        sent_counts = filtered["sentiment"].astype(str).value_counts().reset_index()
        sent_counts.columns = ["sentiment", "count"]
        fig_sent = px.pie(sent_counts, names="sentiment", values="count", title="Sentiment mix")
        col2.plotly_chart(fig_sent, use_container_width=True)

        monthly = (
            filtered.dropna(subset=["review_date"])
            .groupby("review_month", as_index=False)
            .agg(review_count=("score", "size"), average_score=("score", "mean"))
            .sort_values("review_month")
        )
        fig_trend = px.line(monthly, x="review_month", y="review_count", title="Reviews over time")
        st.plotly_chart(fig_trend, use_container_width=True)

        fig_len = px.histogram(
            filtered,
            x="review_length_words",
            nbins=60,
            title="Distribution of review length",
        )
        st.plotly_chart(fig_len, use_container_width=True)

    with tabs[1]:
        st.subheader("Product-level analysis")
        min_reviews_product = st.slider("Minimum reviews per product", 1, 50, 5)
        product_summary = (
            filtered.groupby(["product_id", "product_title"], as_index=False)
            .agg(
                review_count=("score", "size"),
                average_score=("score", "mean"),
                average_helpfulness=("helpfulness_ratio", "mean"),
                median_price=("price", "median"),
            )
        )
        product_summary = product_summary[product_summary["review_count"] >= min_reviews_product]

        top_products = product_summary.sort_values(["average_score", "review_count"], ascending=[False, False]).head(20)
        fig_top = px.bar(
            top_products.sort_values("average_score"),
            x="average_score",
            y="product_title",
            orientation="h",
            hover_data=["review_count", "median_price"],
            title="Highest-rated products among filtered reviews",
        )
        st.plotly_chart(fig_top, use_container_width=True)

        active_products = product_summary.sort_values("review_count", ascending=False).head(20)
        fig_active = px.bar(
            active_products.sort_values("review_count"),
            x="review_count",
            y="product_title",
            orientation="h",
            hover_data=["average_score", "median_price"],
            title="Most-reviewed products",
        )
        st.plotly_chart(fig_active, use_container_width=True)

        st.dataframe(product_summary.sort_values("review_count", ascending=False), use_container_width=True)

    with tabs[2]:
        st.subheader("Text analytics")
        text_sample = filtered["combined_text"].dropna().sample(min(len(filtered), 8000), random_state=42)

        col1, col2 = st.columns([1, 1])
        with col1:
            terms = top_terms(text_sample, n_terms=25)
            if not terms.empty:
                fig_terms = px.bar(
                    terms.sort_values("score"),
                    x="score",
                    y="term",
                    orientation="h",
                    title="Top TF-IDF review terms",
                )
                st.plotly_chart(fig_terms, use_container_width=True)
        with col2:
            wc_fig = make_wordcloud(text_sample)
            if wc_fig is not None:
                st.pyplot(wc_fig, use_container_width=True)

        st.subheader("Example reviews")
        sample_reviews = filtered.sample(min(10, len(filtered)), random_state=7)[
            ["product_title", "score", "sentiment", "summary", "review_text"]
        ]
        st.dataframe(sample_reviews, use_container_width=True)

    with tabs[3]:
        st.subheader("Baseline sentiment classifier")
        st.write("The model predicts positive, neutral, or negative sentiment from review text using TF-IDF and logistic regression.")

        with st.spinner("Training model on cleaned data..."):
            model_result = train_sentiment_model(df)

        if model_result is None:
            st.warning("Not enough data to train the model.")
        else:
            pipe, acc, report, cm, labels = model_result
            st.metric("Held-out accuracy", f"{acc:.3f}")

            report_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "class"})
            st.dataframe(report_df, use_container_width=True)

            cm_df = pd.DataFrame(cm, index=labels, columns=labels)
            fig_cm = px.imshow(
                cm_df,
                text_auto=True,
                title="Confusion matrix",
                labels=dict(x="Predicted", y="Actual", color="Count"),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            st.subheader("Try your own review")
            user_review = st.text_area(
                "Write a watch review",
                "The watch looks beautiful and the battery life is excellent, but the strap feels cheap.",
                height=120,
            )
            if user_review.strip():
                pred = pipe.predict([user_review])[0]
                probs = pipe.predict_proba([user_review])[0]
                prob_df = pd.DataFrame({"class": pipe.classes_, "probability": probs}).sort_values("probability", ascending=False)
                st.success(f"Predicted sentiment: {pred}")
                st.dataframe(prob_df, use_container_width=True)

    with tabs[4]:
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
