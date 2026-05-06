import pandas as pd
import streamlit as st
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "Data"


@st.cache_data
def load_reviews_with_fake_detection() -> pd.DataFrame:
    """Load the main reviews dataset with fake detection columns."""
    df = pd.read_csv(DATA_DIR / "reviews_with_fake_detection.csv", low_memory=False)
    if "at" in df.columns:
        df["at"] = pd.to_datetime(df["at"], errors="coerce")
    return df


@st.cache_data
def load_sentiment_predictions() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "sentiment_predictions.csv", low_memory=False)
    return df


@st.cache_data
def load_sentiment_model_results() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "sentiment_model_results.csv")


@st.cache_data
def load_fake_reviews_detected() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "fake_reviews_detected.csv", low_memory=False)


@st.cache_data
def load_reviews_clean_for_sentiment() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "reviews_clean_for_sentiment.csv", low_memory=False)
