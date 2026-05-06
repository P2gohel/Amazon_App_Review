import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import IsolationForest
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from utils.data_loader import load_reviews_clean_for_sentiment
from utils.ui import apply_style, hero, sidebar_brand

apply_style()
sidebar_brand()

# Constants 
RANDOM_SEED = 42

# Train & cache models

@st.cache_resource
def train_sentiment_models():
    """Train Logistic Regression and Naive Bayes on the clean dataset."""
    df = load_reviews_clean_for_sentiment()
    df = df.dropna(subset=["content_clean"])
    df["sentiment"] = (df["score"] >= 4).astype(int)

    X_text = df["content_clean"].values
    y = df["sentiment"].values

    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_tfidf = tfidf.fit_transform(X_text)

    lr = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1
    )
    lr.fit(X_tfidf, y)

    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_tfidf, y)

    return tfidf, lr, nb


@st.cache_resource
def train_isolation_forest():
    """Train Isolation Forest on the full dataset features."""
    df = load_reviews_clean_for_sentiment()
    ml_features = [
        "score", "thumbsUpCount", "char_len", "word_count", "avg_word_len",
        "caps_ratio", "exclamation_count", "question_count", "emoji_count",
        "has_url", "is_duplicate_text", "dup_text_weight", "reviews_by_userName",
        "is_anonymous_user", "unique_word_ratio", "stopword_ratio",
        "punctuation_ratio", "vader_compound", "hour", "day_of_week",
        "score_extremity",
    ]
    X = df[ml_features].fillna(0).values

    iso = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=0.10,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    iso.fit(X)
    return iso, ml_features


# Feature engineering for a single review 

def engineer_features(text: str, score: int) -> dict:
    """Compute the same features used in training for a single review."""
    analyzer = SentimentIntensityAnalyzer()
    vader = analyzer.polarity_scores(text)

    words = text.split()
    word_count = len(words)
    char_len = len(text)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    unique_words = set(w.lower() for w in words)
    unique_word_ratio = len(unique_words) / word_count if word_count > 0 else 1.0

    alpha_chars = [c for c in text if c.isalpha()]
    caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) if alpha_chars else 0

    import string
    stopwords_set = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
        "your", "yours", "yourself", "yourselves", "he", "him", "his",
        "himself", "she", "her", "hers", "herself", "it", "its", "itself",
        "they", "them", "their", "theirs", "themselves", "what", "which",
        "who", "whom", "this", "that", "these", "those", "am", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "having",
        "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
        "or", "because", "as", "until", "while", "of", "at", "by", "for",
        "with", "about", "against", "between", "through", "during", "before",
        "after", "above", "below", "to", "from", "up", "down", "in", "out",
        "on", "off", "over", "under", "again", "further", "then", "once",
        "not", "no", "nor", "so", "too", "very", "can", "will", "just",
        "don", "should", "now",
    }
    lower_words = [w.lower().strip(string.punctuation) for w in words]
    stopword_count = sum(1 for w in lower_words if w in stopwords_set)
    stopword_ratio = stopword_count / word_count if word_count > 0 else 0

    punct_count = sum(1 for c in text if c in string.punctuation)
    punctuation_ratio = punct_count / char_len if char_len > 0 else 0

    exclamation_count = text.count("!")
    question_count = text.count("?")
    digit_count = sum(1 for c in text if c.isdigit())
    has_url = 1 if re.search(r"https?://|www\.", text) else 0

    # Emoji detection (simple unicode range check)
    emoji_count = len(re.findall(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
        r"\U00002702-\U000027B0\U0000FE00-\U0000FE0F"
        r"\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F]", text
    ))

    repeated_char_flag = 1 if re.search(r"(.)\1{3,}", text) else 0
    score_extremity = abs(score - 3)

    return {
        "score": score,
        "thumbsUpCount": 0,
        "char_len": char_len,
        "word_count": word_count,
        "avg_word_len": avg_word_len,
        "caps_ratio": caps_ratio,
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "emoji_count": emoji_count,
        "has_url": has_url,
        "is_duplicate_text": 0,
        "dup_text_weight": 0,
        "reviews_by_userName": 1,
        "is_anonymous_user": 0,
        "unique_word_ratio": unique_word_ratio,
        "stopword_ratio": stopword_ratio,
        "punctuation_ratio": punctuation_ratio,
        "vader_compound": vader["compound"],
        "hour": 12,
        "day_of_week": 3,
        "score_extremity": score_extremity,
        # extras for rule checking
        "repeated_char_flag": repeated_char_flag,
        "digit_count": digit_count,
        "is_very_short": 1 if word_count <= 3 else 0,
        "vader_neg": vader["neg"],
        "vader_pos": vader["pos"],
    }


def apply_rules(f: dict) -> dict:
    """Apply the 10 heuristic rules. Returns dict of rule results."""
    rules = {}
    rules["R1: Generic short + extreme"] = int(f["word_count"] <= 3 and f["score"] in (1, 5))
    rules["R2: Extreme + no engagement + short"] = int(
        f["score"] in (1, 5) and f["thumbsUpCount"] == 0 and f["word_count"] < 10
    )
    rules["R3: Excessive caps (>50%)"] = int(f["caps_ratio"] > 0.5 and f["char_len"] > 20)
    rules["R4: Weighted duplicate text"] = 0  # single review can't be duplicate
    rules["R5: Excessive exclamation (>5)"] = int(f["exclamation_count"] > 5)
    rules["R6: Many emojis (>3)"] = int(f["emoji_count"] > 3)
    rules["R7: VADER-score mismatch"] = int(
        (f["vader_compound"] > 0.5 and f["score"] == 1)
        or (f["vader_compound"] < -0.5 and f["score"] == 5)
    )
    rules["R8: Low lexical diversity"] = int(
        f["unique_word_ratio"] < 0.4 and f["word_count"] > 10
    )
    rules["R9: Has URL"] = int(f["has_url"] == 1)
    rules["R10: Short anonymous + extreme"] = 0  # user is not anonymous
    return rules


#  Clean text

def clean_text(text: str) -> str:
    """Basic text cleaning matching the EDA pipeline."""
    text = text.lower().strip()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Single-review prediction helper (reused for batch)

def predict_one(review_text: str, score: int, tfidf, lr_model, nb_model,
                iso_model, iso_features) -> dict:
    """Run sentiment + fake detection on one review and return a flat dict."""
    features = engineer_features(review_text, score)
    cleaned = clean_text(review_text)

    # Sentiment
    X_input = tfidf.transform([cleaned])
    lr_pred = int(lr_model.predict(X_input)[0])
    lr_proba = lr_model.predict_proba(X_input)[0]
    nb_pred = int(nb_model.predict(X_input)[0])
    nb_proba = nb_model.predict_proba(X_input)[0]

    # Fake detection
    rule_results = apply_rules(features)
    rule_votes = sum(rule_results.values())
    iso_input = np.array([[features[f] for f in iso_features]])
    iso_pred = iso_model.predict(iso_input)[0]
    iso_score = float(iso_model.score_samples(iso_input)[0])
    iso_flag = 1 if iso_pred == -1 else 0
    total_votes = rule_votes + iso_flag

    if total_votes >= 3:
        verdict = "Likely Fake"
    elif total_votes >= 2:
        verdict = "Suspicious"
    else:
        verdict = "Genuine"

    return {
        "review": review_text,
        "score": score,
        "LR_sentiment": "Positive" if lr_pred == 1 else "Negative",
        "LR_confidence": float(lr_proba[1]),
        "NB_sentiment": "Positive" if nb_pred == 1 else "Negative",
        "NB_confidence": float(nb_proba[1]),
        "VADER_compound": features["vader_compound"],
        "rule_votes": rule_votes,
        "iso_anomaly": iso_flag,
        "iso_score": iso_score,
        "total_votes": total_votes,
        "verdict": verdict,
    }


# Page UI 

hero(
    title="Live Prediction",
    subtitle="Score one review or a CSV batch for sentiment (LR + NB) and "
             "fake-review likelihood (10 rules + Isolation Forest).",
    icon="⚡",
)

# Load models (cached after first run)
with st.spinner("Loading models (first time only)..."):
    tfidf, lr_model, nb_model = train_sentiment_models()
    iso_model, iso_features = train_isolation_forest()

tab_single, tab_batch = st.tabs(["Single Review", "Batch Upload (CSV)"])

# Tab 1: Single Review 

with tab_single:
    col_input, col_rating = st.columns([3, 1])

    with col_input:
        review_text = st.text_area(
            "Write your review",
            height=150,
            placeholder="e.g. This app is amazing! It works perfectly and I love the interface.",
        )

    with col_rating:
        star_rating = st.selectbox("Star Rating", options=[1, 2, 3, 4, 5], index=4)
        st.markdown("")
        st.markdown("")
        predict_btn = st.button("Predict", type="primary", use_container_width=True)

    # Prediction (inside Single Review tab)
    if predict_btn and review_text.strip():
        features = engineer_features(review_text, star_rating)
        cleaned = clean_text(review_text)

        # --- Sentiment ---
        X_input = tfidf.transform([cleaned])
        lr_pred = lr_model.predict(X_input)[0]
        lr_proba = lr_model.predict_proba(X_input)[0]
        nb_pred = nb_model.predict(X_input)[0]
        nb_proba = nb_model.predict_proba(X_input)[0]

        # --- Fake detection ---
        rule_results = apply_rules(features)
        rule_votes = sum(rule_results.values())

        iso_input = np.array([[features[f] for f in iso_features]])
        iso_pred = iso_model.predict(iso_input)[0]  # -1 = anomaly
        iso_score = iso_model.score_samples(iso_input)[0]
        iso_flag = 1 if iso_pred == -1 else 0

        total_votes = rule_votes + iso_flag
        is_fake_conservative = total_votes >= 4
        is_fake_moderate = total_votes >= 3
        is_fake_aggressive = total_votes >= 2

        st.markdown("---")

        #  Results: Sentiment
        st.subheader("Sentiment Prediction")

        s1, s2, s3 = st.columns(3)

        lr_label = "Positive" if lr_pred == 1 else "Negative"
        nb_label = "Positive" if nb_pred == 1 else "Negative"
        vader_label = "Positive" if features["vader_compound"] >= 0 else "Negative"

        s1.metric("Logistic Regression", lr_label)
        s2.metric("Naive Bayes", nb_label)
        s3.metric("VADER (baseline)", vader_label)

        # Confidence bar
        st.markdown("**Model Confidence**")
        conf_col1, conf_col2 = st.columns(2)
        with conf_col1:
            st.caption("Logistic Regression")
            st.progress(float(lr_proba[1]))
            st.caption(f"Negative {lr_proba[0]:.1%}  |  Positive {lr_proba[1]:.1%}")
        with conf_col2:
            st.caption("Naive Bayes")
            st.progress(float(nb_proba[1]))
            st.caption(f"Negative {nb_proba[0]:.1%}  |  Positive {nb_proba[1]:.1%}")

        st.markdown("---")

        #Results: Fake Detection
        st.subheader("Fake Review Detection")

        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Rule Votes", f"{rule_votes} / 10")
        f2.metric("Isolation Forest", "Anomaly" if iso_flag else "Normal",
                  delta=f"score = {iso_score:.4f}")
        f3.metric("Total Votes", f"{total_votes} / 11")

        if is_fake_moderate:
            f4.metric("Verdict", "Likely Fake", delta="moderate threshold", delta_color="inverse")
        elif is_fake_aggressive:
            f4.metric("Verdict", "Suspicious", delta="aggressive threshold", delta_color="inverse")
        else:
            f4.metric("Verdict", "Looks Genuine", delta="passed all checks")

        # Rule breakdown
        with st.expander("Rule Breakdown", expanded=True):
            rule_df = pd.DataFrame([
                {"Rule": name, "Flagged": "Yes" if fired else "No"}
                for name, fired in rule_results.items()
            ])
            rule_df["Flagged"] = rule_df["Flagged"].apply(
                lambda x: f"🔴 {x}" if x == "Yes" else f"🟢 {x}"
            )
            st.dataframe(rule_df, use_container_width=True, hide_index=True)

        # Threshold summary
        with st.expander("Detection Thresholds"):
            thresh_data = [
                {"Threshold": "Conservative", "Required Votes": ">= 4", "Your Votes": total_votes, "Result": "Fake" if is_fake_conservative else "Clean"},
                {"Threshold": "Moderate", "Required Votes": ">= 3", "Your Votes": total_votes, "Result": "Fake" if is_fake_moderate else "Clean"},
                {"Threshold": "Aggressive", "Required Votes": ">= 2", "Your Votes": total_votes, "Result": "Fake" if is_fake_aggressive else "Clean"},
            ]
            thresh_df = pd.DataFrame(thresh_data)
            thresh_df["Result"] = thresh_df["Result"].apply(
                lambda x: f"🔴 {x}" if x == "Fake" else f"🟢 {x}"
            )
            st.dataframe(thresh_df, use_container_width=True, hide_index=True)
            st.caption(f"Your review received **{total_votes} out of 11** possible votes. "
                       f"A review needs at least 2 votes to be flagged as suspicious.")

        st.markdown("---")

        #  Feature summary
        with st.expander("Extracted Features"):
            feat_display = {
                "Word Count": features["word_count"],
                "Character Length": features["char_len"],
                "Avg Word Length": f"{features['avg_word_len']:.2f}",
                "Caps Ratio": f"{features['caps_ratio']:.3f}",
                "Unique Word Ratio": f"{features['unique_word_ratio']:.3f}",
                "Stopword Ratio": f"{features['stopword_ratio']:.3f}",
                "Punctuation Ratio": f"{features['punctuation_ratio']:.3f}",
                "Exclamation Marks": features["exclamation_count"],
                "Question Marks": features["question_count"],
                "Emoji Count": features["emoji_count"],
                "Has URL": "Yes" if features["has_url"] else "No",
                "VADER Compound": f"{features['vader_compound']:.4f}",
                "Score Extremity": features["score_extremity"],
            }
            feat_df = pd.DataFrame(
                [{"Feature": k, "Value": v} for k, v in feat_display.items()]
            )
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

    elif predict_btn:
        st.warning("Please enter a review before clicking Predict.")


# Tab 2: Batch Upload (CSV)
with tab_batch:
    st.markdown(
        """
        Upload a **CSV** containing multiple reviews. The file must include:
        - a text column (default name: `review`, also accepts `content` or `text`)
        - a rating column (default name: `score`, also accepts `rating` or `stars`, values 1–5)

        Each row is scored for sentiment and fake-review likelihood, then you can download the results.
        """
    )

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    # Allow the user to cap very large files to keep the UI responsive.
    max_rows = st.number_input(
        "Max rows to process (0 = all)", min_value=0, max_value=100000, value=1000, step=100
    )

    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        # Resolve column names flexibly
        cols_lower = {c.lower(): c for c in df_in.columns}
        review_col = next((cols_lower[c] for c in ("review", "content", "text") if c in cols_lower), None)
        score_col = next((cols_lower[c] for c in ("score", "rating", "stars") if c in cols_lower), None)

        if review_col is None or score_col is None:
            st.error(
                f"CSV must contain a review column (review/content/text) and a score column "
                f"(score/rating/stars). Found: {list(df_in.columns)}"
            )
            st.stop()

        st.success(f"Loaded **{len(df_in)}** rows. Using columns: `{review_col}` + `{score_col}`.")
        st.dataframe(df_in.head(5), use_container_width=True)

        if max_rows > 0 and len(df_in) > max_rows:
            df_in = df_in.head(max_rows)
            st.info(f"Processing first {max_rows} rows only.")

        run_batch = st.button("Run Batch Prediction", type="primary")

        if run_batch:
            # Drop rows with empty review or invalid score
            df_in = df_in.dropna(subset=[review_col, score_col]).copy()
            df_in[score_col] = pd.to_numeric(df_in[score_col], errors="coerce")
            df_in = df_in.dropna(subset=[score_col])
            df_in[score_col] = df_in[score_col].astype(int).clip(1, 5)

            results = []
            progress = st.progress(0.0, text="Predicting...")
            total = len(df_in)

            for i, (_, row) in enumerate(df_in.iterrows(), start=1):
                text = str(row[review_col]).strip()
                if not text:
                    continue
                try:
                    res = predict_one(
                        text, int(row[score_col]),
                        tfidf, lr_model, nb_model, iso_model, iso_features,
                    )
                    results.append(res)
                except Exception as e:
                    results.append({
                        "review": text, "score": int(row[score_col]),
                        "verdict": f"error: {e}",
                    })
                if i % 25 == 0 or i == total:
                    progress.progress(i / total, text=f"Predicting... {i}/{total}")

            progress.empty()

            if not results:
                st.warning("No valid rows to process.")
                st.stop()

            out_df = pd.DataFrame(results)
            st.markdown("---")
            st.subheader("Batch Results")

            # Summary metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total", len(out_df))
            m2.metric("Positive (LR)", int((out_df["LR_sentiment"] == "Positive").sum()))
            m3.metric("Likely Fake", int((out_df["verdict"] == "Likely Fake").sum()))
            m4.metric("Suspicious", int((out_df["verdict"] == "Suspicious").sum()))

            # Distribution charts
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Sentiment distribution (Logistic Regression)")
                st.bar_chart(out_df["LR_sentiment"].value_counts())
            with c2:
                st.caption("Fake-review verdict distribution")
                st.bar_chart(out_df["verdict"].value_counts())

            # Full table
            st.dataframe(out_df, use_container_width=True, hide_index=True)

            # Download button
            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download results as CSV",
                data=csv_bytes,
                file_name="batch_predictions.csv",
                mime="text/csv",
            )
    else:
        st.info("Waiting for a CSV file. Expected columns: `review`, `score`.")
