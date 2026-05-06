import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.ui import apply_style, hero, sidebar_brand, pipeline_chips, section
from utils.data_loader import load_reviews_with_fake_detection, load_sentiment_model_results

st.set_page_config(
    page_title="Amazon Reviews Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_style()

# Loading data upfront since it's used in multiple places (sidebar KPIs, filters, etc.)

df_all = load_reviews_with_fake_detection()
model_results = load_sentiment_model_results()

total_all = len(df_all)
fake_all = int(df_all["is_fake_moderate"].sum()) if "is_fake_moderate" in df_all.columns else 0

acc_vals = model_results.loc[
    model_results.iloc[:, 0].str.contains("Logistic", case=False, na=False), "Accuracy"
].values
best_acc = float(acc_vals[0]) if len(acc_vals) else None

# sidebar
sidebar_brand(total_reviews=total_all, fake_count=fake_all, best_acc=best_acc)
st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")

year_min = int(df_all["year"].min())
year_max = int(df_all["year"].max())
year_range = st.sidebar.slider(
    "Year range",
    min_value=year_min, max_value=year_max,
    value=(year_min, year_max),
    step=1,
)
score_filter = st.sidebar.multiselect(
    "Star rating", options=[1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5]
)

df = df_all[
    (df_all["year"] >= year_range[0]) &
    (df_all["year"] <= year_range[1]) &
    (df_all["score"].isin(score_filter))
]

# Hero Section
hero(
    title="Amazon Reviews Analysis",
    subtitle="End-to-end exploration, fake-review detection and sentiment "
             "classification across ~82k Amazon app reviews.",
    icon="📊",
)

if year_range != (year_min, year_max) or len(score_filter) < 5:
    st.info(
        f"Filtered view: **{len(df):,}** of {total_all:,} reviews "
        f"(years {year_range[0]}–{year_range[1]}, ratings {sorted(score_filter)})"
    )

# KPI tile row
total_reviews = len(df)
fake_count = int(df["is_fake_moderate"].sum()) if "is_fake_moderate" in df.columns else 0
avg_rating = df["score"].mean() if total_reviews else 0
clean_count = total_reviews - fake_count

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Reviews", f"{total_reviews:,}")
k2.metric("Avg Rating", f"{avg_rating:.2f} ★" if total_reviews else "—")
k3.metric(
    "Likely Fake",
    f"{fake_count:,}",
    delta=f"{fake_count/total_reviews:.1%}" if total_reviews else None,
    delta_color="inverse",
)
k4.metric("Best Model Acc.", f"{best_acc:.2%}" if best_acc else "—")

if total_reviews == 0:
    st.warning("No reviews match the current filters.")
    st.stop()

# Pipeline section
section("Analysis pipeline", icon="🔁")
pipeline_chips([
    "Raw reviews",
    "EDA & feature engineering",
    "Fake detection",
    "Sentiment analysis",
    "Dashboard",
])

# Rating distribution 
section("Rating distribution", icon="⭐")

score_counts = df["score"].value_counts().sort_index()
fig_scores = go.Figure(go.Bar(
    x=score_counts.index.astype(str),
    y=score_counts.values,
    marker_color=["#ef4444", "#f97316", "#eab308", "#84cc16", "#22c55e"],
    text=[f"{v:,}" for v in score_counts.values],
    textposition="outside",
    hovertemplate="<b>%{x} ★</b><br>%{y:,} reviews<extra></extra>",
))
fig_scores.update_layout(
    xaxis_title="Star rating",
    yaxis_title="Number of reviews",
    height=380,
    margin=dict(t=20, b=40, l=20, r=20),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_scores, use_container_width=True)

# Data quality snapshot
section("Data quality snapshot", icon="🔍")
st.caption("Share of reviews flagged by quality signals in the current filter.")

def _share(col: str) -> float:
    if col in df.columns and len(df):
        return float(df[col].astype(bool).sum()) / len(df)
    return 0.0

q1, q2, q3, q4, q5 = st.columns(5)
q1.metric("Anonymous users", f"{_share('is_anonymous_user'):.1%}")
q2.metric("Very short (≤3 words)", f"{_share('is_very_short'):.1%}")
q3.metric("Duplicate text", f"{_share('is_duplicate_text'):.1%}")
q4.metric("Contains URL", f"{_share('has_url'):.1%}")
q5.metric("Contains emoji",
          f"{(df['emoji_count'] > 0).mean():.1%}" if "emoji_count" in df.columns else "—")

#  Model comparison
section("Sentiment model comparison", icon="🤖")

if not model_results.empty and "Accuracy" in model_results.columns:
    model_name_col = model_results.columns[0]
    acc_df = model_results[[model_name_col, "Accuracy"]].dropna()
    acc_values = acc_df["Accuracy"].astype(float).tolist()
    acc_models = acc_df[model_name_col].astype(str).tolist()

    vmin, vmax = min(acc_values), max(acc_values)
    pad = max(0.02, (vmax - vmin) * 0.5)
    y_min, y_max = max(0.0, vmin - pad), min(1.0, vmax + pad)

    palette = ["#4F8BF9", "#f59e0b", "#22c55e", "#ef4444"]
    fig_cmp = go.Figure(go.Bar(
        x=acc_models,
        y=acc_values,
        marker_color=[palette[i % len(palette)] for i in range(len(acc_models))],
        text=[f"{v:.2%}" for v in acc_values],
        textposition="outside",
        textfont=dict(size=14, color="white"),
        hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.4f}<extra></extra>",
        width=[0.45] * len(acc_models),
    ))
    fig_cmp.update_layout(
        height=420,
        xaxis=dict(title=""),
        yaxis=dict(range=[y_min, y_max], title="Accuracy", tickformat=".1%"),
        margin=dict(t=40, b=40, l=20, r=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig_cmp, use_container_width=True)
    st.caption(f"Y-axis is zoomed to {y_min:.0%}–{y_max:.0%} so the gap between models is visible.")

with st.expander("Raw model-results table"):
    st.dataframe(
        model_results.style.format({
            c: "{:.4f}" for c in model_results.select_dtypes("number").columns
        }),
        use_container_width=True,
        hide_index=True,
    )

# Feature cards 
section("Explore the other pages", icon="🧭")

st.markdown(
    """
    <div class="feature-grid">
      <div class="feature-card">
        <span class="feature-icon">🕵️</span>
        <div class="feature-title">Fake Review Detection</div>
        <p class="feature-desc">10 heuristic rules combined with an Isolation Forest. Switch thresholds live and drill into flagged reviews.</p>
        <span class="feature-meta">Interactive</span>
      </div>
      <div class="feature-card">
        <span class="feature-icon">💬</span>
        <div class="feature-title">Sentiment Analysis</div>
        <p class="feature-desc">Logistic Regression vs Naive Bayes, with VADER baseline. Live confusion matrix on filtered data.</p>
        <span class="feature-meta">Interactive</span>
      </div>
      <div class="feature-card">
        <span class="feature-icon">⚡</span>
        <div class="feature-title">Live Prediction</div>
        <p class="feature-desc">Score a single review or upload a CSV batch. Download predictions as a ready-to-use results file.</p>
        <span class="feature-meta">Interactive</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


section("What's inside", icon="🧪")

c1, c2 = st.columns(2)
with c1:
    st.markdown(
        """
        **Fake-review detection**
        - 10 rule-based heuristics (short reviews, caps, duplicates, URLs, lexical diversity, VADER–score mismatch, …)
        - Isolation Forest anomaly detection on 21 engineered features
        - Three thresholds: **conservative**, **moderate**, **aggressive**
        """
    )
with c2:
    st.markdown(
        """
        **Sentiment classification**
        - Logistic Regression (class-weighted) on TF-IDF 1–2-grams
        - Multinomial Naive Bayes as a strong baseline
        - VADER compound score as a rule-based reference
        """
    )

# Tech stack information
section("Tech stack", icon="🛠️")
st.markdown(
    """
    <div class="tech-row">
      <span class="tech-badge">Python</span>
      <span class="tech-badge">Streamlit</span>
      <span class="tech-badge">pandas</span>
      <span class="tech-badge">scikit-learn</span>
      <span class="tech-badge">Plotly</span>
      <span class="tech-badge">VADER</span>
      <span class="tech-badge">TF-IDF</span>
      <span class="tech-badge">Isolation Forest</span>
      <span class="tech-badge">Logistic Regression</span>
      <span class="tech-badge">Naive Bayes</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Dataset files used"):
    st.markdown(
        """
        | File | Used In |
        |------|---------|
        | `reviews_with_fake_detection.csv` | EDA, Fake Detection |
        | `sentiment_predictions.csv` | Sentiment Analysis |
        | `sentiment_model_results.csv` | Sentiment Analysis |
        | `reviews_clean_for_sentiment.csv` | Live Prediction |
        | `fake_reviews_detected.csv` | Fake Detection |
        """
    )
