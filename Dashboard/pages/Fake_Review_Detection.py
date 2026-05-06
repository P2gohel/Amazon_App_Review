import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from utils.data_loader import load_reviews_with_fake_detection, load_fake_reviews_detected
from utils.ui import apply_style, hero, section, sidebar_brand

apply_style()

IMG_DIR = Path(__file__).resolve().parent.parent.parent / "Images"

RULE_LABELS = {
    "rule_1": "R1 · Very short review",
    "rule_2": "R2 · Extreme rating + short",
    "rule_3": "R3 · Excessive caps",
    "rule_4": "R4 · Duplicate text",
    "rule_5": "R5 · Anonymous user",
    "rule_6": "R6 · Excessive punctuation",
    "rule_7": "R7 · URL present",
    "rule_8": "R8 · Repeated characters",
    "rule_9": "R9 · Low lexical diversity",
    "rule_10": "R10 · VADER-score mismatch",
}


def show_image(filename: str, caption: str | None = None):
    path = IMG_DIR / filename
    if path.exists():
        st.image(path.read_bytes(), caption=caption, use_column_width=True)
    else:
        st.info(f"(Image not available: {filename})")


#  Data
df = load_reviews_with_fake_detection()
total_all = len(df)
fake_mod_all = int(df["is_fake_moderate"].sum()) if "is_fake_moderate" in df.columns else 0

#  Sidebar 
sidebar_brand(total_reviews=total_all, fake_count=fake_mod_all)
st.sidebar.markdown("---")
st.sidebar.markdown("### Detection controls")

threshold = st.sidebar.radio(
    "Detection threshold",
    options=["conservative", "moderate", "aggressive"],
    index=1,
    help="Conservative ≥4 votes · Moderate ≥3 · Aggressive ≥2",
)
threshold_col = f"is_fake_{threshold}"

score_filter = st.sidebar.multiselect(
    "Star rating", options=[1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5]
)

df_f = df[df["score"].isin(score_filter)]

#  Header
hero(
    title="Fake Review Detection",
    subtitle="10 heuristic rules combined with an Isolation Forest anomaly detector. "
             "Switch thresholds in the sidebar to see how detections change.",
    icon="🕵️",
)

# KPI row 
section("Detections at current threshold", icon="⚠️")

total = len(df_f)
fake_c = int(df_f["is_fake_conservative"].sum()) if "is_fake_conservative" in df_f.columns else 0
fake_m = int(df_f["is_fake_moderate"].sum()) if "is_fake_moderate" in df_f.columns else 0
fake_a = int(df_f["is_fake_aggressive"].sum()) if "is_fake_aggressive" in df_f.columns else 0
selected = {"conservative": fake_c, "moderate": fake_m, "aggressive": fake_a}[threshold]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total reviews", f"{total:,}")
c2.metric(
    f"{threshold.title()} (selected)",
    f"{selected:,}",
    delta=f"{selected/total:.1%}" if total else "—",
    delta_color="inverse",
)
c3.metric("Conservative", f"{fake_c:,}",
          delta=f"{fake_c/total:.1%}" if total else "—", delta_color="off")
c4.metric("Aggressive", f"{fake_a:,}",
          delta=f"{fake_a/total:.1%}" if total else "—", delta_color="off")

#  Rule fire rates 
section("Rule fire rates", icon="📏")
st.caption("Share of reviews each heuristic rule fires on, ordered by frequency. Hover for exact counts.")

rule_cols = [c for c in df_f.columns if c.startswith("rule_") and c != "rule_vote_count"]
if rule_cols:
    rates = pd.DataFrame({
        "rule": [RULE_LABELS.get(c, c) for c in rule_cols],
        "fire_count": [int(df_f[c].sum()) for c in rule_cols],
    })
    rates["fire_rate"] = rates["fire_count"] / max(total, 1)
    rates = rates.sort_values("fire_rate", ascending=True)

    fig_rules = go.Figure(go.Bar(
        x=rates["fire_rate"], y=rates["rule"],
        orientation="h",
        marker_color="#4F8BF9",
        text=[f"{r:.1%} · {c:,}" for r, c in zip(rates["fire_rate"], rates["fire_count"])],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>fires on %{x:.2%} of reviews<br>(%{customdata:,} rows)<extra></extra>",
        customdata=rates["fire_count"],
    ))
    fig_rules.update_layout(
        xaxis=dict(tickformat=".0%", title="Fire rate"),
        yaxis=dict(title=""),
        height=420,
        margin=dict(t=20, b=40, l=20, r=60),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_rules, use_container_width=True)
else:
    show_image("rule_fire_rates.png")

#  Static extras (kept for richness)
with st.expander("More analyses (feature importance · PCA · sensitivity)"):
    tab_fi, tab_pca, tab_agree, tab_sens, tab_other = st.tabs(
        ["Feature Importance", "PCA", "Method Agreement", "Contamination Sensitivity", "Duplicate / Outliers"]
    )
    with tab_fi:
        show_image("feature_importance.png")
    with tab_pca:
        show_image("pca_fake_vs_clean.png")
    with tab_agree:
        show_image("method_agreement_heatmap.png")
    with tab_sens:
        show_image("contamination_sensitivity.png")
    with tab_other:
        show_image("duplicate_analysis.png", caption="Duplicate analysis")
        show_image("outlier_analysis.png", caption="Outlier analysis")

# Flagged reviews table 
section("Flagged reviews", icon="🚩")

fake_df = load_fake_reviews_detected()
display_cols = ["content", "score", "rule_vote_count", "iso_forest",
                "is_fake_conservative", "is_fake_moderate", "is_fake_aggressive"]
available = [c for c in display_cols if c in fake_df.columns]

# Filter to the selected threshold when possible
if threshold_col in fake_df.columns:
    sample = fake_df[fake_df[threshold_col] == True]
else:
    sample = fake_df

search = st.text_input("Search review content", placeholder="e.g. amazing, refund, crash")
min_votes = st.slider(
    "Min rule votes",
    min_value=0,
    max_value=int(fake_df["rule_vote_count"].max()) if "rule_vote_count" in fake_df.columns else 10,
    value=0,
)

if search and "content" in sample.columns:
    s = search.lower()
    sample = sample[sample["content"].astype(str).str.lower().str.contains(s, na=False)]

if "rule_vote_count" in sample.columns and min_votes > 0:
    sample = sample[sample["rule_vote_count"] >= min_votes]

st.caption(f"Showing {min(len(sample), 200):,} of {len(sample):,} matching rows.")
st.dataframe(sample[available].head(200), use_container_width=True, hide_index=True)

# Rules reference
with st.expander("Rule reference"):
    st.markdown(
        """
        | # | Rule | Logic |
        |---|------|-------|
        | 1 | Very short review | Word count ≤ 3 |
        | 2 | Extreme rating + short | 1- or 5-star & short text |
        | 3 | All caps heavy | Caps ratio > 0.5 |
        | 4 | Duplicate text | Exact duplicate content |
        | 5 | Anonymous user | Generic / missing username |
        | 6 | Excessive punctuation | Exclamation / question marks |
        | 7 | URL present | Contains hyperlink |
        | 8 | Repeated characters | Excessive char repetition |
        | 9 | Low unique-word ratio | Repetitive vocabulary |
        | 10 | Sentiment mismatch | VADER vs star rating disagree |
        """
    )
