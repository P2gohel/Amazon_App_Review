import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from utils.data_loader import load_sentiment_predictions, load_sentiment_model_results
from utils.ui import apply_style, hero, section, sidebar_brand

apply_style()

# ── Paths ──────────────────────────────────────────────────────────────
IMG_DIR = Path(__file__).resolve().parent.parent.parent / "Images"


def show_image(filename: str, caption: str | None = None):
    path = IMG_DIR / filename
    if path.exists():
        st.image(path.read_bytes(), caption=caption, use_column_width=True)
    else:
        st.info(f"(Image not available: {filename})")


# ── Data ───────────────────────────────────────────────────────────────
model_results = load_sentiment_model_results()
preds = load_sentiment_predictions()

# Resolve columns defensively
name_col = model_results.columns[0]
pred_col = next((c for c in ("predicted_label", "prediction", "y_pred") if c in preds.columns), None)
truth_col = next((c for c in ("sentiment", "true_label", "label", "y_true") if c in preds.columns), None)
proba_col = next((c for c in ("predicted_proba_positive", "proba_positive", "score_positive") if c in preds.columns), None)

# Normalise labels to lowercase strings for easy filtering / plotting
def _norm_label(v):
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    if s in ("1", "pos", "positive", "true"):
        return "positive"
    if s in ("0", "neg", "negative", "false"):
        return "negative"
    return s

if pred_col:
    preds["_pred_norm"] = preds[pred_col].map(_norm_label)
if truth_col:
    preds["_truth_norm"] = preds[truth_col].map(_norm_label)

# ── Sidebar ────────────────────────────────────────────────────────────
total_all = len(preds)
pos_all = int((preds.get("_pred_norm") == "positive").sum()) if "_pred_norm" in preds.columns else 0

try:
    acc_vals = model_results.loc[
        model_results[name_col].str.contains("Logistic", case=False, na=False), "Accuracy"
    ].values
    best_acc = float(acc_vals[0]) if len(acc_vals) else float(model_results["Accuracy"].max())
except Exception:
    best_acc = None

sidebar_brand(total_reviews=total_all, best_acc=best_acc)
st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")

score_filter = st.sidebar.multiselect(
    "Star rating", options=[1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5]
) if "score" in preds.columns else [1, 2, 3, 4, 5]

sentiment_filter = st.sidebar.multiselect(
    "Predicted sentiment",
    options=["positive", "negative"],
    default=["positive", "negative"],
) if "_pred_norm" in preds.columns else ["positive", "negative"]

# Apply filters
df_f = preds.copy()
if "score" in df_f.columns:
    df_f = df_f[df_f["score"].isin(score_filter)]
if "_pred_norm" in df_f.columns:
    df_f = df_f[df_f["_pred_norm"].isin(sentiment_filter)]

# ── Header ─────────────────────────────────────────────────────────────
hero(
    title="Sentiment Analysis",
    subtitle="Positive vs Negative classification with Logistic Regression and "
             "Naive Bayes, benchmarked against a VADER baseline.",
    icon="💬",
)

if len(df_f) < total_all:
    st.info(
        f"Filtered view: **{len(df_f):,}** of {total_all:,} predictions "
        f"(ratings {sorted(score_filter)}, sentiment {sorted(sentiment_filter)})"
    )

# ── KPI row ────────────────────────────────────────────────────────────
section("Key metrics", icon="📊")

total = len(df_f)
pos = int((df_f.get("_pred_norm") == "positive").sum()) if "_pred_norm" in df_f.columns else 0
neg = total - pos
avg_score = df_f["score"].mean() if "score" in df_f.columns and total else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Predictions", f"{total:,}")
c2.metric("Positive", f"{pos:,}", delta=f"{pos/total:.1%}" if total else "—")
c3.metric("Negative", f"{neg:,}",
          delta=f"{neg/total:.1%}" if total else "—", delta_color="inverse")
c4.metric("Avg rating", f"{avg_score:.2f} ★" if total else "—")

if total == 0:
    st.warning("No predictions match the current filters.")
    st.stop()

# ── Model performance (interactive) ────────────────────────────────────
section("Model performance comparison", icon="🤖")

metric_cols = [c for c in model_results.columns
               if c.lower() in ("accuracy", "precision", "recall", "f1", "f1-score", "roc_auc", "auc")]

if "Accuracy" in model_results.columns:
    acc_df = model_results[[name_col, "Accuracy"]].dropna()
    acc_vals = acc_df["Accuracy"].astype(float).tolist()
    acc_models = acc_df[name_col].astype(str).tolist()

    vmin, vmax = min(acc_vals), max(acc_vals)
    pad = max(0.02, (vmax - vmin) * 0.5)
    y_min, y_max = max(0.0, vmin - pad), min(1.0, vmax + pad)

    palette = ["#4F8BF9", "#f59e0b", "#22c55e", "#ef4444", "#7C3AED", "#06b6d4"]
    fig_cmp = go.Figure(go.Bar(
        x=acc_models,
        y=acc_vals,
        marker_color=[palette[i % len(palette)] for i in range(len(acc_models))],
        text=[f"{v:.2%}" for v in acc_vals],
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

with st.expander("Raw results table"):
    st.dataframe(
        model_results.style.format({
            c: "{:.4f}" for c in model_results.select_dtypes("number").columns
        }),
        use_container_width=True, hide_index=True,
    )

# ── Confusion matrix (interactive heatmap) ─────────────────────────────
section("Confusion matrix of Logistic Regression", icon="🎯")

if "_pred_norm" in df_f.columns and "_truth_norm" in df_f.columns:
    cm_df = df_f.dropna(subset=["_pred_norm", "_truth_norm"])
    labels = ["negative", "positive"]
    # Plain crosstab + reindex to plain Index (no CategoricalIndex)
    raw = pd.crosstab(cm_df["_truth_norm"], cm_df["_pred_norm"])
    matrix = raw.reindex(index=labels, columns=labels, fill_value=0).astype(int)

    tn, fp = int(matrix.loc["negative", "negative"]), int(matrix.loc["negative", "positive"])
    fn, tp = int(matrix.loc["positive", "negative"]), int(matrix.loc["positive", "positive"])
    total_cm = tn + fp + fn + tp
    accuracy = (tn + tp) / total_cm if total_cm else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    col_cm, col_mt = st.columns([2, 1])
    with col_cm:
        if total_cm == 0:
            st.info("No rows available to build the confusion matrix.")
        else:
            # Rows are shown top-down; list the y labels in reverse so "True positive"
            # sits at the top of the heatmap without needing autorange tricks.
            y_order = ["positive", "negative"]
            z_display = [matrix.loc[y].tolist() for y in y_order]
            x_labels = [f"Pred {l}" for l in labels]
            y_labels = [f"True {l}" for l in y_order]

            fig_cm = go.Figure(data=go.Heatmap(
                z=z_display,
                x=x_labels,
                y=y_labels,
                colorscale="Blues",
                showscale=True,
                hovertemplate="%{y}<br>%{x}<br>count: %{z:,}<extra></extra>",
            ))

            # Per-cell annotations with a contrast-aware color: light cells get
            # dark text, dark cells get white text.
            z_max = max(max(row) for row in z_display) or 1
            for i, y_lbl in enumerate(y_labels):
                for j, x_lbl in enumerate(x_labels):
                    val = z_display[i][j]
                    color = "white" if val > z_max * 0.55 else "#0f172a"
                    fig_cm.add_annotation(
                        x=x_lbl, y=y_lbl,
                        text=f"<b>{val:,}</b><br>({val/total_cm*100:.1f}%)",
                        showarrow=False,
                        font=dict(color=color, size=14),
                    )

            fig_cm.update_layout(
                height=380,
                margin=dict(t=20, b=40, l=20, r=20),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    with col_mt:
        st.metric("Accuracy", f"{accuracy:.2%}")
        st.metric("Precision", f"{precision:.2%}")
        st.metric("Recall", f"{recall:.2%}")
        st.metric("F1", f"{f1:.2%}")

    st.caption(
        f"Confusion matrix is computed **on the filtered predictions** "
        f"(TP={tp:,}, FP={fp:,}, FN={fn:,}, TN={tn:,})."
    )
else:
    st.info("Confusion matrix requires both true-label and predicted-label columns.")
    show_image("sentiment_confusion_matrices.png")

# ── Static extras (word clouds, ROC, etc.) ─────────────────────────────
with st.expander("More analyses (ROC · word clouds · top words · cross-validation)"):
    t1, t2, t3, t4, t5 = st.tabs(
        ["ROC curves", "Cross-validation", "Top words", "Word clouds", "Impact analysis"]
    )
    with t1:
        show_image("sentiment_roc_curves.png")
    with t2:
        show_image("sentiment_cross_validation.png")
    with t3:
        show_image("sentiment_top_words.png")
    with t4:
        show_image("sentiment_wordclouds.png")
    with t5:
        show_image("sentiment_impact_analysis.png")

# ── Sample predictions with search ─────────────────────────────────────
section("Sample predictions", icon="📝")

search = st.text_input("Search review text", placeholder="e.g. refund, amazing, crash")

display_cols = [c for c in ["content", "score", truth_col, "vader_compound",
                            pred_col, proba_col] if c and c in df_f.columns]

table_df = df_f.copy()
if search and "content" in table_df.columns:
    mask = table_df["content"].astype(str).str.lower().str.contains(search.lower(), na=False)
    table_df = table_df[mask]

only_mistakes = st.checkbox(
    "Show only misclassifications (true ≠ predicted)",
    value=False,
) if "_pred_norm" in df_f.columns and "_truth_norm" in df_f.columns else False

if only_mistakes:
    table_df = table_df[table_df["_pred_norm"] != table_df["_truth_norm"]]

st.caption(f"Showing up to 200 of {len(table_df):,} matching rows.")
if display_cols:
    st.dataframe(table_df[display_cols].head(200), use_container_width=True, hide_index=True)
else:
    st.info("No displayable columns in the predictions file.")
