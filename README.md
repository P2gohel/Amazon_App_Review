# Improving Insight Quality in Amazon App Reviews through Sentiment Analysis and Anomaly Detection

**Module:** Interdisciplinary Group Project (UFCFWQ-45-M)
**Team:** IGP Team 11
**Submission:** April 2026

---

## Introduction

Online reviews strongly influence customer purchasing decisions, but their reliability is often reduced by fake reviews, repeated comments, low-quality text, and sentiment–rating mismatches. This project develops a data-driven analytical framework for Amazon app reviews that combines **exploratory data analysis (EDA)**, **anomaly detection**, **sentiment classification**, and **interactive visualisation** to improve the quality of insights gained from user-generated review data.

The pipeline processes ~82,000 Amazon app reviews through a four-phase workflow:

```
Raw Reviews  →  EDA & Feature Engineering  →  Fake Review Detection  →  Sentiment Analysis  →  Interactive Dashboard
```

**Key results**

- ~82,069 cleaned reviews analysed across 26 engineered features
- 2,873 reviews (≈3.5%) flagged as suspicious by ensemble voting (10 rules + Isolation Forest)
- Best sentiment model: **Logistic Regression — 92.28% accuracy, F1 = 0.9227, ROC AUC = 0.9678**
- Multi-page Streamlit dashboard with live single-review and batch CSV prediction

---

## Demo Video

A walkthrough of the dashboard and project pipeline is available here:

**[▶ Project Demo Video (UWE Panopto)](https://uwe.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=cd4d341e-e419-4778-ab63-b43b014989b7)**

---
---

## Project Phases

### Phase 1 — Exploratory Data Analysis

_Notebook: [Code/EDA.ipynb](Code/EDA.ipynb)_

Loads and cleans the raw Amazon app review dataset (82,069 reviews, 32 features after processing). Removes duplicates, normalises text, extracts temporal features (year, month, day, hour), and engineers 26 numeric features grouped into text length, punctuation, behavioural, user-level, lexical, rating-derived, and duplicate-detection categories. Produces visualisations of score distribution, word count, thumbs-up engagement, temporal posting patterns, VADER sentiment alignment, and a correlation heatmap.

### Phase 2 — Fake Review / Anomaly Detection

_Notebook: [Code/isolationForest.ipynb](Code/isolationForest.ipynb)_

A hybrid detection system combining **10 rule-based heuristic voters** (targeting short extreme-rated reviews, repeated text, excessive caps, emoji overuse, sentiment–rating mismatches, etc.) with an **Isolation Forest** model (`n_estimators=200`, `contamination=0.10`) operating on 21 numeric features. A contamination sensitivity analysis (5% / 10% / 15%) justifies the 0.10 setting. Ensemble voting across all 11 voters produces the final fake/genuine classification, with PCA visualisation, agreement heatmaps, and per-rule fire-rate diagnostics.

### Phase 3 — Sentiment Analysis

_Notebook: [Code/sentimentAnalysis_2models.ipynb](Code/sentimentAnalysis_2models.ipynb)_

Sentiment labels are derived from star ratings (1–2 → Negative, 4–5 → Positive, 3 → excluded), giving 72,485 labelled samples. Text is vectorised with **TF-IDF** (`max_features=10,000`, `ngram_range=(1,2)`) and split 80/20 stratified. Two models are trained and compared:

| Model                   |   Accuracy | Weighted F1 |    ROC AUC |
| ----------------------- | ---------: | ----------: | ---------: |
| Logistic Regression     | **92.28%** |  **0.9227** | **0.9678** |
| Multinomial Naive Bayes |     90.49% |      0.9027 |     0.9561 |

Validation includes confusion matrices, ROC curves, 5-fold stratified cross-validation (LR mean F1 = 0.9214 ± 0.0016), VADER baseline comparison, top-word analysis, and an impact study comparing filtered vs unfiltered training data.

### Phase 4 — Interactive Dashboard

_Folder: [Dashboard/](Dashboard/)_

Multi-page Streamlit application that unifies the outputs from all earlier phases.

- **Project Overview** — KPI tiles, rating distribution, model comparison
- **Fake Review Detection** — rule diagnostics, IF feature importance, PCA, agreement heatmap, sensitivity analysis
- **Sentiment Analysis** — model comparison, confusion matrices, ROC curves, cross-validation, word clouds
- **Live Prediction** — single-review input or CSV batch upload for real-time sentiment + fake-review prediction

---

## Contribution Matrix

| Member               | Main Role                                                                    | Phase(s) | Key Contributions                                                                                                                                                                                                                                                                                                                                                                              |
| -------------------- | ---------------------------------------------------------------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Puja Gohel**       | Data Cleaning, EDA & EDA Dashboard Page                                      | 1, 4     | Loaded and cleaned the raw dataset; checked missing values and duplicates; prepared cleaned datasets; produced EDA visualisations; developed the EDA page of the dashboard; contributed to literature review and report documentation                                                                                                                                                          |
| **Ajay Bhuj**        | Feature Engineering, Sentiment Modelling & Live Prediction Page              | 1, 3, 4  | Created behavioural and linguistic features; produced EDA visualisations; trained Logistic Regression and Naive Bayes models; generated evaluation outputs; developed the Live Prediction page of the dashboard; contributed to research & development section and demo video                                                                                                                  |
| **Subhashree Thapa** | Isolation Forest Model & Anomaly Dashboard Page                              | 2, 4     | Selected anomaly detection features; implemented rule-based fake review indicators; trained the Isolation Forest model; generated anomaly scores; developed the Anomaly Detection dashboard page; contributed to literature review and final report checking                                                                                                                                   |
| **Pranika Shrestha** | Coordination Lead, Anomaly Evaluation & Sentiment Dashboard Page             | 2, 3, 4  | Coordinated overall team progress; performed contamination sensitivity analysis; conducted ensemble voting and anomaly validation; created PCA visualisations and correlation heatmaps; exported cleaned sentiment datasets; developed the Sentiment Analysis dashboard page; contributed to literature review, evaluation, reflection, introduction, conclusion, and final report preparation |
| **Aharan Elankovan** | Sentiment Data Preparation, Problem–Solution Section & Documentation Support | 3        | Prepared sentiment data; created sentiment labels; performed train–test split; applied TF-IDF vectorisation; completed problem and solution section; supported documentation; contributed to literature review and final report checking                                                                                                                                                       |
| **All Members**      | Literature Review, Dashboard & Final Portfolio                               | All      | Contributed to literature review; participated in dashboard development; supported evaluation discussion; performed report checking; managed GitLab organisation; assisted in final portfolio preparation                                                                                                                                                                                      |

---

## Repository Structure

```
Final/
├── Code/            # Jupyter notebooks for the analytical pipeline
├── Dashboard/       # Streamlit web application
├── Data/            # Raw and processed CSV datasets
├── Documents/       # Project report, literature review, meeting minutes
├── Images/          # Figures and plots used in the report and dashboard
└── README.md        # This file
```

### `Code/`

Jupyter notebooks implementing the three analytical phases.

| File                              | Description                                                                                                |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `EDA.ipynb`                       | Phase 1 — data loading, cleaning, feature engineering (26 features), and exploratory visualisations        |
| `isolationForest.ipynb`           | Phase 2 — rule-based detectors, Isolation Forest training, contamination sensitivity, ensemble voting      |
| `sentimentAnalysis_2models.ipynb` | Phase 3 — TF-IDF vectorisation, Logistic Regression and Naive Bayes training, evaluation, cross-validation |

### `Dashboard/`

Streamlit application bringing together the outputs of all phases.

| Path                             | Description                                                     |
| -------------------------------- | --------------------------------------------------------------- |
| `app.py`                         | Entry point and Project Overview landing page                   |
| `pages/Fake_Review_Detection.py` | Phase 2 results, diagnostics, and visualisations                |
| `pages/Sentiment_Analysis.py`    | Phase 3 model comparison, evaluation metrics, and word analysis |
| `pages/Live_Prediction.py`       | Single-review and CSV batch live prediction                     |
| `utils/data_loader.py`           | Cached data loaders for the CSVs in `Data/`                     |
| `utils/ui.py`                    | Shared UI components and styling helpers                        |
| `assets/style.css`               | Global stylesheet for consistent dashboard styling              |
| `sample_batch.csv`               | Sample CSV demonstrating the expected batch-prediction format   |
| `requirements.txt`               | Python dependencies required to run the dashboard               |

### `Data/`

Only `amazon_reviews.csv` is the **original raw dataset**. All other CSV files in this folder are **generated automatically** by running the notebooks in [`Code/`](Code/) in order — each notebook reads the outputs of the previous one and produces the next set of files.

| File                              | Source                                | Description                                                                 |
| --------------------------------- | ------------------------------------- | --------------------------------------------------------------------------- |
| `amazon_reviews.csv`              | **Raw input (provided)**              | Original Amazon app reviews dataset — ~84k records, the only file needed to start the pipeline |
| `reviews_clean_all.csv`           | Generated by `EDA.ipynb`              | Cleaned reviews with all 26 engineered features (Phase 1 output)            |
| `reviews_clean_for_sentiment.csv` | Generated by `EDA.ipynb`              | Cleaned reviews filtered for sentiment modelling (3-star reviews excluded)  |
| `reviews_with_fake_detection.csv` | Generated by `isolationForest.ipynb`  | Cleaned reviews with rule votes, Isolation Forest scores, and ensemble flag |
| `fake_reviews_detected.csv`       | Generated by `isolationForest.ipynb`  | Subset of reviews flagged as suspicious by the ensemble voter               |
| `lr_sentiment_predictions.csv`    | Generated by `sentimentAnalysis_2models.ipynb` | Per-review sentiment predictions from the Logistic Regression model |
| `sentiment_predictions.csv`       | Generated by `sentimentAnalysis_2models.ipynb` | Combined per-review sentiment predictions across models             |
| `sentiment_model_results.csv`     | Generated by `sentimentAnalysis_2models.ipynb` | Summary of model accuracy, F1, ROC AUC, and cross-validation metrics |

> **Pipeline order:** `amazon_reviews.csv` → run `EDA.ipynb` → run `isolationForest.ipynb` → run `sentimentAnalysis_2models.ipynb`. Each step writes its output CSVs into this folder for the next stage (and for the dashboard) to consume.

### `Documents/`

Project write-ups and supporting documentation.

| File / Folder                                                                                                | Description                                                                                                                                                                         |
| ------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Report.pdf](Documents/Report.pdf)                                                                           | Complete project report covering introduction, literature review, methodology, results, evaluation, reflection, and conclusion                                                      |
| [Output_and_Evaluation.pdf](Documents/Output_and_Evaluation.pdf)                                             | Consolidated output and evaluation document presenting the results, visualisations, and critical evaluation of EDA, anomaly detection, and sentiment analysis stages                |
| [Research_and_Development.pdf](Documents/Research%20and%20Development/Research_and_Development.pdf) | Detailed write-up of the three-phase analytical pipeline (EDA, anomaly detection, sentiment analysis) and dashboard development (located in the `Research and Development/` subfolder) |
| [annotated_literature_review.md](Documents/annotated_literature_review.md)                                   | Annotated review of academic literature on sentiment analysis and fake review detection, including summaries, critical evaluation, and relevance to the project                     |
| [problems_encountered_and_solutions_implemented.md](Documents/problems_encountered_and_solutions_implemented.md) | Record of technical and team challenges faced during the project (e.g. data cleaning issues, model evaluation, dashboard integration) and the solutions implemented to resolve them |
| [roles_and_responsibilities.md](Documents/roles_and_responsibilities.md)                                     | Breakdown of each team member's main role, phase ownership, and individual responsibilities                                                                                         |
| [sprint_retrospectives.md](Documents/sprint_retrospectives.md)                                               | Reflections at the end of each sprint covering what went well, what didn't, and improvements carried forward                                                                        |
| [Meeting Minutes/](Documents/Meeting%20Minutes/)                                                             | Folder containing weekly meeting notes (`Meeting minute 1` … `Meeting minute 14`) recording discussions, decisions, action items, and progress updates throughout the project       |

### `Images/`

Figures referenced in the report and embedded in the dashboard, including:

- EDA: `score_distribution.png`, `word_count_distribution.png`, `thumbsup_distribution.png`, `temporal_patterns.png`, `correlation_heatmap.png`, `outlier_analysis.png`, `class_imbalance_analysis.png`, `duplicate_analysis.png`, `vader_analysis.png`, `app_version_frequency.png`, `word_count_vs_score.png`
- Anomaly detection: `rule_fire_rates.png`, `feature_importance.png`, `contamination_sensitivity.png`, `vote_distribution.png`, `method_agreement_heatmap.png`, `pca_fake_vs_clean.png`, `fake_vs_real_comparison.png`
- Sentiment: `sentiment_distribution.png`, `sentiment_model_comparison.png`, `sentiment_confusion_matrices.png`, `sentiment_roc_curves.png`, `sentiment_cross_validation.png`, `sentiment_top_words.png`, `sentiment_wordclouds.png`, `sentiment_vader_vs_model.png`, `sentiment_impact_analysis.png`

---

## Getting Started

```bash
# 1. Clone the repository
git clone <gitlab-repo-url>
cd Final

# 2. Install dependencies
pip install -r Dashboard/requirements.txt

# 3. Launch the dashboard
streamlit run Dashboard/app.py
```

The app opens at `http://localhost:8501`.

To re-run the pipeline end-to-end, execute the notebooks in `Code/` in this order:

1. `EDA.ipynb`
2. `isolationForest.ipynb`
3. `sentimentAnalysis_2models.ipynb`

---

## Tech Stack

- **Language:** Python 3.10+
- **ML / Analysis:** scikit-learn, pandas, numpy, vaderSentiment
- **Visualisation:** Plotly, Matplotlib, Seaborn, WordCloud
- **Application:** Streamlit
- **Project Management:** Trello (sprint board), Gantt chart, GitLab (version control), Google Drive (document sharing)

---

## Project Management

The project followed an **iterative Agile methodology with bi-weekly sprints**. Tasks were tracked on a Trello board organised by phase, with a Gantt chart used for high-level timeline planning. Weekly meetings were held to review progress, discuss blockers, and assign upcoming tasks. Meeting minutes are stored in `Documents/`.
