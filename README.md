# OpenPowerlifting — Data Analysis & Wilks Score Prediction

An end-to-end data science project on the [OpenPowerlifting](https://www.openpowerlifting.org/) dataset, covering exploratory data analysis across 9 dimensions and three machine learning models that predict an athlete's Wilks score from pre-competition information. The best model (LightGBM with athlete history features) achieves R² = 54.8%.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Analysis Summary](#analysis-summary)
- [Machine Learning Model](#machine-learning-model)
- [Key Findings](#key-findings)

---

## Project Overview

This project explores competitive powerlifting data to uncover trends in athlete performance, equipment usage, federation behavior, and demographic patterns. Three regression models (Linear Regression, Random Forest, and LightGBM) are trained to predict an athlete's Wilks score — a bodyweight-normalized strength metric — using only information available before a competition begins.

---

## Dataset

The data comes from the [OpenPowerlifting project](https://www.openpowerlifting.org/data), which aggregates results from hundreds of powerlifting federations worldwide.

Two CSV files are required:

| File | Description |
|---|---|
| `openpowerlifting.csv` | One row per competition entry (lifter results) |
| `meets.csv` | One row per meet (date, location, federation) |

**Download the data from:** https://www.openpowerlifting.org/data  
Place both CSV files in the same directory as the scripts before running.

> The CSV files are not included in this repository due to their size.

---

## Project Structure

```
.
├── analysis.py               # EDA script — 9 analyses + shared load_data()
├── ml_model.py               # ML pipeline — LR + Random Forest + LightGBM
├── meets.csv                 # (download separately)
├── openpowerlifting.csv      # (download separately)
├── requirements.txt
└── README.md
```

**Generated outputs** (created after running the scripts):

```
analysis_1_gender.png … analysis_9_correlation_heatmap.png
ml_wilks_distribution.png, ml_evaluation.png, ml_residuals.png
```

**Saved model artifacts** (created after running `ml_model.py`):

```
wilks_rf_model.joblib       — Random Forest
wilks_encoders.joblib       — shared LabelEncoders (RF)
wilks_lr_pipeline.joblib    — LR preprocessing pipeline
wilks_lgb_model.joblib      — LightGBM
wilks_lgb_encoders.joblib   — LGB encoders
```

---

## Setup & Installation

**Python 3.8 or higher is required.**

1. Clone the repository:
   ```bash
   git clone https://github.com/thelearner00/OpenPowerlifting-analysis.git
   cd openPowerlifting-analysis
   ```

2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate         # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the dataset from https://www.openpowerlifting.org/data and place `meets.csv` and `openpowerlifting.csv` in the project root.

---

## How to Run

**Run the EDA (generates 9 chart PNGs):**
```bash
python analysis.py
```

**Run the ML model (requires analysis.py in the same folder):**
```bash
python ml_model.py
```

The ML script imports `load_data()` from `analysis.py`, so both files must be in the same directory.

---

## Analysis Summary

| # | Analysis | Key Question |
|---|---|---|
| 1 | Gender Comparison | How do male and female athletes differ in Total and Wilks? |
| 2 | Equipment Breakdown | How does equipment type (Raw, Wraps, Single/Multi-ply) affect total lifted? |
| 3 | DQ Analysis | Which federations have the highest disqualification rates? |
| 4 | Time Series | How has participation and average performance changed since 1998? |
| 5 | Federation Performance | Which federations have the most participants? |
| 6 | Country Breakdown | Which countries host the most competition entries? |
| 7 | Age Group Performance | How does Wilks score vary across age groups? |
| 8 | Raw vs Equipped Trend | How has the shift from equipped to raw powerlifting evolved over time? |
| 9 | Correlation Heatmap | How do numeric performance features (Total, Wilks, lifts, bodyweight, age) correlate with each other? |

---

## Machine Learning Model

**Goal:** Predict an athlete's Wilks score using only pre-competition information (no lift results).

**Features used:**
- Sex
- Equipment type
- Age group
- Bodyweight (kg)
- Weight class (kg)

**LightGBM additionally uses:** raw Age, competition Year, and six athlete-history features derived from prior competition records — career average Wilks, most recent Wilks, career peak, consistency score (std), rolling 3-meet average, and competition count. No lift data is used; all features are pre-competition information.

**Models trained:**

| Model | MAE | R² |
|---|---|---|
| Linear Regression (baseline) | 94 Wilks pts | 12.3% |
| Random Forest Regressor | 83 Wilks pts | 27.4% |
| LightGBM + Athlete History | **58 Wilks pts** | **54.8%** |

The LightGBM model uses six athlete history features computed from prior competition results (career average Wilks, most recent Wilks, career peak, consistency score, rolling 3-meet average, and competition count) alongside the five demographic features. Predicting Wilks from pre-competition data alone is inherently difficult — the ceiling is around 55–60% R² because individual strength on any given day depends on factors not observable before the meet.

**Three prediction functions are exposed:** `predict_wilks()` (RF), `predict_wilks_lr()` (LR), and `predict_wilks_lgb()` (LightGBM) for making predictions on new athletes given their profile.

---

## Key Findings

- **Equipped lifters post significantly higher totals** than raw lifters, but Wilks scores (which normalize for bodyweight) narrow the gap considerably.
- **Raw powerlifting surpassed equipped** in participation share around 2012–2013 and has continued growing.
- **Open (24–39) athletes achieve the highest Wilks scores** on average, with Junior athletes slightly below and Masters declining gradually with age.
- **The USA dominates participation** by a large margin, followed by Canada and Australia.
- **DQ rates vary substantially by federation**, suggesting different judging standards or athlete populations.

---

## Author

Mohammed Atea  
mahammedatea2@yahoo.com
