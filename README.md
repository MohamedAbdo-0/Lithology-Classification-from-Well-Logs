# Lithology Classification from Well Logs

Predicting subsurface rock type in real time from well log measurements — an end-to-end
machine learning project built for a Data Science / Data Analyst portfolio, with a focus on
upstream oil & gas applications relevant to companies like ADNOC.

## Problem and relevance

Identifying the rock type (lithology) at a given depth is essential for formation evaluation,
reservoir identification, and drilling decisions. Traditionally this relies on physical core
or cuttings samples analyzed in a lab — a process that can take days and adds cost to
drilling operations. This project trains a model to predict lithology directly from
instantaneous well log measurements (Gamma Ray, Neutron Porosity, Bulk Density, Sonic Transit
Time, Photoelectric Factor), giving an instant, confidence-scored estimate that can support —
not replace — professional geological interpretation.

## Data

- **Sources:** FORCE 2020 (Norwegian North Sea, public competition dataset) and Kansas SEG
  2016 (public academic dataset), combined into a single unified file of 1,204,414 rows
  across 109 wells.
- **Labeling gap discovered and fixed:** the initial merged dataset had lithology labels for
  only 0.3% of rows. FORCE 2020 labels were recovered by re-merging against the official
  competition training data on well + depth (100% match on 1,170,511 rows); Kansas SEG 2016
  labels were already present. A third source (Volve, one well) has no public lithology
  labels and is used only as an unlabeled demonstration well, excluded from training/evaluation.
- **Final labeled dataset:** 1,174,660 rows, 108 wells, 20 lithology classes — heavily
  imbalanced (Shale alone is ~61% of rows; the smallest class, Basement, is ~0.01%).
- **Physical outlier cleaning:** GR, PEF, RHOB, and DT values outside physically plausible
  ranges (e.g. PEF > 15 barns/electron, GR > 300 API) were identified and treated as missing
  rather than dropped, then imputed — these were confirmed tool/sensor artifacts, not real
  geology.

## Method

- **Train/test split by well** (not by row) using `GroupShuffleSplit`, so no well appears in
  both sets — this avoids data leakage between depth-adjacent samples of the same well.
- **Missing values** imputed using training-set medians only (never leaked from test).
- **Class imbalance** handled with `class_weight='balanced'` rather than naive resampling.
- **Model comparison:** Random Forest and XGBoost were both trained; Random Forest performed
  meaningfully better and was selected as the final model.
- **Model size control:** `max_depth` and `min_samples_leaf` were tuned to keep the serialized
  model at a deployable size (~600 MB) without materially hurting performance.

## Results

| Metric | Value |
|---|---|
| Model | Random Forest (100 trees) |
| Classes | 20 lithology types |
| Accuracy | 66% |
| Macro-F1 | 0.33 |

Accuracy alone is misleading here because of class imbalance — macro-F1 (which weighs every
class equally) is the honest metric, and it reflects that the model performs very well on
common classes (Halite, Shale) but poorly on rare, easily-confused classes (Chalk, Dolomite,
Marl). This is documented in detail in `reports/limitations.md`.

**Feature importance:** Gamma Ray (GR) and Photoelectric Factor (PEF) are consistently the
two most important logs, together accounting for roughly half the model's decision weight.

## Known limitations

- **Class imbalance:** rare classes (Bafflestone, Basement, Coal) have very limited support,
  and several classes have zero representation in some held-out test wells simply because
  they occur in few wells overall.
- **Confusion pairs:** Chalk is frequently misclassified as Sandstone (overlapping porosity
  signature); Dolomite is affected by incomplete NPHI/RHOB/DT coverage in part of the
  training data (Kansas wells lack these logs entirely).
- **Not validated on UAE/ADNOC data:** this project uses public North Sea and Kansas
  datasets, not proprietary UAE field data. UAE reservoirs are typically carbonate-dominated
  (limestone/dolomite), while this dataset is siliciclastic/shale-heavy — direct deployment
  on real UAE wells would require retraining or transfer learning on local, confidential data.
- **Prediction confidence is informative, not just cosmetic:** low-confidence predictions
  (below ~40%) correlate strongly with actual errors in testing, and should be flagged for
  manual geological review rather than trusted directly.

Full details: `reports/limitations.md`.

## Repository structure

```
adnoc_lithology_ml/
├── README.md
├── data/
│   └── unified_well_logs_v2.csv          # not committed (see Data section below)
├── models/
│   ├── random_forest_lithology.joblib    # not committed — see note below
│   ├── train_medians.json
│   ├── physical_ranges.json
│   └── features.json
├── src/
│   ├── train.py                          # data prep, cleaning, training, evaluation
│   ├── predict.py                        # command-line batch prediction on new data
│   └── app.py                            # Streamlit web interface
├── notebooks/
│   └── lithology_classification_walkthrough.ipynb   # narrative EDA + modeling walkthrough
└── reports/
    ├── model_comparison.csv
    ├── feature_importance.csv
    ├── limitations.md
    └── figures/
        ├── feature_importance.png
        ├── well_track_true_vs_predicted.png
        └── project_cover_linkedin.png
```

## How to reproduce

```bash
pip install -r requirements.txt

# Train the model (recreates models/*.joblib and reports/*)
python src/train.py --data data/unified_well_logs_v2.csv

# Run batch prediction on new well log data
python src/predict.py data/new_well_data.csv reports/predictions_output.csv

# Launch the interactive web app
streamlit run src/app.py
```

## Tech stack

Python · pandas · scikit-learn · XGBoost · matplotlib · Streamlit · joblib

## Note on data and model files

`unified_well_logs_v2.csv` (~155 MB) and `random_forest_lithology.joblib` (~600 MB) are
**not included in this repository** due to GitHub's file size limits. See "Getting the data
and model" below for how to obtain or regenerate them.

## Disclaimer

This is a portfolio/demonstration project using public datasets. It is a decision-support
tool intended to illustrate a machine learning workflow for lithology classification, and is
not a substitute for professional petrophysical or geological interpretation.
