import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import os
from huggingface_hub import hf_hub_download

HF_REPO = "mohamedabdo2060/Lithology-Classification-from-Well-Logs" 

@st.cache_resource
def load_artifacts():
    model_path = hf_hub_download(repo_id=HF_REPO, filename="random_forest_lithology.joblib")
    medians_path = hf_hub_download(repo_id=HF_REPO, filename="train_medians.json")
    ranges_path = hf_hub_download(repo_id=HF_REPO, filename="physical_ranges.json")
    features_path = hf_hub_download(repo_id=HF_REPO, filename="features.json")

    model = joblib.load(model_path)
    with open(medians_path) as f: medians = json.load(f)
    with open(ranges_path) as f: ranges = json.load(f)
    with open(features_path) as f: features = json.load(f)
    return model, medians, ranges, features

model, medians, ranges, features = load_artifacts()
# ============================================================
# CONFIGURATION
# ============================================================


# ============================================================
# PROFESSIONAL UI / CSS
# ============================================================

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: "Inter", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 15% 0%, rgba(78,205,196,.08), transparent 30%),
                radial-gradient(circle at 90% 10%, rgba(82,113,255,.08), transparent 28%),
                #0B1220;
        }

        .block-container {
            max-width: 1500px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3, h4 {
            color: #F8FAFC !important;
            font-weight: 700 !important;
        }

        p, label, .stMarkdown {
            color: #CBD5E1;
        }

        /* Header */
        .hero {
            padding: 28px 32px;
            border: 1px solid #24344D;
            border-radius: 20px;
            background: linear-gradient(135deg, #111D31 0%, #16263D 100%);
            box-shadow: 0 12px 35px rgba(0,0,0,.22);
            margin-bottom: 22px;
        }

        .hero-title {
            font-size: 34px;
            font-weight: 800;
            color: #F8FAFC;
            margin-bottom: 5px;
        }

        .hero-subtitle {
            color: #94A3B8;
            font-size: 15px;
            line-height: 1.6;
        }

        .badge {
            display: inline-block;
            padding: 5px 11px;
            border-radius: 999px;
            background: rgba(78,205,196,.12);
            border: 1px solid rgba(78,205,196,.28);
            color: #5EEAD4;
            font-size: 12px;
            font-weight: 700;
            margin-right: 6px;
        }

        /* KPI cards */
        div[data-testid="stMetric"] {
            background: linear-gradient(145deg, #111C2E, #16243A);
            border: 1px solid #263852;
            border-radius: 15px;
            padding: 17px 16px;
            box-shadow: 0 7px 20px rgba(0,0,0,.14);
        }

        div[data-testid="stMetric"]:hover {
            border-color: #3A8F8A;
            transform: translateY(-2px);
            transition: .2s ease;
        }

        div[data-testid="stMetricValue"] {
            color: #5EEAD4 !important;
            font-weight: 800 !important;
        }

        div[data-testid="stMetricLabel"] {
            color: #94A3B8 !important;
            font-weight: 600;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 7px;
            background: #0F1929;
            padding: 7px;
            border-radius: 13px;
        }

        .stTabs [data-baseweb="tab"] {
            color: #94A3B8;
            background: transparent;
            border-radius: 9px;
            padding: 10px 18px;
            font-weight: 600;
        }

        .stTabs [aria-selected="true"] {
            color: #5EEAD4 !important;
            background: #172A3B !important;
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #4ECDC4, #35B8B0);
            color: #07131E;
            border: none;
            border-radius: 9px;
            font-weight: 800;
            min-height: 42px;
            box-shadow: 0 6px 16px rgba(78,205,196,.18);
        }

        .stButton > button:hover {
            filter: brightness(1.08);
            transform: translateY(-1px);
        }

        /* Containers */
        .panel {
            background: #111C2E;
            border: 1px solid #263852;
            border-radius: 15px;
            padding: 20px;
            margin: 8px 0 16px 0;
        }

        .section-title {
            color: #F8FAFC;
            font-size: 18px;
            font-weight: 750;
            margin-bottom: 4px;
        }

        .section-caption {
            color: #64748B;
            font-size: 12px;
            margin-bottom: 14px;
        }

        .success-box {
            background: rgba(16,185,129,.10);
            border: 1px solid rgba(16,185,129,.25);
            border-left: 4px solid #10B981;
            border-radius: 10px;
            padding: 13px 16px;
            color: #A7F3D0;
        }

        .warning-box {
            background: rgba(245,158,11,.10);
            border: 1px solid rgba(245,158,11,.25);
            border-left: 4px solid #F59E0B;
            border-radius: 10px;
            padding: 13px 16px;
            color: #FDE68A;
        }

        .danger-box {
            background: rgba(239,68,68,.10);
            border: 1px solid rgba(239,68,68,.25);
            border-left: 4px solid #EF4444;
            border-radius: 10px;
            padding: 13px 16px;
            color: #FCA5A5;
        }

        .prediction-card {
            background: linear-gradient(145deg, #12283A, #142E3A);
            border: 1px solid #2A5E61;
            border-radius: 18px;
            padding: 24px;
            text-align: center;
            box-shadow: 0 12px 28px rgba(0,0,0,.18);
        }

        .prediction-label {
            color: #94A3B8;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .08em;
        }

        .prediction-name {
            color: #5EEAD4;
            font-size: 30px;
            font-weight: 800;
            margin: 7px 0;
        }

        .prediction-confidence {
            color: #E2E8F0;
            font-size: 17px;
            font-weight: 700;
        }

        .info-card {
            background: #0F1A2B;
            border: 1px solid #263852;
            border-radius: 13px;
            padding: 15px 17px;
            margin-top: 8px;
        }

        .info-card-title {
            color: #5EEAD4;
            font-weight: 750;
            margin-bottom: 5px;
        }

        [data-testid="stSidebar"] {
            background: #08111E;
            border-right: 1px solid #1F2E43;
        }

        [data-testid="stFileUploader"] {
            background: #101B2D;
            border: 1.5px dashed #34506C;
            border-radius: 13px;
            padding: 10px;
        }

        .footer {
            color: #64748B;
            text-align: center;
            font-size: 12px;
            padding-top: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# GEOLOGICAL REFERENCE
# ============================================================

LITHOLOGY_INFO = {
    "Shale": (
        "Fine-grained sedimentary rock formed from compacted clay and mud.",
        "Typically higher GR and relatively low permeability; commonly acts as seal/source rock."
    ),
    "Sandstone": (
        "Clastic rock composed mainly of sand-sized grains.",
        "Often lower GR with potentially favorable porosity and reservoir properties."
    ),
    "Sandstone/Shale": (
        "Interbedded sandstone and shale.",
        "Intermediate or heterogeneous log response."
    ),
    "Limestone": (
        "Carbonate rock composed mainly of calcium carbonate.",
        "Typically low GR with relatively high PEF; porosity can vary considerably."
    ),
    "Dolomite": (
        "Carbonate rock dominated by dolomite mineral.",
        "Often denser than limestone and may form good reservoir intervals."
    ),
    "Marl": (
        "Calcareous clay containing both carbonate and clay minerals.",
        "Intermediate GR/PEF response between shale and limestone."
    ),
    "Chalk": (
        "Soft, fine-grained porous form of limestone.",
        "Often low GR with relatively high porosity."
    ),
    "Halite": (
        "Rock salt (NaCl), an evaporite mineral.",
        "Very low GR and distinctive density response."
    ),
    "Anhydrite": (
        "Dense calcium sulfate evaporite mineral.",
        "Very low GR and relatively high density."
    ),
    "Coal": (
        "Organic sedimentary rock formed from compressed plant material.",
        "Very low density and commonly slow sonic response."
    ),
    "Tuff": (
        "Volcanic rock formed from compacted volcanic ash.",
        "Log response varies with composition and alteration."
    ),
    "Basement": (
        "Igneous or metamorphic crystalline rock beneath sedimentary units.",
        "Marks the base of the sedimentary section in many well interpretations."
    ),
    "Wackestone": (
        "Mud-supported carbonate texture containing more than 10% grains.",
        "Carbonate depositional texture."
    ),
    "Mudstone": (
        "Fine-grained carbonate mud rock with limited grains.",
        "Usually associated with lower-energy carbonate environments."
    ),
    "Packstone-grainstone": (
        "Grain-supported carbonate rock with variable mud content.",
        "Generally associated with higher-energy carbonate environments."
    ),
    "Bafflestone": (
        "Carbonate rock built by organisms that baffled sediment.",
        "Can represent organic buildup or reef-related facies."
    ),
    "Nonmarine sandstone": (
        "Sandstone deposited in a continental environment.",
        "May represent river, lake, or aeolian deposition."
    ),
    "Nonmarine coarse siltstone": (
        "Coarse silt-grained rock from a continental setting.",
        "Intermediate grain-size nonmarine facies."
    ),
    "Nonmarine fine siltstone": (
        "Fine silt-grained rock from a continental setting.",
        "Intermediate grain-size nonmarine facies."
    ),
    "Marine siltstone/shale": (
        "Fine-grained rock deposited in a marine setting.",
        "Typically associated with marine depositional environments."
    ),
}

FEATURE_NAMES = {
    "GR": "Gamma Ray",
    "PEF": "Photoelectric Factor",
    "NPHI": "Neutron Porosity",
    "RHOB": "Bulk Density",
    "DT": "Sonic Transit Time",
}

FEATURE_UNITS = {
    "GR": "API",
    "NPHI": "v/v",
    "RHOB": "g/cc",
    "DT": "µs/ft",
    "PEF": "barns/e",
}

# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_artifacts():
    model_path = os.path.join(MODEL_DIR, "random_forest_lithology.joblib")
    median_path = os.path.join(MODEL_DIR, "train_medians.json")
    ranges_path = os.path.join(MODEL_DIR, "physical_ranges.json")
    features_path = os.path.join(MODEL_DIR, "features.json")

    required = [model_path, median_path, ranges_path, features_path]
    missing = [p for p in required if not os.path.exists(p)]

    if missing:
        raise FileNotFoundError(
            "Missing model artifacts:\n" + "\n".join(missing)
        )

    model = joblib.load(model_path)

    with open(median_path, encoding="utf-8") as f:
        medians = json.load(f)

    with open(ranges_path, encoding="utf-8") as f:
        ranges = json.load(f)

    with open(features_path, encoding="utf-8") as f:
        features = json.load(f)

    return model, medians, ranges, features


try:
    model, medians, ranges, features = load_artifacts()
    model_loaded = True
except Exception as exc:
    model_loaded = False
    model = medians = ranges = features = None
    st.error(f"Unable to load model artifacts: {exc}")
    st.stop()


def clean_and_impute(df):
    """Apply physical-range validation and training-median imputation."""
    X_new = df[features].copy()

    for log, (lo, hi) in ranges.items():
        invalid = (X_new[log] < lo) | (X_new[log] > hi)
        X_new.loc[invalid, log] = np.nan

    for col in features:
        X_new[col] = X_new[col].fillna(medians[col])

    return X_new


def predict_with_probabilities(df):
    """Return predictions, confidence, top-3 classes and probability matrix."""
    X_new = clean_and_impute(df)

    proba = model.predict_proba(X_new)
    classes = np.asarray(model.classes_)

    top1_idx = np.argmax(proba, axis=1)
    predictions = classes[top1_idx]
    confidences = proba[np.arange(len(proba)), top1_idx]

    top3_text = []
    top3_data = []

    for row in proba:
        idx = np.argsort(row)[::-1][:3]
        valid = [(classes[i], row[i]) for i in idx if row[i] > 0.01]

        top3_text.append(
            " · ".join(f"{name} ({prob * 100:.1f}%)" for name, prob in valid)
        )
        top3_data.append(valid)

    return predictions, confidences, top3_text, proba, classes


def confidence_level(value):
    if value >= 0.75:
        return "High"
    if value >= 0.40:
        return "Moderate"
    return "Low"


def confidence_color_class(value):
    if value >= 0.75:
        return "success-box"
    if value >= 0.40:
        return "warning-box"
    return "danger-box"


def make_bar_chart(series, title, xlabel="", top_n=None):
    """Create a clean horizontal bar chart."""
    data = series.sort_values(ascending=True)
    if top_n is not None:
        data = data.tail(top_n)

    fig, ax = plt.subplots(figsize=(8, max(3.5, len(data) * 0.35)))
    fig.patch.set_facecolor("#111C2E")
    ax.set_facecolor("#111C2E")

    bars = ax.barh(data.index.astype(str), data.values, height=0.62, color="#4ECDC4")

    ax.set_title(title, color="#F8FAFC", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color="#94A3B8", fontsize=9)

    ax.tick_params(axis="both", colors="#CBD5E1", labelsize=9)
    ax.grid(axis="x", alpha=0.10, color="#CBD5E1")

    for spine in ax.spines.values():
        spine.set_visible(False)

    max_value = float(data.max()) if len(data) else 0
    offset = max_value * 0.015 if max_value else 0.1

    for bar, value in zip(bars, data.values):
        ax.text(
            bar.get_width() + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,.0f}",
            va="center",
            color="#CBD5E1",
            fontsize=8,
        )

    fig.tight_layout()
    return fig


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🪨 Lithology AI")
    st.caption("Well Log Intelligence Platform")
    st.markdown("---")

    st.markdown("### Model")
    st.write("**Algorithm:** Random Forest")
    st.write(f"**Trees:** {getattr(model, 'n_estimators', 'N/A')}")
    st.write(f"**Classes:** {len(model.classes_)}")
    st.write("**Task:** Multi-class classification")

    st.markdown("---")
    st.markdown("### Input logs")

    for feature in features:
        st.write(f"• **{FEATURE_NAMES.get(feature, feature)}**")

    st.markdown("---")
    st.markdown("### Feature importance")

    importances = pd.Series(
        model.feature_importances_,
        index=features
    ).sort_values(ascending=True)

    fig_side, ax_side = plt.subplots(figsize=(4.2, 3.2))
    fig_side.patch.set_facecolor("#08111E")
    ax_side.set_facecolor("#08111E")

    labels = [FEATURE_NAMES.get(x, x) for x in importances.index]
    ax_side.barh(labels, importances.values * 100, color="#4ECDC4", height=.58)

    ax_side.tick_params(colors="#94A3B8", labelsize=8)
    ax_side.set_xlabel("Importance (%)", color="#94A3B8", fontsize=8)

    for spine in ax_side.spines.values():
        spine.set_visible(False)

    ax_side.grid(axis="x", alpha=.08)
    fig_side.tight_layout()
    st.pyplot(fig_side, use_container_width=True)
    plt.close(fig_side)

    st.markdown("---")
    st.caption(
        "Decision-support tool. Predictions should be reviewed together with "
        "professional geological interpretation."
    )

# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div>
            <span class="badge">AI / MACHINE LEARNING</span>
            <span class="badge">WELL LOGS</span>
            <span class="badge">GEOLOGY</span>
        </div>
        <div class="hero-title">Lithology Classification from Well Logs</div>
        <div class="hero-subtitle">
            An interactive machine-learning dashboard for predicting subsurface
            lithology from Gamma Ray, Neutron Porosity, Bulk Density,
            Sonic Transit Time and Photoelectric Factor logs.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_batch, tab_manual, tab_performance = st.tabs(
    [
        "📁 Batch Prediction",
        "🎯 Single Reading",
        "📊 Model Performance",
    ]
)

# ============================================================
# TAB 1 — BATCH PREDICTION
# ============================================================

with tab_batch:

    st.markdown('<div class="section-title">Batch well-log classification</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Upload a CSV containing the required well-log features.</div>',
        unsafe_allow_html=True,
    )

    required_text = " · ".join(
        f"`{f}` ({FEATURE_UNITS.get(f, '')})" for f in features
    )
    st.info(f"Required features: {required_text}")

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="The file should contain the five model input features.",
    )

    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file, low_memory=False)
        except Exception as exc:
            st.error(f"Could not read the CSV file: {exc}")
            st.stop()

        missing_cols = [f for f in features if f not in df.columns]

        if missing_cols:
            st.error(
                "The uploaded file is missing required columns: "
                + ", ".join(missing_cols)
            )
            st.stop()

        # Ensure model inputs are numeric.
        for col in features:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # --------------------------------------------------------
        # INPUT DATA QUALITY
        # --------------------------------------------------------

        with st.expander("🔎 Input data quality", expanded=False):
            q1, q2, q3, q4 = st.columns(4)

            q1.metric("Rows", f"{len(df):,}")
            q2.metric("Columns", f"{len(df.columns):,}")
            q3.metric(
                "Missing input cells",
                f"{df[features].isna().sum().sum():,}"
            )
            q4.metric(
                "Duplicate rows",
                f"{df.duplicated().sum():,}"
            )

            quality_df = pd.DataFrame({
                "Feature": features,
                "Missing": [df[f].isna().sum() for f in features],
                "Missing %": [
                    df[f].isna().mean() * 100 for f in features
                ],
                "Min": [df[f].min() for f in features],
                "Median": [df[f].median() for f in features],
                "Max": [df[f].max() for f in features],
            })

            st.dataframe(
                quality_df.style.format({
                    "Missing %": "{:.2f}%",
                    "Min": "{:.3f}",
                    "Median": "{:.3f}",
                    "Max": "{:.3f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        # --------------------------------------------------------
        # PREDICTION
        # --------------------------------------------------------

        with st.spinner("Validating logs and running model inference..."):
            preds, confs, top3, proba, classes = predict_with_probabilities(df)

        df["Predicted Lithology"] = preds
        df["Confidence"] = confs
        df["Confidence Level"] = [confidence_level(x) for x in confs]
        df["Top-3 Predictions"] = top3

        # --------------------------------------------------------
        # SUMMARY KPIs
        # --------------------------------------------------------

        avg_conf = float(df["Confidence"].mean())
        low_conf_pct = float((df["Confidence"] < 0.40).mean() * 100)
        high_conf_pct = float((df["Confidence"] >= 0.75).mean() * 100)
        unique_lithologies = df["Predicted Lithology"].nunique()

        st.markdown("### Prediction overview")

        k1, k2, k3, k4, k5 = st.columns(5)

        k1.metric("Rows classified", f"{len(df):,}")
        k2.metric("Average confidence", f"{avg_conf * 100:.1f}%")
        k3.metric("High-confidence", f"{high_conf_pct:.1f}%")
        k4.metric("Low-confidence", f"{low_conf_pct:.1f}%")
        k5.metric("Lithologies detected", unique_lithologies)

        if low_conf_pct >= 20:
            st.markdown(
                f'<div class="warning-box">⚠️ {low_conf_pct:.1f}% of predictions '
                'have confidence below 40%. These intervals deserve additional '
                'geological review.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="success-box">✓ {100 - low_conf_pct:.1f}% of '
                'predictions have confidence at or above 40%.</div>',
                unsafe_allow_html=True,
            )

        # --------------------------------------------------------
        # DISTRIBUTION + CONFIDENCE
        # --------------------------------------------------------

        st.markdown("### Prediction analytics")

        chart1, chart2 = st.columns(2)

        with chart1:
            counts = df["Predicted Lithology"].value_counts()
            st.pyplot(
                make_bar_chart(
                    counts,
                    "Predicted Lithology Distribution",
                    "Number of rows",
                ),
                use_container_width=True,
            )
            plt.close("all")

        with chart2:
            fig_conf, ax_conf = plt.subplots(figsize=(8, 5.5))
            fig_conf.patch.set_facecolor("#111C2E")
            ax_conf.set_facecolor("#111C2E")

            ax_conf.hist(
                df["Confidence"],
                bins=20,
                color="#4ECDC4",
                edgecolor="#0B1220",
                alpha=.9,
            )

            ax_conf.axvline(.40, linestyle="--", color="#F59E0B", linewidth=1.5)
            ax_conf.axvline(.75, linestyle="--", color="#10B981", linewidth=1.5)

            ax_conf.set_title(
                "Prediction Confidence Distribution",
                color="#F8FAFC",
                fontsize=13,
                fontweight="bold",
            )
            ax_conf.set_xlabel("Confidence", color="#94A3B8")
            ax_conf.set_ylabel("Rows", color="#94A3B8")
            ax_conf.tick_params(colors="#CBD5E1")

            for spine in ax_conf.spines.values():
                spine.set_visible(False)

            ax_conf.grid(alpha=.08)
            fig_conf.tight_layout()
            st.pyplot(fig_conf, use_container_width=True)
            plt.close(fig_conf)

        # --------------------------------------------------------
        # DETAILED RESULTS
        # --------------------------------------------------------

        st.markdown("### Detailed prediction results")

        selected_liths = st.multiselect(
            "Filter by predicted lithology",
            sorted(df["Predicted Lithology"].unique()),
        )

        selected_conf = st.multiselect(
            "Filter by confidence level",
            ["High", "Moderate", "Low"],
            default=[],
        )

        filtered_df = df.copy()

        if selected_liths:
            filtered_df = filtered_df[
                filtered_df["Predicted Lithology"].isin(selected_liths)
            ]

        if selected_conf:
            filtered_df = filtered_df[
                filtered_df["Confidence Level"].isin(selected_conf)
            ]

        display_columns = [
            c for c in [
                "WELL",
                "DEPTH",
                *features,
                "Predicted Lithology",
                "Confidence",
                "Confidence Level",
                "Top-3 Predictions",
            ]
            if c in filtered_df.columns
        ]

        display_df = filtered_df[display_columns].head(2000).copy()

        if "Confidence" in display_df.columns:
            display_df["Confidence"] = display_df["Confidence"].map(
                lambda x: f"{x * 100:.1f}%"
            )

        st.caption(
            f"Showing {len(display_df):,} rows out of {len(filtered_df):,} "
            "matching rows."
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            height=480,
            hide_index=True,
        )

        # --------------------------------------------------------
        # UNCERTAIN INTERVALS
        # --------------------------------------------------------

        st.markdown("### ⚠️ Most uncertain predictions")

        uncertain = df.sort_values("Confidence").head(20)

        uncertain_cols = [
            c for c in [
                "WELL",
                "DEPTH",
                *features,
                "Predicted Lithology",
                "Confidence",
                "Top-3 Predictions",
            ]
            if c in uncertain.columns
        ]

        uncertain_view = uncertain[uncertain_cols].copy()
        uncertain_view["Confidence"] = (
            uncertain_view["Confidence"] * 100
        ).round(1)

        st.dataframe(
            uncertain_view,
            use_container_width=True,
            hide_index=True,
        )

        # --------------------------------------------------------
        # WELL SUMMARY
        # --------------------------------------------------------

        if "WELL" in df.columns:

            st.markdown("### Well-level summary")

            well_summary = (
                df.groupby("WELL")
                .agg(
                    Rows=("Predicted Lithology", "size"),
                    Avg_Confidence=("Confidence", "mean"),
                    Low_Confidence=(
                        "Confidence",
                        lambda x: (x < .40).mean() * 100,
                    ),
                    Lithologies=("Predicted Lithology", "nunique"),
                )
                .sort_values("Avg_Confidence")
            )

            well_summary["Avg_Confidence"] *= 100
            well_summary["Avg_Confidence"] = well_summary["Avg_Confidence"].round(1)
            well_summary["Low_Confidence"] = well_summary["Low_Confidence"].round(1)

            st.dataframe(
                well_summary,
                use_container_width=True,
                hide_index=False,
            )

        # --------------------------------------------------------
        # WELL LOG TRACK VIEW
        # --------------------------------------------------------

        if "WELL" in df.columns and "DEPTH" in df.columns:

            st.markdown("### Well log interpretation view")

            wells = sorted(df["WELL"].dropna().astype(str).unique())

            if wells:
                well_choice = st.selectbox(
                    "Select well",
                    wells,
                    key="well_view",
                )

                well_df = (
                    df[df["WELL"].astype(str) == well_choice]
                    .sort_values("DEPTH")
                    .copy()
                )

                if not well_df.empty:

                    fig_tracks, axes = plt.subplots(
                        1,
                        len(features) + 1,
                        figsize=(17, 9),
                        sharey=True,
                    )

                    fig_tracks.patch.set_facecolor("#111C2E")

                    for ax, log in zip(axes[:-1], features):

                        ax.set_facecolor("#111C2E")
                        ax.plot(
                            well_df[log],
                            well_df["DEPTH"],
                            color="#4ECDC4",
                            linewidth=.75,
                        )

                        ax.set_title(
                            f"{FEATURE_NAMES.get(log, log)}\n"
                            f"({FEATURE_UNITS.get(log, '')})",
                            color="#F8FAFC",
                            fontsize=9,
                            fontweight="bold",
                        )

                        ax.set_xlabel(log, color="#94A3B8", fontsize=8)
                        ax.tick_params(colors="#94A3B8", labelsize=7)
                        ax.grid(alpha=.08)

                        for spine in ax.spines.values():
                            spine.set_color("#263852")

                    # Lithology track
                    unique_liths = list(
                        well_df["Predicted Lithology"].dropna().unique()
                    )

                    cmap = plt.get_cmap("tab20")
                    color_map = {
                        lith: cmap(i % 20)
                        for i, lith in enumerate(unique_liths)
                    }

                    y = well_df["DEPTH"].to_numpy()
                    x = np.arange(len(well_df))

                    axes[-1].scatter(
                        np.zeros(len(well_df)),
                        y,
                        c=[
                            color_map[x]
                            for x in well_df["Predicted Lithology"]
                        ],
                        s=12,
                        marker="s",
                    )

                    axes[-1].set_title(
                        "Predicted\nLithology",
                        color="#F8FAFC",
                        fontsize=9,
                        fontweight="bold",
                    )
                    axes[-1].set_xticks([])
                    axes[-1].tick_params(colors="#94A3B8", labelsize=7)
                    axes[-1].set_xlim(-1, 1)

                    axes[0].set_ylabel("Depth", color="#CBD5E1")
                    axes[0].invert_yaxis()

                    for lith, color in color_map.items():
                        axes[-1].plot(
                            [],
                            [],
                            marker="s",
                            linestyle="",
                            color=color,
                            label=lith,
                        )

                    legend = axes[-1].legend(
                        loc="center left",
                        bbox_to_anchor=(1.05, .5),
                        frameon=False,
                        fontsize=7,
                    )

                    for text in legend.get_texts():
                        text.set_color("#CBD5E1")

                    fig_tracks.tight_layout()
                    st.pyplot(fig_tracks, use_container_width=True)
                    plt.close(fig_tracks)

        # --------------------------------------------------------
        # GEOLOGICAL REFERENCE
        # --------------------------------------------------------

        st.markdown("### Geological reference")

        predicted_classes = sorted(df["Predicted Lithology"].unique())

        for lith in predicted_classes:
            desc, response = LITHOLOGY_INFO.get(
                lith,
                (
                    "No detailed description available for this class.",
                    "",
                ),
            )

            with st.expander(f"🪨 {lith}"):
                st.markdown(f"**Description:** {desc}")
                if response:
                    st.markdown(f"**Typical log response:** {response}")

        # --------------------------------------------------------
        # DOWNLOADS
        # --------------------------------------------------------

        st.markdown("### Export results")

        download_col1, download_col2 = st.columns(2)

        with download_col1:
            csv_bytes = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇️ Download predictions CSV",
                data=csv_bytes,
                file_name="lithology_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with download_col2:
            summary_export = (
                df["Predicted Lithology"]
                .value_counts()
                .rename_axis("Lithology")
                .reset_index(name="Rows")
            )

            summary_bytes = summary_export.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇️ Download lithology summary",
                data=summary_bytes,
                file_name="lithology_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ============================================================
# TAB 2 — SINGLE READING
# ============================================================

with tab_manual:

    st.markdown('<div class="section-title">Single-point lithology prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Enter one set of well-log measurements and inspect the model probability profile.</div>',
        unsafe_allow_html=True,
    )

    with st.container():
        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            gr = st.number_input(
                "GR (API)",
                min_value=0.0,
                max_value=300.0,
                value=75.0,
                step=1.0,
            )

        with c2:
            nphi = st.number_input(
                "NPHI (v/v)",
                min_value=0.0,
                max_value=1.0,
                value=.25,
                step=.01,
            )

        with c3:
            rhob = st.number_input(
                "RHOB (g/cc)",
                min_value=1.0,
                max_value=3.2,
                value=2.40,
                step=.01,
            )

        with c4:
            dt = st.number_input(
                "DT (µs/ft)",
                min_value=40.0,
                max_value=240.0,
                value=100.0,
                step=1.0,
            )

        with c5:
            pef = st.number_input(
                "PEF (barns/e)",
                min_value=0.0,
                max_value=15.0,
                value=3.0,
                step=.1,
            )

    if st.button("🔍 Classify reading", type="primary", use_container_width=True):

        single_df = pd.DataFrame(
            [{
                "GR": gr,
                "NPHI": nphi,
                "RHOB": rhob,
                "DT": dt,
                "PEF": pef,
            }]
        )

        _, confs, _, proba_matrix, classes = predict_with_probabilities(single_df)

        proba = proba_matrix[0]
        top_idx = int(np.argmax(proba))
        pred_name = classes[top_idx]
        pred_conf = float(proba[top_idx])

        st.markdown("### Prediction")

        left, right = st.columns([1.05, 1])

        with left:
            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="prediction-label">Predicted lithology</div>
                    <div class="prediction-name">{pred_name}</div>
                    <div class="prediction-confidence">
                        Confidence: {pred_conf * 100:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            css_class = confidence_color_class(pred_conf)

            if pred_conf >= .75:
                message = "High-confidence model prediction."
            elif pred_conf >= .40:
                message = "Moderate confidence. Consider additional geological context."
            else:
                message = "Low confidence. Manual geological review is recommended."

            st.markdown(
                f'<div class="{css_class}">{"✓" if pred_conf >= .75 else "⚠️"} {message}</div>',
                unsafe_allow_html=True,
            )

        with right:

            desc, response = LITHOLOGY_INFO.get(
                pred_name,
                (
                    "No detailed description available for this class.",
                    "",
                ),
            )

            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-card-title">🪨 Geological interpretation</div>
                    <div>{desc}</div>
                    <br>
                    <div class="info-card-title">Typical log response</div>
                    <div>{response}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### Input measurements")

        input_display = pd.DataFrame({
            "Log": [FEATURE_NAMES.get(f, f) for f in features],
            "Value": [
                gr if "GR" in features else np.nan,
                nphi if "NPHI" in features else np.nan,
                rhob if "RHOB" in features else np.nan,
                dt if "DT" in features else np.nan,
                pef if "PEF" in features else np.nan,
            ],
            "Unit": [FEATURE_UNITS.get(f, "") for f in features],
        })

        st.dataframe(
            input_display,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Top class probabilities")

        top_n = min(8, len(classes))
        top_indices = np.argsort(proba)[-top_n:][::-1]

        probability_df = pd.DataFrame({
            "Lithology": classes[top_indices],
            "Probability (%)": proba[top_indices] * 100,
        })

        probability_df["Probability (%)"] = probability_df[
            "Probability (%)"
        ].round(2)

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True,
        )

        fig_prob, ax_prob = plt.subplots(figsize=(9, 4.8))
        fig_prob.patch.set_facecolor("#111C2E")
        ax_prob.set_facecolor("#111C2E")

        names = probability_df["Lithology"].iloc[::-1]
        values = probability_df["Probability (%)"].iloc[::-1]

        ax_prob.barh(names, values, color="#4ECDC4", height=.58)

        ax_prob.set_title(
            "Model Probability Profile",
            color="#F8FAFC",
            fontsize=13,
            fontweight="bold",
        )
        ax_prob.set_xlabel("Probability (%)", color="#94A3B8")
        ax_prob.tick_params(colors="#CBD5E1")

        for spine in ax_prob.spines.values():
            spine.set_visible(False)

        ax_prob.grid(axis="x", alpha=.08)

        fig_prob.tight_layout()
        st.pyplot(fig_prob, use_container_width=True)
        plt.close(fig_prob)


# ============================================================
# TAB 3 — MODEL PERFORMANCE
# ============================================================

with tab_performance:

    st.markdown('<div class="section-title">Model performance & transparency</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Performance information available for the current trained model.</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "The reported evaluation uses unseen wells rather than randomly selected "
        "rows, which is a more realistic test of generalization to new wells."
    )

    p1, p2, p3, p4 = st.columns(4)

    p1.metric("Overall Accuracy", "66%")
    p2.metric("Macro-F1", "0.33")
    p3.metric("Classes Evaluated", "20")
    p4.metric("Algorithm", "Random Forest")

    st.markdown("### What the metrics mean")

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-title">Accuracy</div>
                Measures the percentage of predictions that were classified correctly.
                It can look relatively strong when one or a few classes dominate the dataset.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric_col2:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-title">Macro-F1</div>
                Gives each lithology class equal weight. It is therefore more sensitive
                to weak performance on rare classes than overall accuracy.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Feature importance")

    importance_pct = (
        pd.Series(model.feature_importances_, index=features)
        .sort_values(ascending=False)
        * 100
    )

    importance_display = pd.DataFrame({
        "Log": [
            FEATURE_NAMES.get(x, x)
            for x in importance_pct.index
        ],
        "Feature": importance_pct.index,
        "Importance (%)": importance_pct.values.round(2),
    })

    left, right = st.columns([1.2, 1])

    with left:
        st.pyplot(
            make_bar_chart(
                importance_pct,
                "Relative Feature Importance",
                "Importance (%)",
            ),
            use_container_width=True,
        )
        plt.close("all")

    with right:
        st.dataframe(
            importance_display,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### Known model limitations")

    st.markdown(
        """
        <div class="warning-box">
            <b>Class imbalance:</b> the training data is strongly dominated by Shale,
            so accuracy alone should not be used as the only quality indicator.
        </div>
        <br>
        <div class="warning-box">
            <b>Rare lithologies:</b> classes with fewer training examples can be
            substantially harder to classify reliably.
        </div>
        <br>
        <div class="warning-box">
            <b>Log overlap:</b> lithologies can have similar petrophysical signatures,
            which can create genuine ambiguity between classes.
        </div>
        <br>
        <div class="danger-box">
            <b>Important:</b> this dashboard is a machine-learning decision-support
            system. It should not replace geological interpretation, core information,
            seismic context, or other domain evidence.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Reported weak spots")

    st.write(
        "• Chalk can be confused with Sandstone because of overlapping porosity signatures."
    )
    st.write(
        "• Dolomite performance is affected by incomplete log coverage in parts of the training data."
    )
    st.write(
        "• Low-confidence predictions should receive additional geological review."
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    """
    <div class="footer">
        Random Forest lithology classifier · FORCE 2020 & Kansas SEG 2016 datasets ·
        Portfolio / decision-support application
    </div>
    """,
    unsafe_allow_html=True,
)
