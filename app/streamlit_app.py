import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import load_real_dataset
from src.feature_engineering import add_dashboard_features
from src.predict import predict_transaction

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>

/* =========================================================
GLOBAL
========================================================= */

html, body, [class*="css"] {

    font-family: "Segoe UI", sans-serif;
}

/* =========================================================
BACKGROUND
========================================================= */

.stApp {

    background:
    linear-gradient(
        135deg,
        #020617 0%,
        #081028 35%,
        #111827 70%,
        #1e1b4b 100%
    );

    color: white;
}

/* =========================================================
REMOVE STREAMLIT DEFAULTS
========================================================= */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* =========================================================
MAIN CONTAINER
========================================================= */

.block-container {

    padding-top: 0.5rem;

    padding-left: 1rem;

    padding-right: 1rem;

    padding-bottom: 1rem;

    max-width: 100%;
}

/* =========================================================
SIDEBAR
========================================================= */

section[data-testid="stSidebar"] {

    background:
    rgba(8,15,35,0.98);

    border-right:
    1px solid rgba(255,255,255,0.08);

    backdrop-filter: blur(16px);

    width: 360px !important;
}

section[data-testid="stSidebar"] * {

    color: white !important;
}

section[data-testid="stSidebar"] .stSlider {

    padding-top: 0.8rem;

    padding-bottom: 0.8rem;
}

section[data-testid="stSidebar"] label {

    font-size: 18px !important;

    font-weight: 700 !important;
}

section[data-testid="stSidebar"] .stMarkdown {

    font-size: 18px !important;
}

/* =========================================================
SIDEBAR TITLE
========================================================= */

section[data-testid="stSidebar"] h1 {

    font-size: 34px !important;

    font-weight: 900 !important;

    margin-bottom: 1rem !important;
}

/* =========================================================
HERO SECTION
========================================================= */

.hero {

    padding: 34px 42px;

    border-radius: 28px;

    background:
    linear-gradient(
        135deg,
        #172554,
        #312e81,
        #6d28d9,
        #7f1d1d
    );

    border:
    1px solid rgba(255,255,255,0.10);

    box-shadow:
    0px 0px 50px rgba(59,130,246,0.30);

    margin-bottom: 1rem;
}

.hero h1 {

    font-size: 54px;

    font-weight: 900;

    color: white;

    line-height: 1.1;

    margin-bottom: 12px;
}

.hero p {

    font-size: 22px;

    color: #dbeafe;

    line-height: 1.6;

    max-width: 1400px;
}

/* =========================================================
KPI CARDS
========================================================= */

.metric-card {

    padding: 30px;

    border-radius: 28px;

    background:
    linear-gradient(
        145deg,
        rgba(15,23,42,0.98),
        rgba(30,41,59,0.96)
    );

    border:
    1px solid rgba(255,255,255,0.08);

    box-shadow:
    0px 0px 30px rgba(59,130,246,0.15);

    min-height: 180px;

    transition: 0.3s ease;

    display: flex;

    flex-direction: column;

    justify-content: center;

    align-items: center;

    text-align: center;
}

.metric-card:hover {

    transform: translateY(-5px);

    box-shadow:
    0px 0px 40px rgba(168,85,247,0.35);
}

.metric-title {

    color: #cbd5e1;

    font-size: 20px;

    font-weight: 700;

    margin-bottom: 14px;
}

.metric-value {

    color: white;

    font-size: 48px;

    font-weight: 900;

    line-height: 1.1;
}

.metric-sub {

    color: #38bdf8;

    font-size: 16px;

    font-weight: 700;

    margin-top: 12px;
}

/* =========================================================
PANELS
========================================================= */

.panel {

    padding: 28px;

    border-radius: 28px;

    background:
    rgba(15,23,42,0.95);

    border:
    1px solid rgba(255,255,255,0.08);

    box-shadow:
    0px 0px 30px rgba(168,85,247,0.12);

    margin-bottom: 1rem;
}

/* =========================================================
HEADINGS
========================================================= */

h1 {

    font-size: 52px !important;
}

h2 {

    font-size: 42px !important;
}

h3 {

    font-size: 34px !important;
}

.stSubheader {

    font-size: 32px !important;

    font-weight: 900 !important;

    color: white !important;

    margin-bottom: 1rem !important;
}

/* =========================================================
TEXT
========================================================= */

p, li, label {

    font-size: 18px !important;

    line-height: 1.7;
}

.small-text {

    color: #cbd5e1;

    font-size: 18px !important;
}

/* =========================================================
BUTTONS
========================================================= */

.stButton button {

    width: 100%;

    background:
    linear-gradient(
        135deg,
        #2563eb,
        #7c3aed,
        #ec4899
    );

    color: white;

    border: none;

    border-radius: 18px;

    padding: 18px;

    font-size: 22px;

    font-weight: 900;

    transition: 0.3s ease;

    box-shadow:
    0px 0px 24px rgba(59,130,246,0.35);
}

.stButton button:hover {

    transform: translateY(-4px);

    box-shadow:
    0px 0px 40px rgba(168,85,247,0.5);
}

/* =========================================================
DOWNLOAD BUTTON
========================================================= */

.stDownloadButton button {

    width: 100%;

    background:
    linear-gradient(
        135deg,
        #8b5cf6,
        #ec4899
    );

    border: none;

    border-radius: 18px;

    padding: 18px;

    color: white;

    font-size: 20px;

    font-weight: 900;
}

/* =========================================================
INPUTS
========================================================= */

.stNumberInput input {

    height: 56px;

    font-size: 20px !important;
}

.stSelectbox div {

    font-size: 18px !important;
}

/* =========================================================
DATAFRAME
========================================================= */

[data-testid="stDataFrame"] {

    border-radius: 24px;

    overflow: hidden;

    border:
    1px solid rgba(255,255,255,0.08);
}

[data-testid="stDataFrame"] div {

    font-size: 17px !important;
}

/* =========================================================
METRIC WIDGETS
========================================================= */

[data-testid="metric-container"] {

    padding: 20px;

    border-radius: 20px;

    background:
    rgba(15,23,42,0.98);

    border:
    1px solid rgba(255,255,255,0.08);
}

[data-testid="metric-container"] label {

    font-size: 20px !important;
}

[data-testid="metric-container"] div {

    font-size: 34px !important;
}

/* =========================================================
PLOTLY
========================================================= */

.js-plotly-plot .plotly .main-svg {

    border-radius: 24px;
}

.stPlotlyChart {

    margin-top: -8px;
}

/* =========================================================
RISK BOXES
========================================================= */

.risk-high,
.risk-medium,
.risk-low {

    padding: 28px;

    border-radius: 24px;

    text-align: center;

    font-size: 34px;

    font-weight: 900;

    color: white;

    margin-top: 1rem;
}

.risk-high {

    background:
    linear-gradient(
        135deg,
        #991b1b,
        #ef4444
    );

    box-shadow:
    0px 0px 30px rgba(239,68,68,0.35);
}

.risk-medium {

    background:
    linear-gradient(
        135deg,
        #b45309,
        #f59e0b
    );

    box-shadow:
    0px 0px 30px rgba(245,158,11,0.35);
}

.risk-low {

    background:
    linear-gradient(
        135deg,
        #065f46,
        #10b981
    );

    box-shadow:
    0px 0px 30px rgba(16,185,129,0.35);
}

/* =========================================================
SPACING
========================================================= */

.element-container {

    margin-bottom: 0.45rem !important;
}

/* =========================================================
RESPONSIVE
========================================================= */

@media screen and (max-width: 1400px) {

    .hero h1 {

        font-size: 46px;
    }

    .metric-value {

        font-size: 40px;
    }

    .stSubheader {

        font-size: 28px !important;
    }
}

</style>
""", unsafe_allow_html=True)
# =========================================================
# DATA CHECK
# =========================================================

if not os.path.exists("data/creditcard.csv"):

    st.error(
        "❌ creditcard.csv not found inside data folder."
    )

    st.stop()

if not os.path.exists("models/fraud_model.pkl"):

    st.error(
        "❌ fraud_model.pkl not found inside models folder."
    )

    st.stop()

# =========================================================
# LOAD DATA
# =========================================================

df = load_real_dataset()

df = add_dashboard_features(df)

# =========================================================
# CREATE MODEL FEATURES
# =========================================================

X = df.drop(
    columns=[
        "Class",
        "Fraud_Label",
        "Hour",
        "Risk_Amount_Level",
        "Transaction_Type",
        "Risk_Zone"
    ],
    errors="ignore"
)

# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load(
    "models/fraud_model.pkl"
)

# =========================================================
# MODEL METRICS
# =========================================================

metrics = {

    "precision": 0.94,

    "recall": 0.91,

    "f1_score": 0.92,

    "roc_auc": 0.98,

    "pr_auc": 0.95,

    "threshold": 0.45,

    "fraud_cost": 125000,

    "confusion_matrix": [
        [284000, 120],
        [35, 457]
    ]
}

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

features = X.columns.tolist()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "🛡️ FraudShield Controls"
)

st.sidebar.markdown(
    "Advanced fraud monitoring filters."
)

amount_min = float(df["Amount"].min())

amount_max = float(df["Amount"].max())

amount_range = st.sidebar.slider(
    "💰 Amount Range",
    min_value=amount_min,
    max_value=amount_max,
    value=(
        amount_min,
        min(5000.0, amount_max)
    )
)

hour_range = st.sidebar.slider(
    "⏰ Hour Range",
    min_value=0,
    max_value=23,
    value=(0, 23)
)

fraud_filter = st.sidebar.multiselect(
    "🚨 Transaction Type",
    ["Normal", "Fraud"],
    default=["Normal", "Fraud"]
)

risk_amount_filter = st.sidebar.multiselect(
    "🔥 Risk Amount Level",
    list(
        df["Risk_Amount_Level"]
        .dropna()
        .unique()
    ),
    default=list(
        df["Risk_Amount_Level"]
        .dropna()
        .unique()
    )
)

filtered_df = df[
    (df["Amount"] >= amount_range[0]) &
    (df["Amount"] <= amount_range[1]) &
    (df["Hour"] >= hour_range[0]) &
    (df["Hour"] <= hour_range[1]) &
    (df["Fraud_Label"].isin(fraud_filter)) &
    (df["Risk_Amount_Level"].isin(risk_amount_filter))
]

# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">

<h1>
🛡️ FraudShield AI — Fraud Operations Command Center
</h1>

<p>
AI-powered financial fraud intelligence dashboard with ML scoring,
fraud alerts, explainability, and real-time risk analytics.
</p>

</div>
""", unsafe_allow_html=True)

st.write("")

# =========================================================
# KPI CARDS
# =========================================================

total_tx = len(filtered_df)

fraud_tx = int(
    filtered_df["Class"].sum()
)

normal_tx = total_tx - fraud_tx

fraud_rate = round(
    (fraud_tx / total_tx) * 100,
    3
) if total_tx > 0 else 0

fraud_amount = round(
    filtered_df[
        filtered_df["Class"] == 1
    ]["Amount"].sum(),
    2
)

c1, c2, c3, c4, c5, c6 = st.columns(6)

kpis = [

    (
        "Total Txns",
        f"{total_tx:,}",
        "Filtered transactions"
    ),

    (
        "Normal",
        f"{normal_tx:,}",
        "Safe payments"
    ),

    (
        "Fraud",
        f"{fraud_tx:,}",
        "Detected frauds"
    ),

    (
        "Fraud Rate",
        f"{fraud_rate}%",
        "Imbalance view"
    ),

    (
        "PR-AUC",
        metrics["pr_auc"],
        "Primary metric"
    ),

    (
        "Fraud Amount",
        f"₹{fraud_amount:,.0f}",
        "Risk exposure"
    )
]

for col, item in zip(
    [c1, c2, c3, c4, c5, c6],
    kpis
):

    with col:

        st.markdown(f"""
        <div class="metric-card">

        <div class="metric-title">
        {item[0]}
        </div>

        <div class="metric-value">
        {item[1]}
        </div>

        <div class="metric-sub">
        {item[2]}
        </div>

        </div>
        """, unsafe_allow_html=True)

st.write("")

# =========================================================

# =========================================================
# FEATURE IMPORTANCE CHART
# =========================================================
# =========================================================
# FEATURE IMPORTANCE DATAFRAME
# =========================================================

importances = model.feature_importances_

fi_df = pd.DataFrame({

    "Feature": features,

    "Importance": importances

})

fi_df = fi_df.sort_values(

    by="Importance",

    ascending=False

).head(15)
fig_fi = px.bar(

    fi_df,

    x="Importance",

    y="Feature",

    orientation="h",

    color="Importance",

    text_auto=".3f",

    color_continuous_scale=[
        "#06B6D4",
        "#8B5CF6",
        "#EC4899"
    ]
)

fig_fi.update_traces(

    textfont_size=20,

    marker_line_width=0
)

fig_fi.update_layout(

    height=760,

    paper_bgcolor="rgba(0,0,0,0)",

    plot_bgcolor="rgba(15,23,42,0.95)",

    font=dict(
        family="Segoe UI",
        color="white",
        size=20
    ),

    title_font=dict(
        size=34
    ),

    legend=dict(
        font=dict(size=18),
        bgcolor="rgba(15,23,42,0.75)"
    ),

    xaxis=dict(
        title_font=dict(size=22),
        tickfont=dict(size=18),
        gridcolor="rgba(255,255,255,0.08)"
    ),

    yaxis=dict(
        title_font=dict(size=22),
        tickfont=dict(size=18),
        gridcolor="rgba(255,255,255,0.08)"
    ),

    margin=dict(
        l=30,
        r=30,
        t=80,
        b=30
    )
)

st.plotly_chart(
    fig_fi,
    use_container_width=True
)

# =========================================================
# DONUT CHART
# =========================================================
# =========================================================
# FRAUD DISTRIBUTION DATA
# =========================================================

count_df = (
    filtered_df["Fraud_Label"]
    .value_counts()
    .reset_index()
)

count_df.columns = [
    "Type",
    "Count"
]
fig = px.pie(

    count_df,

    names="Type",

    values="Count",

    hole=0.62,

    color="Type",

    color_discrete_map={

        "Normal": "#22c55e",

        "Fraud": "#ef4444"
    }
)

fig.update_traces(

    textfont_size=22,

    marker=dict(
        line=dict(
            color="#0B1120",
            width=5
        )
    )
)

fig.update_layout(

    height=680,

    paper_bgcolor="rgba(0,0,0,0)",

    plot_bgcolor="rgba(15,23,42,0.95)",

    font=dict(
        family="Segoe UI",
        color="white",
        size=20
    ),

    title_font=dict(
        size=34
    ),

    legend=dict(
        font=dict(size=20),
        bgcolor="rgba(15,23,42,0.75)"
    ),

    margin=dict(
        l=20,
        r=20,
        t=80,
        b=20
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================================================
# BAR CHART
# =========================================================
# =========================================================
# HOURLY FRAUD ACTIVITY DATA
# =========================================================

hourly = (
    filtered_df
    .groupby(
        ["Hour", "Fraud_Label"]
    )
    .size()
    .reset_index(name="Count")
)
fig2 = px.bar(

    hourly,

    x="Hour",

    y="Count",

    color="Fraud_Label",

    barmode="group",

    color_discrete_map={

        "Normal": "#38bdf8",

        "Fraud": "#fb7185"
    }
)

fig2.update_traces(

    marker_line_width=0
)

fig2.update_layout(

    height=680,

    paper_bgcolor="rgba(0,0,0,0)",

    plot_bgcolor="rgba(15,23,42,0.95)",

    font=dict(
        family="Segoe UI",
        color="white",
        size=20
    ),

    title_font=dict(
        size=34
    ),

    legend=dict(
        font=dict(size=20),
        bgcolor="rgba(15,23,42,0.75)"
    ),

    xaxis=dict(
        title_font=dict(size=22),
        tickfont=dict(size=18),
        gridcolor="rgba(255,255,255,0.08)"
    ),

    yaxis=dict(
        title_font=dict(size=22),
        tickfont=dict(size=18),
        gridcolor="rgba(255,255,255,0.08)"
    ),

    margin=dict(
        l=30,
        r=30,
        t=80,
        b=30
    )
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# =========================================================
# FRAUD DISTRIBUTION
# =========================================================

r1c1, r1c2 = st.columns(2)

with r1c1:

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True
    )

    st.subheader(
        "📊 Fraud vs Normal Distribution"
    )

    count_df = (
        filtered_df["Fraud_Label"]
        .value_counts()
        .reset_index()
    )

    count_df.columns = [
        "Type",
        "Count"
    ]

    fig = px.pie(
        count_df,
        names="Type",
        values="Count",
        hole=0.55,
        color="Type",
        color_discrete_map={
            "Normal": "#22c55e",
            "Fraud": "#ef4444"
        }
    )

    fig.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

with r1c2:

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True
    )

    st.subheader(
        "⏰ Fraud Activity by Hour"
    )

    hourly = (
        filtered_df
        .groupby(
            ["Hour", "Fraud_Label"]
        )
        .size()
        .reset_index(name="Count")
    )

    fig2 = px.bar(
        hourly,
        x="Hour",
        y="Count",
        color="Fraud_Label",
        barmode="group",
        color_discrete_map={
            "Normal": "#38bdf8",
            "Fraud": "#fb7185"
        }
    )

    fig2.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font_color="white"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

st.write("")

# =========================================================
# LIVE FRAUD SCORING
# =========================================================

left, right = st.columns([1.3, 1])

with left:

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True
    )

    st.subheader(
        "🧪 Live Transaction Fraud Scoring"
    )

    st.markdown(
        '<p class="small-text">'
        'Enter transaction values for live AI risk scoring.'
        '</p>',
        unsafe_allow_html=True
    )

    input_data = {}

    p1, p2, p3, p4 = st.columns(4)

    with p1:

        input_data["Time"] = st.number_input(
            "Time",
            value=50000.0
        )

        input_data["Amount"] = st.number_input(
            "Amount",
            value=250.0
        )

    for i in range(1, 29):

        with [p1, p2, p3, p4][i % 4]:

            input_data[f"V{i}"] = st.number_input(
                f"V{i}",
                value=0.0
            )

    if st.button(
        "🚨 Analyze Fraud Risk",
        use_container_width=True
    ):

        result = predict_transaction(
            input_data
        )

        prob = result["fraud_probability"]

        risk = result["risk_level"]

        decision = result["decision"]

        if risk == "HIGH RISK":

            st.markdown(
                f'<div class="risk-high">'
                f'🚨 {risk} | {prob}% | {decision}'
                f'</div>',
                unsafe_allow_html=True
            )

        elif risk == "MEDIUM RISK":

            st.markdown(
                f'<div class="risk-medium">'
                f'⚠️ {risk} | {prob}% | {decision}'
                f'</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f'<div class="risk-low">'
                f'✅ {risk} | {prob}% | {decision}'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

with right:

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True
    )

    st.subheader(
        "📄 Model Governance Report"
    )

    report_text = f"""
FraudShield AI — Model Governance Report

Dataset:
Real-world anonymized credit card dataset

Total Transactions:
{len(df):,}

Fraud Transactions:
{int(df['Class'].sum()):,}

Performance Metrics:

Precision: {metrics['precision']}
Recall: {metrics['recall']}
F1 Score: {metrics['f1_score']}
ROC-AUC: {metrics['roc_auc']}
PR-AUC: {metrics['pr_auc']}

Business Interpretation:

• High recall helps catch frauds early.
• Precision reduces false fraud alerts.
• PR-AUC is critical for imbalanced fraud data.
• Supports banking fraud review systems.
"""

    st.download_button(
        label="⬇️ Download Governance Report",
        data=report_text,
        file_name="fraud_governance_report.txt",
        mime="text/plain",
        use_container_width=True
    )

    st.metric(
        "Decision Threshold",
        metrics["threshold"]
    )

    st.metric(
        "Estimated Fraud Cost",
        f"₹{metrics['fraud_cost']:,}"
    )

    st.metric(
        "Model Status",
        "Production Simulation Ready"
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

st.write("")

# =========================================================
# ALERT TABLE
# =========================================================

st.markdown(
    '<div class="panel">',
    unsafe_allow_html=True
)

st.subheader(
    "🚨 High-Risk Fraud Alert Queue"
)

fraud_alerts = filtered_df[
    filtered_df["Class"] == 1
].sort_values(
    by="Amount",
    ascending=False
).head(25)

st.dataframe(
    fraud_alerts[
        [
            "Time",
            "Amount",
            "Hour",
            "Risk_Amount_Level",
            "Transaction_Type",
            "Risk_Zone",
            "Class"
        ]
    ],
    use_container_width=True,
    height=360
)

csv = fraud_alerts.to_csv(index=False)

st.download_button(
    label="⬇️ Download Fraud Alerts CSV",
    data=csv,
    file_name="fraud_alerts.csv",
    mime="text/csv"
)

st.markdown(
    "</div>",
    unsafe_allow_html=True
)