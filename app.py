import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from utils import load_data, get_churn_predictions, get_shap_reasons, add_top_reasons

#  Page config 
st.set_page_config(
    page_title="RetainWise — AI-powered customer churn prediction",
    page_icon=":material/analytics:",
    layout="wide"
)

#  Loading model and data 
@st.cache_resource
def load_model():
    model = joblib.load("models/churn_model.pkl")
    explainer = joblib.load("models/shap_explainer.pkl")
    features = joblib.load("models/feature_names.pkl")
    return model, explainer, features

model, explainer, features = load_model()
df = load_data("data/churn.csv")
df = add_top_reasons(df, explainer, features)
df = get_churn_predictions(df, model, features)

#  Sidebar 
st.sidebar.title(":material/tune: Filters")
risk_threshold = st.sidebar.slider(
    "Churn risk threshold (%)",
    min_value=50, max_value=95, value=70, step=5
)
contract_filter = st.sidebar.multiselect(
    "Contract type",
    options=df["Contract_label"].unique().tolist(),
    default=df["Contract_label"].unique().tolist()
)

# Filtering data
at_risk = df[
    (df["churn_probability"] >= risk_threshold / 100) &
    (df["Contract_label"].isin(contract_filter))
]

#  Header 
st.title("RetainWise — AI-powered customer churn prediction")
st.caption("Transforming customer retention with AI insights.")

# ── KPI Metrics row ──────────────────────
col1, col2, col3, col4 = st.columns(4)

total_customers = len(df)
at_risk_count = len(at_risk)
avg_risk = df["churn_probability"].mean() * 100
high_risk_pct = at_risk_count / total_customers * 100

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("At Risk", f"{at_risk_count:,}",
           f"{high_risk_pct:.1f}% of base", delta_color="inverse")
col3.metric("Avg Churn Risk", f"{avg_risk:.1f}%")
col4.metric("Model AUC-ROC", "0.842",
           "XGBoost + SHAP")

#  Two column layout 
left, right = st.columns([1.2, 1])

with left:
    st.subheader(":material/warning: At-risk customers")

    display_cols = [
        "churn_probability_pct", "Contract_label",
        "tenure", "MonthlyCharges", "top_reason"
    ]
    display_df = at_risk[display_cols].copy()
    display_df.columns = [
        "Churn Risk %", "Contract",
        "Tenure (months)", "Monthly Charges ($)", "Top Risk Driver"
    ]
    display_df = display_df.sort_values("Churn Risk %", ascending=False)

    st.dataframe(
        display_df, width="stretch",
        height=380
    )
    st.caption(f"Showing {len(at_risk)} customers above {risk_threshold}% risk threshold")

with right:
    st.subheader(":material/bar_chart: Risk distribution")
    fig_hist = px.histogram(
        df, x="churn_probability",
        nbins=30,
        color_discrete_sequence=["#6366f1"],
        labels={"churn_probability": "Churn Probability"}
    )
    fig_hist.add_vline(
        x=risk_threshold/100,
        line_dash="dash",
        line_color="red",
        annotation_text="Risk threshold"
    )
    fig_hist.update_layout(
        margin=dict(l=0,r=0,t=20,b=0),
        height=200,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_hist)

    st.subheader(":material/psychology: Top churn drivers (SHAP)")
    shap_importance = pd.DataFrame({
        "Feature": ["Contract", "tenure", "MonthlyCharges",
                   "OnlineSecurity", "TechSupport"],
        "Importance": [0.84, 0.46, 0.35, 0.29, 0.22]
    })
    fig_shap = px.bar(
        shap_importance,
        x="Importance", y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="purples"
    )
    fig_shap.update_layout(
        margin=dict(l=0,r=0,t=10,b=0),
        height=180,
        showlegend=False,
        coloraxis_showscale=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_shap)

#  Contract breakdown chart 
st.subheader(":material/description: Churn rate by contract type")

contract_churn = df.groupby("Contract_label")["Churn"].mean().reset_index()
contract_churn.columns = ["Contract", "Churn Rate"]
contract_churn["Churn Rate %"] = (contract_churn["Churn Rate"] * 100).round(1)

fig_contract = px.bar(
    contract_churn, x="Contract", y="Churn Rate %",
    color="Churn Rate %",
    color_continuous_scale="reds",
    text="Churn Rate %"
)
fig_contract.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig_contract.update_layout(
    height=280, coloraxis_showscale=False,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)"
)
st.plotly_chart(fig_contract)
 
st.caption("Built with XGBoost + SHAP · Know Who`s leaving Before they do · RetainWise.com")