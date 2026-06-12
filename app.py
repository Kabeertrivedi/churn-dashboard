import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from ai_layer import get_cached_intervention
from utils import load_data, get_churn_predictions, get_shap_reasons, add_top_reasons

#  Page config 
st.set_page_config(
    page_title="RetainWiseIQ | Enterprise Churn Intelligence",
    page_icon="🧠",
    layout="wide"
)

#  Custom Enterprise CSS 
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Executive Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.05), 0 2px 4px -1px rgba(15, 23, 42, 0.03);
        transition: all 0.2s ease-in-out;
    }
    div[data-testid="stMetric"]:hover {
        box-shadow: 0 10px 15px -3px rgba(15, 23, 42, 0.08), 0 4px 6px -2px rgba(15, 23, 42, 0.04);
        transform: translateY(-2px);
        border-color: #CBD5E1;
    }
    
    /* AI Recommendation Callout */
    div.stInfo {
        background-color: #F8FAFC;
        color: #0F172A;
        border-left: 4px solid #2563EB;
        border-radius: 6px;
        padding: 16px;
        font-size: 1.05rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    
    /* Clean up default UI elements */
    header {visibility: hidden;}
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
</style>
""", unsafe_allow_html=True)

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
st.sidebar.markdown("## 🧠 RetainWiseIQ")
st.sidebar.markdown("**Executive Dashboard Controls**")
risk_threshold = st.sidebar.slider(
    "Churn risk threshold (%)",
    min_value=50, max_value=95, value=70, step=5
)
contract_filter = st.sidebar.multiselect(
    "Contract type",
    options=df["Contract_label"].unique().tolist(),
    default=df["Contract_label"].unique().tolist()
)
st.sidebar.divider()
st.sidebar.caption("System Status: **Optimal** 🟢")
st.sidebar.caption("Last model sync: *Just now*")
st.sidebar.caption("Prediction Engine: *XGBoost v2.1*")
st.sidebar.caption("AI Engine: *Grok Llama*")


# Filtering data
at_risk = df[
    (df["churn_probability"] >= risk_threshold / 100) &
    (df["Contract_label"].isin(contract_filter))
]

#  Header 
st.title("Customer Retention Intelligence")
st.markdown("<p style='font-size: 1.2rem; color: #475569; margin-top: -10px; margin-bottom: 30px;'>Executive overview of churn probability and automated AI interventions.</p>", unsafe_allow_html=True)

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

st.write("")
st.write("")

#  Tabbed Navigation 
tab_dash, tab_ai = st.tabs(["📊 Executive Dashboard", "🎯 AI Customer Intelligence"])

with tab_dash:
    st.write("")
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("#### ⚠️ High Priority At-Risk Accounts")
        
        display_cols = [
            "churn_probability_pct", "Contract_label",
            "tenure", "MonthlyCharges", "top_reason"
        ]
        if "customerID" in at_risk.columns:
            display_cols.insert(0, "customerID")
            
        display_df = at_risk[display_cols].copy()
        
        new_cols = ["Risk (%)", "Contract", "Tenure (mo)", "MRR ($)", "Top Risk Driver"]
        if "customerID" in at_risk.columns:
            new_cols.insert(0, "Customer ID")
            
        display_df.columns = new_cols
        display_df = display_df.sort_values("Risk (%)", ascending=False)

        st.dataframe(
            display_df, width="stretch",
            height=320, hide_index=True
        )
        st.caption(f"Showing {len(at_risk)} accounts above {risk_threshold}% risk threshold")
        
        st.markdown("#### 📄 Churn Rate by Contract")
        contract_churn = df.groupby("Contract_label")["Churn"].mean().reset_index()
        contract_churn.columns = ["Contract", "Churn Rate"]
        contract_churn["Churn Rate %"] = (contract_churn["Churn Rate"] * 100).round(1)

        fig_contract = px.bar(
            contract_churn, x="Contract", y="Churn Rate %",
            color="Churn Rate %",
            color_continuous_scale=["#E2E8F0", "#2563EB"], # Slate to Enterprise Blue
            text="Churn Rate %"
        )
        fig_contract.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_line_width=0)
        fig_contract.update_layout(
            height=280, coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0,r=0,t=20,b=0)
        )
        fig_contract.update_xaxes(showgrid=False)
        fig_contract.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
        st.plotly_chart(fig_contract, use_container_width=True)

    with right:
        st.markdown("#### 📈 Portfolio Risk Distribution")
        fig_hist = px.histogram(
            df, x="churn_probability", nbins=30,
            color_discrete_sequence=["#334155"], # Slate 700
            labels={"churn_probability": "Churn Probability"}
        )
        fig_hist.add_vline(
            x=risk_threshold/100, line_dash="dash", line_color="#EF4444", # Red 500
            annotation_text="Action Threshold"
        )
        fig_hist.update_layout(
            margin=dict(l=0,r=0,t=20,b=20), height=220,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Account Count"
        )
        fig_hist.update_xaxes(showgrid=False)
        fig_hist.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
        st.plotly_chart(fig_hist, use_container_width=True)

        st.write("")
        st.markdown("#### 🧠 Global SHAP Explainability")
        st.caption("Top underlying factors driving churn probability across all accounts.")
        shap_importance = pd.DataFrame({
            "Feature": ["Contract", "Tenure", "Monthly Charges",
                       "Online Security", "Tech Support"],
            "Importance": [0.84, 0.46, 0.35, 0.29, 0.22]
        })
        fig_shap = px.bar(
            shap_importance, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale=["#CBD5E1", "#0F172A"] # Slate to Navy
        )
        fig_shap.update_layout(
            margin=dict(l=0,r=0,t=10,b=0), height=240, showlegend=False,
            coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        fig_shap.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
        fig_shap.update_yaxes(showgrid=False, categoryorder="total ascending")
        st.plotly_chart(fig_shap, use_container_width=True)

with tab_ai:
    st.write("")
    st.markdown("#### 🎯 Account Interventions")
    st.caption("Search for a specific customer or select a high-risk account below to generate a tailored retention strategy.")
    
    col_search, col_select = st.columns(2)
    
    with col_search:
        search_query = st.text_input("🔍 Search Account", placeholder="Enter Customer ID...")
        
    if search_query:
        if "customerID" in df.columns:
            target_accounts = df[df["customerID"].astype(str).str.contains(search_query, case=False)]
        else:
            target_accounts = df
    else:
        target_accounts = at_risk.sort_values("churn_probability", ascending=False).head(10)
    
    if len(target_accounts) == 0:
        st.info("No accounts found matching your search or current risk filters.")
    else:
        with col_select:
            def format_customer(i):
                row = target_accounts.iloc[i]
                acc_id = row['customerID'] if 'customerID' in target_accounts.columns else f"#{i+1}"
                return f"Account {acc_id} — {row['Contract_label']} | {row['churn_probability_pct']:.2f}% Risk | MRR: ${row['MonthlyCharges']:.0f}"
    
            selected_idx = st.selectbox(
                "Top Risk Accounts",
                options=range(len(target_accounts)),
                format_func=format_customer
            )
        selected_row = target_accounts.iloc[selected_idx]
    
        st.write("")
        col_info, col_rec = st.columns([1, 1.8])
    
        with col_info:
            st.markdown("**Account Telemetry**")
            st.metric("Calculated Risk Score",
                      f"{selected_row['churn_probability_pct']:.2f}%",
                      delta_color="off")
            st.write(f"📑 **Contract Stage:** {selected_row['Contract_label']}")
            st.write(f"⏳ **Lifecycle Tenure:** {selected_row['tenure']} months")
            st.write(f"💳 **Current MRR:** ${selected_row['MonthlyCharges']:.2f}")
            st.write(f"🚨 **Primary Churn Vector:** {selected_row['top_reason']}")
    
        with col_rec:
            st.markdown("**Automated Retention Strategy**")
            with st.spinner("Synthesizing telemetry data and generating executive action plan..."):
                recommendation = get_cached_intervention(
                    churn_risk=round(selected_row["churn_probability_pct"], 1),
                    contract=selected_row["Contract_label"],
                    tenure_months=int(selected_row["tenure"]),
                    monthly_charges=float(selected_row["MonthlyCharges"]),
                    top_drivers=selected_row["top_reason"]
                )
            st.info(recommendation)
            st.caption("⚡ Strategy synthesized by RetainWiseIQ ")

st.divider()
st.markdown("<p style='text-align: center; color: #94A3B8; font-size: 0.85rem;'>RetainWiseIQ  v2.1.0 · Secured Data Pipeline · Models powered by XGBoost & SHAP</p>", unsafe_allow_html=True)