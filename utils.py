import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

CONTRACT_MAP = {0: "Month-to-month", 1: "One year", 2: "Two year"}

FEATURE_LABELS = {
    "Contract": "Contract type",
    "tenure": "Low tenure",
    "MonthlyCharges": "High monthly charges",
    "OnlineSecurity": "No online security",
    "TechSupport": "No tech support",
    "TotalCharges": "Low total charges",
    "InternetService": "Internet service type",
    "PaymentMethod": "Payment method",
}

def load_data(path: str) -> pd.DataFrame:
    """Loading and cleaning the churn dataset."""
    df = pd.read_csv(path)


    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)


    df["Contract_label"] = df["Contract"]
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Keep customerID for UI search functionality
    # df.drop(columns=["customerID"], inplace=True, errors="ignore")

    # Encoding categorical columns
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        if col not in ["customerID", "Contract_label"]:
            df[col] = le.fit_transform(df[col])

    return df

def get_churn_predictions(df: pd.DataFrame, model, features: list) -> pd.DataFrame:
    """Adding churn probability predictions to the dataframe."""
    X = df[features]
    proba = model.predict_proba(X)[:, 1]
    df["churn_probability"] = proba
    df["churn_probability_pct"] = (proba * 100).round(1)
    df["risk_label"] = pd.cut(
        proba,
        bins=[0, 0.4, 0.7, 1.0],
        labels=["Low", "Medium", "High"]
    )
    return df


def get_shap_reasons(explainer, row: pd.Series, features: list,
                    top_n: int = 3) -> str:
    """Returning plain English top SHAP reason for one customer."""
    shap_vals = explainer.shap_values(row.values.reshape(1, -1))[0]
    top_idx = np.argsort(np.abs(shap_vals))[::-1][:top_n]
    reasons = []
    for i in top_idx:
        feat = features[i]
        label = FEATURE_LABELS.get(feat, feat)
        reasons.append(label)
    return ", ".join(reasons)

# Adding top reason column to at-risk rows efficiently
def add_top_reasons(df: pd.DataFrame, explainer, features: list) -> pd.DataFrame:
    X = df[features]
    shap_vals = explainer.shap_values(X)
    top_idx = np.argmax(np.abs(shap_vals), axis=1)
    df["top_reason"] = [
        FEATURE_LABELS.get(features[i], features[i])
        for i in top_idx
    ]
    return df