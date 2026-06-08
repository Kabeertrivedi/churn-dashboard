import pandas as pd
import numpy as np
import joblib
import shap
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report,roc_auc_score,confusion_matrix)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

#  LOADING DATA
df = pd.read_csv("data/churn.csv")
print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")

#  CLEANING DATA
# Dropping ID columns if they exist 
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)
# Fixing TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print(f"  Nulls after TotalCharges fix: {df['TotalCharges'].isnull().sum()}")

# Filling those nulls with median
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
# Encoding target: Yes=1, No=0
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Encoding all remaining string columns to numbers
cat_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
print(f"  Encoding {len(cat_cols)} categorical columns: {cat_cols}")
# Encoding categorical variables
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

#  SPLITTING DATA
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows")

#  TRAINING MODEL
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=2.7,  # handles class imbalance (more No than Yes)
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train, y_train)

#  MODEL EVALUATION
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n  Model Performance ")
print(f"  AUC-ROC Score : {roc_auc_score(y_test, y_proba):.3f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn','Churn'])}")

#  SHAP EXPLAINER
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Top 5 most important features overall
shap_df = pd.DataFrame({
    "feature": X_test.columns,
    "importance": np.abs(shap_values).mean(axis=0)
}).sort_values("importance", ascending=False)

print("\n Top 5 Churn Drivers ")
print(shap_df.head(5).to_string(index=False))

# SAVING MODEL AND METADATA
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/churn_model.pkl")
joblib.dump(explainer, "models/shap_explainer.pkl")
joblib.dump(X_test.columns.tolist(), "models/feature_names.pkl")
