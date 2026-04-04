# ============================================================
# CUSTOMER CHURN PREDICTION — STREAMLIT APP
# app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(
    page_title="Churn Prediction Tool",
    page_icon="📉",
    layout="wide"
)

# --- Load Model Artifacts ---
@st.cache_resource
def load_artifacts():
    model = joblib.load('Models/xgboost_model.pkl')
    scaler = joblib.load('Models/scaler.pkl')
    feature_names = joblib.load('Models/feature_names.pkl')
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()

# --- Header ---
st.title("📉 Customer Churn Prediction")
st.markdown("**Enter customer details to predict churn risk and understand the key drivers.**")
st.divider()

# --- Sidebar Inputs ---
st.sidebar.header("Customer Profile")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18, 120, 65)
total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 9000.0, float(tenure * monthly_charges))

senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
phone_service = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])

contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])

online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

# --- Build Input DataFrame ---
def build_input():
    input_dict = {f: 0 for f in feature_names}

    # Numeric features
    input_dict['tenure'] = tenure
    input_dict['MonthlyCharges'] = monthly_charges
    input_dict['TotalCharges'] = total_charges
    input_dict['SeniorCitizen'] = 1 if senior_citizen == "Yes" else 0

    # Binary encoded
    input_dict['gender'] = 1 if gender == "Male" else 0
    input_dict['Partner'] = 1 if partner == "Yes" else 0
    input_dict['Dependents'] = 1 if dependents == "Yes" else 0
    input_dict['PhoneService'] = 1 if phone_service == "Yes" else 0
    input_dict['PaperlessBilling'] = 1 if paperless_billing == "Yes" else 0

    # One-hot encoded — match exact feature names from training
    contract_key = f'Contract_{contract}'
    internet_key = f'InternetService_{internet_service}'
    payment_key = f'PaymentMethod_{payment_method}'
    security_key = f'OnlineSecurity_{online_security}'
    backup_key = f'OnlineBackup_{online_backup}'
    device_key = f'DeviceProtection_{device_protection}'
    support_key = f'TechSupport_{tech_support}'
    tv_key = f'StreamingTV_{streaming_tv}'
    movies_key = f'StreamingMovies_{streaming_movies}'
    lines_key = f'MultipleLines_{multiple_lines}'

    # Only set if key exists in feature names
    for key in [contract_key, internet_key, payment_key, security_key,
                backup_key, device_key, support_key, tv_key, movies_key, lines_key]:
        if key in input_dict:
            input_dict[key] = 1
        else:
            st.sidebar.warning(f"Feature not found: {key}")

    return pd.DataFrame([input_dict])

# --- Prediction ---
input_df = build_input()
input_scaled = scaler.transform(input_df)
churn_prob = model.predict_proba(input_scaled)[0][1]

if churn_prob >= 0.4:
    churn_pred = "🔴 High Churn Risk"
elif churn_prob >= 0.25:
    churn_pred = "🟡 Medium Churn Risk"
else:
    churn_pred = "🟢 Low Churn Risk"

priority = "HIGH" if churn_prob >= 0.4 else "MEDIUM" if churn_prob >= 0.25 else "LOW"

# --- Results Display ---
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Churn Probability", f"{churn_prob:.1%}")
with col2:
    st.metric("Risk Level", churn_pred)
with col3:
    st.metric("Retention Priority", priority)

st.divider()

# --- SHAP Explanation ---
st.subheader("Why is this customer at risk?")

@st.cache_resource
def get_explainer(_model, _background):
    return shap.TreeExplainer(_model, _background)

# Load background data for SHAP
@st.cache_resource
def load_background():
    df_bg = pd.read_csv('Data/Telco-Customer-Churn.csv')
    
    # Same preprocessing as notebook
    df_bg.drop(columns=['customerID'], inplace=True)
    df_bg['TotalCharges'] = pd.to_numeric(
        df_bg['TotalCharges'].replace(' ', np.nan), errors='coerce'
    )
    df_bg.dropna(subset=['TotalCharges'], inplace=True)
    df_bg.reset_index(drop=True, inplace=True)

    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
                   'PaperlessBilling', 'Churn']
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in binary_cols:
        df_bg[col] = le.fit_transform(df_bg[col])

    multi_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod'
    ]
    df_bg = pd.get_dummies(df_bg, columns=multi_cols, drop_first=False, dtype=int)
    df_bg = df_bg.drop(columns=['Churn'])
    df_bg = df_bg[feature_names]
    
    # Scale it
    scaler_bg = joblib.load('Models/scaler.pkl')
    return pd.DataFrame(scaler_bg.transform(df_bg), columns=feature_names)

background_data = load_background()
explainer = get_explainer(model, background_data)
shap_values = explainer.shap_values(input_scaled)

fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(
    shap_values.reshape(1, -1),
    pd.DataFrame(input_scaled, columns=feature_names),
    plot_type='bar',
    max_display=10,
    show=False
)
plt.title("Top Features Driving This Prediction", fontweight='bold')
plt.tight_layout()
st.pyplot(fig)
plt.close()