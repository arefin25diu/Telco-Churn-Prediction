import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("churn_model.pkl")

st.title("📊 Customer Churn Prediction App")

st.write("Enter customer details below")

# =====================================
# Input Fields
# =====================================
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0,1])
partner = st.selectbox("Partner", ["Yes","No"])
dependents = st.selectbox("Dependents", ["Yes","No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)

phone = st.selectbox("Phone Service", ["Yes","No"])
multi = st.selectbox("Multiple Lines", ["Yes","No","No phone service"])
internet = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])

security = st.selectbox("Online Security", ["Yes","No","No internet service"])
backup = st.selectbox("Online Backup", ["Yes","No","No internet service"])
device = st.selectbox("Device Protection", ["Yes","No","No internet service"])
support = st.selectbox("Tech Support", ["Yes","No","No internet service"])
tv = st.selectbox("Streaming TV", ["Yes","No","No internet service"])
movies = st.selectbox("Streaming Movies", ["Yes","No","No internet service"])

contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes","No"])
payment = st.selectbox(
    "Payment Method",
    ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"]
)

monthly = st.number_input("Monthly Charges", value=70.0)
total = st.number_input("Total Charges", value=1000.0)

# =====================================
# Prediction
# =====================================
if st.button("Predict Churn"):

    input_df = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone,
        'MultipleLines': multi,
        'InternetService': internet,
        'OnlineSecurity': security,
        'OnlineBackup': backup,
        'DeviceProtection': device,
        'TechSupport': support,
        'StreamingTV': tv,
        'StreamingMovies': movies,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment,
        'MonthlyCharges': monthly,
        'TotalCharges': total
    }])

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("Result")

    if pred == 1:
        st.error(f"Customer likely to churn ⚠️  (Probability: {prob:.2f})")
    else:
        st.success(f"Customer likely to stay ✅  (Probability: {prob:.2f})")
