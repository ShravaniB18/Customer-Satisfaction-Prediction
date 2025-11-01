import streamlit as st
import pandas as pd
import joblib
import datetime
from textblob import TextBlob
import numpy as np

# ============================================
# Streamlit Page Configuration
# ============================================
st.set_page_config(page_title="Customer Satisfaction Predictor", page_icon="🤖", layout="centered")
st.title("🤖 Customer Satisfaction Prediction App")
st.markdown("This app uses a fine-tuned **XGBoost model** to predict customer satisfaction based on support ticket details.")

# ============================================
# Load Fine-Tuned Model
# ============================================
@st.cache_resource
def load_model():
    return joblib.load("fine_tuned_xgb_model.pkl")  # Use the fine-tuned model

model = load_model()

# ============================================
# Input Form for Ticket Details
# ============================================
st.subheader("📋 Enter Ticket Details")
col1, col2 = st.columns(2)

with col1:
    customer_age = st.number_input("Customer Age", min_value=10, max_value=100, value=30)
    gender = st.selectbox("Customer Gender", ["Male", "Female", "Other"])
    product = st.text_input("Product Purchased", "LG Smart TV")
    ticket_type = st.selectbox("Ticket Type", ["Technical issue", "Billing inquiry", "Account issue", "General inquiry"])
    ticket_priority = st.selectbox("Ticket Priority", ["Low", "Medium", "High", "Critical"])

with col2:
    channel = st.selectbox("Ticket Channel", ["Email", "Chat", "Social media", "Phone"])
    date_of_purchase = st.date_input("Date of Purchase", datetime.date(2023, 6, 1))
    first_response_time = st.time_input("First Response Time", datetime.time(10, 30))
    time_to_resolution = st.time_input("Time to Resolution", datetime.time(14, 45))
    ticket_status = st.selectbox("Ticket Status", ["Open", "Pending Customer Response", "Closed"])

subject = st.text_input("Ticket Subject", "Product compatibility")
description = st.text_area("Ticket Description", "I'm having an issue with my device not connecting properly.")
resolution = st.text_area("Resolution (if any)", "")

# ============================================
# Feature Engineering
# ============================================
combined_text = f"{subject} {description} {resolution}".strip().lower()
sentiment_polarity = TextBlob(combined_text).sentiment.polarity

first_response_dt = pd.to_datetime(f"{date_of_purchase} {first_response_time}")
time_to_resolution_dt = pd.to_datetime(f"{date_of_purchase} {time_to_resolution}")
resolution_duration_hours = max((time_to_resolution_dt - first_response_dt).total_seconds() / 3600, 0)

purchase_year = pd.to_datetime(date_of_purchase).year
purchase_month = pd.to_datetime(date_of_purchase).month
first_response_hour = first_response_dt.hour
first_response_dow = first_response_dt.dayofweek

# ============================================
# Build Input DataFrame
# ============================================
input_df = pd.DataFrame({
    "Customer Age": [customer_age],
    "Customer Gender": [gender],
    "Product Purchased": [product],
    "Date of Purchase": [pd.to_datetime(date_of_purchase)],
    "Ticket Type": [ticket_type],
    "Ticket Subject": [subject],
    "Ticket Description": [description],
    "Resolution": [resolution],
    "Ticket Status": [ticket_status],
    "Ticket Priority": [ticket_priority],
    "Ticket Channel": [channel],
    "First Response Time": [first_response_dt],
    "Time to Resolution": [time_to_resolution_dt],
    "resolution_duration_hours": [resolution_duration_hours],
    "sentiment_polarity": [sentiment_polarity],
    "purchase_year": [purchase_year],
    "purchase_month": [purchase_month],
    "first_response_hour": [first_response_hour],
    "first_response_dow": [first_response_dow]
})

# Create Ticket_Text (used during training)
input_df["Ticket_Text"] = (
    input_df["Ticket Subject"].astype(str) + " " +
    input_df["Ticket Description"].astype(str) + " " +
    input_df["Resolution"].astype(str)
).str.strip().str.lower()

# ============================================
# Predict Satisfaction
# ============================================
if st.button("🔮 Predict Satisfaction"):
    try:
        st.info("Processing input and generating prediction...")

        # Load preprocessor (must be saved during training)
        preprocessor = joblib.load("preprocessor.pkl")

        # --- Ensure all expected columns exist ---
        expected_cols = set()
        for name, trans, cols in preprocessor.transformers_:
            if isinstance(cols, (list, tuple)):
                expected_cols.update(cols)
            else:
                expected_cols.add(cols)

        for col in expected_cols:
            if col not in input_df.columns:
                if "text" in col.lower():
                    input_df[col] = ""
                else:
                    input_df[col] = 0

        # --- Transform input using same preprocessor ---
        X_transformed = preprocessor.transform(input_df)

        # Convert sparse to dense if required
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        # --- Predict ---
        prediction = model.predict(X_transformed)[0]
        prob = None
        if hasattr(model, "predict_proba"):
            try:
                prob = model.predict_proba(X_transformed).max()
            except Exception:
                prob = None

        rating_map = {0: "Satisfied 😊", 1: "UnSatisfied 😞"}
        output = rating_map.get(int(prediction), prediction)

        # --- Display Result ---
        if prob is not None:
            st.success(f"✅ Predicted Customer Satisfaction: **{output}** (Confidence: {prob:.2f})")
        else:
            st.success(f"✅ Predicted Customer Satisfaction: **{output}**")


        with st.expander("🔍 View Processed Input Data"):
            st.write(input_df)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
