import os
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

# ---- page setup ----
st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="🌍")
st.title("Wellness Tourism Package — Purchase Predictor")
st.write(
    "Enter a customer's details below. The model predicts whether they are "
    "likely to buy the Wellness Tourism Package, so the sales team can focus "
    "its follow-ups where they actually convert."
)

MODEL_REPO = "iamanshkataria/tourism-package-model"

# ---- load the registered model + the column order it was trained on ----
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename="best_model.joblib")
    cols_path = hf_hub_download(repo_id=MODEL_REPO, filename="model_columns.joblib")
    return joblib.load(model_path), joblib.load(cols_path)

model, model_columns = load_model()

# ---- input form ----
st.subheader("Customer details")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 35)
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration_of_pitch = st.number_input("Duration of Pitch (min)", 0.0, 60.0, 15.0)
    occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    num_person_visiting = st.number_input("Number of Persons Visiting", 1, 10, 3)
    num_followups = st.number_input("Number of Follow-ups", 0, 10, 3)
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])

with col2:
    preferred_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    num_trips = st.number_input("Number of Trips per year", 0.0, 30.0, 3.0)
    passport = st.selectbox("Holds Passport", [0, 1])
    pitch_satisfaction = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
    own_car = st.selectbox("Owns a Car", [0, 1])
    num_children = st.number_input("Number of Children Visiting", 0.0, 5.0, 0.0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", 1000.0, 100000.0, 20000.0)

# ---- assemble one row in the raw schema, then encode to match training ----
if st.button("Predict"):
    row = pd.DataFrame([{
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": num_person_visiting,
        "NumberOfFollowups": num_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": num_trips,
        "Passport": passport,
        "PitchSatisfactionScore": pitch_satisfaction,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": num_children,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
    }])

    row_encoded = pd.get_dummies(row, drop_first=True)
    row_encoded = row_encoded.reindex(columns=model_columns, fill_value=0)

    pred = model.predict(row_encoded)[0]
    proba = model.predict_proba(row_encoded)[0][1]

    if pred == 1:
        st.success(f"Likely to purchase ✅  (probability {proba:.1%})")
    else:
        st.info(f"Unlikely to purchase ❌  (probability {proba:.1%})")
