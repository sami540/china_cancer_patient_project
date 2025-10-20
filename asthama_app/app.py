import streamlit as st
import mlflow
import pandas as pd
import joblib
import time
import os
from collections import Counter
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ======================================================
# MLflow Setup
# ======================================================
dagshub_token = os.getenv("DAGSHUB_TOKEN")  # you can rename CAPSTONE_TEST → DAGSHUB_TOKEN for clarity

if not dagshub_token:
    raise EnvironmentError("❌ DAGSHUB_TOKEN environment variable is not set")

# Set DagsHub authentication for MLflow
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token  # e.g., "wadood123"
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# Define your DagsHub repo info
dagshub_url = "https://dagshub.com"
repo_owner = "samiabdulsami122010"          # e.g., "wadood123"
repo_name = "china_cancer_patient_project"      # your project repo name

# Set the MLflow tracking URI for DagsHub
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

# Optional: print to verify
print("✅ MLflow Tracking URI set to:", mlflow.get_tracking_uri())

MODEL_NAME = "my_model"

def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["Staging"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None

model_version = get_latest_model_version(MODEL_NAME)
if not model_version:
    st.error(f"No model version found for '{MODEL_NAME}' in MLflow registry!")
    st.stop()

model_uri = f"models:/{MODEL_NAME}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)

# ======================================================
# Metrics (simple in-memory counters)
# ======================================================
REQUEST_COUNT = 0
PREDICTION_COUNT = Counter()

# ======================================================
# Expected Columns
# ======================================================
EXPECTED_COLUMNS = [
    "Age","BMI","Family_History","Air_Pollution_Level","Physical_Activity_Level",
    "Occupation_Type","Medication_Adherence","Number_of_ER_Visits",
    "Peak_Expiratory_Flow","FeNO_Level",
    "Gender_Female","Gender_Male","Gender_Other",
    "Smoking_Status_Current","Smoking_Status_Former","Smoking_Status_Never",
    "Allergies_Dust","Allergies_Multiple","Allergies_Pets","Allergies_Pollen",
    "Comorbidities_Both","Comorbidities_Diabetes","Comorbidities_Hypertension"
]

# ======================================================
# Streamlit UI
# ======================================================
st.title("Asthma Risk Prediction")

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0)
    family_history = st.selectbox("Family History", [0,1])
    air_pollution = st.selectbox("Air Pollution Level", ["Low","Moderate","High"])
    physical_activity = st.selectbox("Physical Activity Level", ["Sedentary","Moderate","Active"])
    occupation = st.selectbox("Occupation Type", ["Indoor","Outdoor"])
    medication_adherence = st.selectbox("Medication Adherence", [0,1])
    er_visits = st.number_input("Number of ER Visits", min_value=0)
    peak_flow = st.number_input("Peak Expiratory Flow", min_value=0.0)
    feno = st.number_input("FeNO Level", min_value=0.0)
    gender = st.selectbox("Gender", ["Female","Male","Other"])
    smoking = st.selectbox("Smoking Status", ["Current","Former","Never"])
    allergies = st.selectbox("Allergies", ["Dust","Multiple","Pets","Pollen"])
    comorbidities = st.selectbox("Comorbidities", ["Both","Diabetes","Hypertension"])
    submitted = st.form_submit_button("Predict")

if submitted:
    REQUEST_COUNT += 1
    start_time = time.time()
    
    # Mapping
    air_pollution_map = {"Low": 0, "Moderate": 1, "High": 2}
    physical_activity_map = {"Sedentary": 0, "Moderate": 1, "Active": 2}
    occupation_map = {"Indoor": 0, "Outdoor": 1}
    
    # Prepare input
    data = pd.DataFrame([{
        "Age": age,
        "BMI": bmi,
        "Family_History": family_history,
        "Air_Pollution_Level": air_pollution_map[air_pollution],
        "Physical_Activity_Level": physical_activity_map[physical_activity],
        "Occupation_Type": occupation_map[occupation],
        "Medication_Adherence": medication_adherence,
        "Number_of_ER_Visits": er_visits,
        "Peak_Expiratory_Flow": peak_flow,
        "FeNO_Level": feno,
        "Gender_Female": 1 if gender=="Female" else 0,
        "Gender_Male": 1 if gender=="Male" else 0,
        "Gender_Other": 1 if gender=="Other" else 0,
        "Smoking_Status_Current": 1 if smoking=="Current" else 0,
        "Smoking_Status_Former": 1 if smoking=="Former" else 0,
        "Smoking_Status_Never": 1 if smoking=="Never" else 0,
        "Allergies_Dust": 1 if allergies=="Dust" else 0,
        "Allergies_Multiple": 1 if allergies=="Multiple" else 0,
        "Allergies_Pets": 1 if allergies=="Pets" else 0,
        "Allergies_Pollen": 1 if allergies=="Pollen" else 0,
        "Comorbidities_Both": 1 if comorbidities=="Both" else 0,
        "Comorbidities_Diabetes": 1 if comorbidities=="Diabetes" else 0,
        "Comorbidities_Hypertension": 1 if comorbidities=="Hypertension" else 0
    }])
    
    data = data.reindex(columns=EXPECTED_COLUMNS, fill_value=0)
    
    prediction = model.predict(data)[0]
    result_text = "✅ No Asthma" if prediction==0 else "😷 Has Asthma"
    
    PREDICTION_COUNT[str(prediction)] += 1
    latency = time.time() - start_time
    
    st.write(f"Prediction: {result_text}")
    st.write(f"Request processed in {latency:.2f} seconds")
    st.write(f"Total Requests: {REQUEST_COUNT}")
    st.write(f"Prediction counts: {dict(PREDICTION_COUNT)}")
