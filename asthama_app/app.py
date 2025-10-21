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
# Streamlit UI Initialization (important for Hugging Face)
# ======================================================
st.set_page_config(page_title="Asthma Risk Prediction", layout="wide")
st.title("🌿 Asthma Risk Prediction App")

st.write("Initializing model... please wait ⏳")

# ======================================================
# MLflow + DagsHub Setup
# ======================================================
dagshub_token = os.getenv("DAGSHUB_TOKEN")
dagshub_url = "https://dagshub.com"
repo_owner = "samiabdulsami122010"
repo_name = "china_cancer_patient_project"
MODEL_NAME = "my_model"

model = None

try:
    if not dagshub_token:
        st.warning("⚠️ `DAGSHUB_TOKEN` not found — falling back to local model.")
        raise EnvironmentError("DAGSHUB_TOKEN missing")

    # Set credentials for MLflow (DagsHub)
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
    st.info(f"✅ MLflow Tracking URI set to: {mlflow.get_tracking_uri()}")

    # Load latest model version from MLflow registry
    client = mlflow.MlflowClient()
    latest = client.get_latest_versions(MODEL_NAME, stages=["Staging"]) or client.get_latest_versions(MODEL_NAME, stages=["None"])

    if latest:
        model_version = latest[0].version
        model_uri = f"models:/{MODEL_NAME}/{model_version}"
        st.write(f"📦 Loading model from MLflow registry: `{model_uri}` ...")
        model = mlflow.pyfunc.load_model(model_uri)
        st.success("✅ Model loaded successfully from MLflow!")
    else:
        raise ValueError("No model version found in MLflow registry.")

except Exception as e:
    st.warning(f"⚠️ Could not load model from DagsHub MLflow.\nReason: {e}\n➡️ Loading local fallback model...")

    # Try loading local model
    if os.path.exists("model"):
        model = mlflow.pyfunc.load_model("model")
        st.success("✅ Loaded local model successfully.")
    else:
        st.error("❌ No local model found! Please include a `model/` folder in your repo.")
        st.stop()

# ======================================================
# Metrics (simple counters)
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
# Prediction Form
# ======================================================
st.subheader("🩺 Enter Patient Details")

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0)
    family_history = st.selectbox("Family History", [0, 1])
    air_pollution = st.selectbox("Air Pollution Level", ["Low", "Moderate", "High"])
    physical_activity = st.selectbox("Physical Activity Level", ["Sedentary", "Moderate", "Active"])
    occupation = st.selectbox("Occupation Type", ["Indoor", "Outdoor"])
    medication_adherence = st.selectbox("Medication Adherence", [0, 1])
    er_visits = st.number_input("Number of ER Visits", min_value=0)
    peak_flow = st.number_input("Peak Expiratory Flow", min_value=0.0)
    feno = st.number_input("FeNO Level", min_value=0.0)
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    smoking = st.selectbox("Smoking Status", ["Current", "Former", "Never"])
    allergies = st.selectbox("Allergies", ["Dust", "Multiple", "Pets", "Pollen"])
    comorbidities = st.selectbox("Comorbidities", ["Both", "Diabetes", "Hypertension"])
    submitted = st.form_submit_button("🔍 Predict")

# ======================================================
# Prediction Logic
# ======================================================
if submitted:
    REQUEST_COUNT += 1
    start_time = time.time()

    # Mapping
    air_pollution_map = {"Low": 0, "Moderate": 1, "High": 2}
    physical_activity_map = {"Sedentary": 0, "Moderate": 1, "Active": 2}
    occupation_map = {"Indoor": 0, "Outdoor": 1}

    # Prepare input data
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
        "Gender_Female": 1 if gender == "Female" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
        "Gender_Other": 1 if gender == "Other" else 0,
        "Smoking_Status_Current": 1 if smoking == "Current" else 0,
        "Smoking_Status_Former": 1 if smoking == "Former" else 0,
        "Smoking_Status_Never": 1 if smoking == "Never" else 0,
        "Allergies_Dust": 1 if allergies == "Dust" else 0,
        "Allergies_Multiple": 1 if allergies == "Multiple" else 0,
        "Allergies_Pets": 1 if allergies == "Pets" else 0,
        "Allergies_Pollen": 1 if allergies == "Pollen" else 0,
        "Comorbidities_Both": 1 if comorbidities == "Both" else 0,
        "Comorbidities_Diabetes": 1 if comorbidities == "Diabetes" else 0,
        "Comorbidities_Hypertension": 1 if comorbidities == "Hypertension" else 0
    }])

    data = data.reindex(columns=EXPECTED_COLUMNS, fill_value=0)

    try:
        prediction = model.predict(data)[0]
        result_text = "✅ No Asthma" if prediction == 0 else "😷 Has Asthma"
        PREDICTION_COUNT[str(prediction)] += 1
        latency = time.time() - start_time

        st.success(f"Prediction: **{result_text}**")
        st.write(f"⏱️ Processed in {latency:.2f} seconds")
        st.write(f"📊 Total Requests: {REQUEST_COUNT}")
        st.write(f"🧩 Prediction counts: {dict(PREDICTION_COUNT)}")

    except Exception as e:
        st.error(f"🚨 Error during prediction: {e}")
