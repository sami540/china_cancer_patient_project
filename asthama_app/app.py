from flask import Flask, render_template, request
import mlflow
import pandas as pd
import logging
import time
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import warnings
import os
import pickle
import threading
import gradio as gr
import requests

warnings.filterwarnings("ignore")

# ======================================================
# Logging Setup
# ======================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ======================================================
# MLflow + DagsHub Setup
# ======================================================
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    logger.warning("⚠️ DAGSHUB_TOKEN not found — running in fallback mode.")

dagshub_url = "https://dagshub.com"
repo_owner = "samiabdulsami122010"
repo_name = "china_cancer_patient_project"

if dagshub_token:
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
else:
    mlflow.set_tracking_uri("file:///tmp/mlruns")

MODEL_NAME = "my_model"
emergency_path = './emergency_model/model.pkl'

# ======================================================
# Flask App Initialization
# ======================================================
app = Flask(__name__)

# ======================================================
# Prometheus Metrics
# ======================================================
registry = CollectorRegistry()
REQUEST_COUNT = Counter("app_request_count", "Total number of requests", ["method", "endpoint"], registry=registry)
REQUEST_LATENCY = Histogram("app_request_latency_seconds", "Request latency (seconds)", ["endpoint"], registry=registry)
PREDICTION_COUNT = Counter("model_prediction_count", "Number of predictions per class", ["prediction"], registry=registry)

# ======================================================
# Model Loading Functions
# ======================================================
def get_latest_model_version(model_name: str):
    try:
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=["Production"])
        if not latest_version:
            latest_version = client.get_latest_versions(model_name, stages=["None"])
        return latest_version[0].version if latest_version else None
    except Exception as e:
        logger.error(f"Failed to fetch model version from MLflow: {e}")
        return None

def load_model(file_path: str):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.info(f"✅ Model loaded from {file_path}")
        return model
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error occurred while loading the model: {e}")
        return None

def get_model(model_name: str):
    try:
        model_version = get_latest_model_version(model_name)
        if not model_version:
            raise ValueError("No model version found in MLflow registry.")
        model_uri = f"models:/{model_name}/{model_version}"
        logger.info(f"🔄 Loading model from DagsHub: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("✅ Model loaded successfully from DagsHub.")
        return model
    except Exception as e:
        logger.error(f"⚠️ Failed to load model from DagsHub: {e}")
        logger.info(f"🔁 Loading emergency backup model from {emergency_path}")
        return load_model(emergency_path)

# ======================================================
# Load Model at Startup
# ======================================================
model = get_model(MODEL_NAME)
if model is None:
    logger.error("❌ No model available. Please check emergency_model/model.pkl or DagsHub connection.")

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
# Flask Routes
# ======================================================
@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response

@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    try:
        air_pollution_map = {"Low": 0, "Moderate": 1, "High": 2}
        physical_activity_map = {"Sedentary": 0, "Moderate": 1, "Active": 2}
        occupation_type_map = {"Indoor": 0, "Outdoor": 1}

        data = pd.DataFrame([{
            "Age": float(request.form["Age"]),
            "BMI": float(request.form["BMI"]),
            "Family_History": int(request.form["Family_History"]),
            "Air_Pollution_Level": air_pollution_map[request.form["Air_Pollution_Level"]],
            "Physical_Activity_Level": physical_activity_map[request.form["Physical_Activity_Level"]],
            "Occupation_Type": occupation_type_map[request.form["Occupation_Type"]],
            "Medication_Adherence": int(request.form["Medication_Adherence"]),
            "Number_of_ER_Visits": int(request.form["Number_of_ER_Visits"]),
            "Peak_Expiratory_Flow": float(request.form["Peak_Expiratory_Flow"]),
            "FeNO_Level": float(request.form["FeNO_Level"]),
            "Gender": request.form["Gender"],
            "Smoking_Status": request.form["Smoking_Status"],
            "Allergies": request.form["Allergies"],
            "Comorbidities": request.form["Comorbidities"]
        }])

        data = pd.get_dummies(data)
        if "Has_Asthma" in data.columns:
            data = data.drop(columns=["Has_Asthma"])
        data = data.reindex(columns=EXPECTED_COLUMNS, fill_value=0)

        prediction = model.predict(data)[0]
        result = "✅ No Asthma" if prediction == 0 else "😷 Has Asthma"

        PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)
        return render_template("index.html", result=result)

    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        return render_template("index.html", result=f"Error: {str(e)}")

@app.route("/metrics")
def metrics():
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

# ======================================================
# Hugging Face Compatible Launcher (Gradio SDK)
# ======================================================
def run_flask():
    app.run(host="0.0.0.0", port=8000)

def launch_hf_space():
    """Gradio wrapper to launch Flask app for Hugging Face"""
    import threading, time
    thread = threading.Thread(target=run_flask)
    thread.start()
    time.sleep(3)
    try:
        r = requests.get("http://localhost:8000")
        if r.status_code == 200:
            return "✅ Flask app running at / on port 8000"
    except Exception as e:
        return f"⚠️ Flask app started but not reachable: {e}"

with gr.Blocks() as demo:
    gr.Markdown("## 🚀 Flask App Running inside Hugging Face Space (via Gradio SDK)")
    status = gr.Textbox(label="App Status", value="Click below to start Flask app")
    start_btn = gr.Button("Start Flask App")
    start_btn.click(fn=launch_hf_space, outputs=status)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
