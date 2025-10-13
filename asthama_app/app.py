from flask import Flask, render_template, request
import mlflow
import pandas as pd
import logging
import time
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import warnings

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
# MLflow Setup
# ======================================================
mlflow.set_tracking_uri("http://127.0.0.1:5000")
MODEL_NAME = "my_model_v2"

# ======================================================
# Flask App Initialization
# ======================================================
app = Flask(__name__)

# ======================================================
# Prometheus Metrics
# ======================================================
registry = CollectorRegistry()
REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests", ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Request latency (seconds)", ["endpoint"], registry=registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Number of predictions per class", ["prediction"], registry=registry
)

# ======================================================
# Load Model from MLflow Registry
# ======================================================
def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    versions = client.get_latest_versions(model_name, stages=["Staging", "None"])
    return versions[0].version if versions else None

model_version = get_latest_model_version(MODEL_NAME)
if not model_version:
    raise RuntimeError(f"❌ No model version found for '{MODEL_NAME}' in MLflow registry!")

model_uri = f"models:/{MODEL_NAME}/{model_version}"
logger.info(f"🔄 Loading model from: {model_uri}")
model = mlflow.pyfunc.load_model(model_uri)
logger.info("✅ Model loaded successfully.")

# Try to get input schema (optional)
try:
    input_schema = model.metadata.get_input_schema()
    logger.info(f"📊 Model input schema: {[f.name for f in input_schema]}")
except Exception as e:
    logger.warning("⚠️ Could not load model input schema from MLflow.")
    input_schema = None

# ======================================================
# Expected Columns (match training time)
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
# Routes
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
        occupation_type_map = {'Indoor': 0, 'Outdoor': 1}
        
        # Collect form data
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

# Drop any accidental columns named like the target
        if 'Has_Asthma' in data.columns:
          data = data.drop(columns=['Has_Asthma'])

        data = data.reindex(columns=EXPECTED_COLUMNS, fill_value=0)

        # Predict
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
    """Expose Prometheus metrics."""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

# ======================================================
# Main Entry Point
# ======================================================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
