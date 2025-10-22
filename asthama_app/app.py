import gradio as gr
import mlflow
import pandas as pd
import pickle
import os
import logging
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
# MLflow + DagsHub Setup
# ======================================================
dagshub_token = os.getenv("DAGSHUB_TOKEN")
dagshub_url = "https://dagshub.com"
repo_owner = "samiabdulsami122010"
repo_name = "china_cancer_patient_project"
MODEL_NAME = "my_model"
emergency_path = "./emergency_model/model.pkl"

if dagshub_token:
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
else:
    logger.warning("⚠️ DAGSHUB_TOKEN not found. Using local emergency model.")

# ======================================================
# Model Loading
# ======================================================
def get_latest_model_version(model_name: str):
    try:
        client = mlflow.MlflowClient()
        latest = client.get_latest_versions(model_name, stages=["Production"])
        if not latest:
            latest = client.get_latest_versions(model_name, stages=["None"])
        return latest[0].version if latest else None
    except Exception as e:
        logger.error(f"Error fetching model version: {e}")
        return None

def load_local_model(path: str):
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info("✅ Local model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load local model: {e}")
        raise

def get_model(model_name: str):
    try:
        version = get_latest_model_version(model_name)
        if not version:
            raise ValueError("No model version found in MLflow registry.")
        uri = f"models:/{model_name}/{version}"
        logger.info(f"🔄 Loading model from DagsHub: {uri}")
        model = mlflow.pyfunc.load_model(uri)
        logger.info("✅ Model loaded from DagsHub.")
        return model
    except Exception as e:
        logger.error(f"⚠️ Failed to load model from DagsHub: {e}")
        logger.info("🔁 Loading local fallback model...")
        return load_local_model(emergency_path)

# Load model (try DagsHub → fallback)
model = get_model(MODEL_NAME)

# ======================================================
# Prediction Logic
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

def predict_asthma(Age, BMI, Family_History, Air_Pollution_Level,
                   Physical_Activity_Level, Occupation_Type,
                   Medication_Adherence, Number_of_ER_Visits,
                   Peak_Expiratory_Flow, FeNO_Level,
                   Gender, Smoking_Status, Allergies, Comorbidities):
    try:
        air_pollution_map = {"Low": 0, "Moderate": 1, "High": 2}
        physical_activity_map = {"Sedentary": 0, "Moderate": 1, "Active": 2}
        occupation_type_map = {"Indoor": 0, "Outdoor": 1}

        data = pd.DataFrame([{
            "Age": float(Age),
            "BMI": float(BMI),
            "Family_History": int(Family_History),
            "Air_Pollution_Level": air_pollution_map[Air_Pollution_Level],
            "Physical_Activity_Level": physical_activity_map[Physical_Activity_Level],
            "Occupation_Type": occupation_type_map[Occupation_Type],
            "Medication_Adherence": int(Medication_Adherence),
            "Number_of_ER_Visits": int(Number_of_ER_Visits),
            "Peak_Expiratory_Flow": float(Peak_Expiratory_Flow),
            "FeNO_Level": float(FeNO_Level),
            "Gender": Gender,
            "Smoking_Status": Smoking_Status,
            "Allergies": Allergies,
            "Comorbidities": Comorbidities
        }])

        data = pd.get_dummies(data)
        data = data.reindex(columns=EXPECTED_COLUMNS, fill_value=0)

        pred = model.predict(data)[0]
        return "✅ No Asthma" if pred == 0 else "😷 Has Asthma"
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return f"❌ Prediction failed: {e}"

# ======================================================
# Gradio UI
# ======================================================
with gr.Blocks(title="Asthma Detection App") as demo:
    gr.Markdown("## 🫁 Asthma Detection App\nEnter patient details below to predict asthma likelihood.")

    with gr.Row():
        Age = gr.Number(label="Age")
        BMI = gr.Number(label="BMI")
        Family_History = gr.Radio([0, 1], label="Family History (1=Yes, 0=No)")
        Air_Pollution_Level = gr.Dropdown(["Low", "Moderate", "High"], label="Air Pollution Level")
        Physical_Activity_Level = gr.Dropdown(["Sedentary", "Moderate", "Active"], label="Physical Activity Level")
        Occupation_Type = gr.Dropdown(["Indoor", "Outdoor"], label="Occupation Type")
        Medication_Adherence = gr.Radio([0, 1], label="Medication Adherence (1=Yes, 0=No)")
        Number_of_ER_Visits = gr.Number(label="Number of ER Visits")
        Peak_Expiratory_Flow = gr.Number(label="Peak Expiratory Flow")
        FeNO_Level = gr.Number(label="FeNO Level")

    with gr.Row():
        Gender = gr.Dropdown(["Female", "Male", "Other"], label="Gender")
        Smoking_Status = gr.Dropdown(["Current", "Former", "Never"], label="Smoking Status")
        Allergies = gr.Dropdown(["Dust", "Multiple", "Pets", "Pollen"], label="Allergies")
        Comorbidities = gr.Dropdown(["Both", "Diabetes", "Hypertension"], label="Comorbidities")

    result = gr.Textbox(label="Prediction Result", interactive=False)
    submit = gr.Button("🔍 Predict")

    submit.click(
        predict_asthma,
        inputs=[
            Age, BMI, Family_History, Air_Pollution_Level, Physical_Activity_Level,
            Occupation_Type, Medication_Adherence, Number_of_ER_Visits,
            Peak_Expiratory_Flow, FeNO_Level, Gender, Smoking_Status, Allergies, Comorbidities
        ],
        outputs=result
    )

# ======================================================
# Launch for Hugging Face (No port binding)
# ======================================================
if __name__ == "__main__":
    demo.launch()
