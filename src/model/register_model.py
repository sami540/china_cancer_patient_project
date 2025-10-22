import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import mlflow
import mlflow.sklearn
import os
from mlflow.models.signature import infer_signature

# ======================================================
# Custom Logging Setup
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
    raise EnvironmentError("❌ DAGSHUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "samiabdulsami122010"
repo_name = "china_cancer_patient_project"

mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
logger.info("✅ MLflow Tracking URI set to: %s", mlflow.get_tracking_uri())


# ======================================================
# Utility Functions
# ======================================================
def get_latest_model_version(model_name: str):
    """Fetch the latest model version from MLflow registry."""
    try:
        client = mlflow.MlflowClient()
        latest_versions = client.get_latest_versions(model_name, stages=["Production"])
        if not latest_versions:
            latest_versions = client.get_latest_versions(model_name, stages=["None"])
        return latest_versions[0].version if latest_versions else None
    except Exception as e:
        logger.error(f"Failed to fetch model version from MLflow: {e}")
        return None


def load_model(file_path: str):
    """Load a model from local file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.info(f"✅ Model loaded from {file_path}")
        return model
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading model: {e}")
        raise


def get_model(model_name: str):
    """Try loading model from DagsHub; fallback to emergency local model."""
    try:
        model_version = get_latest_model_version(model_name)
        if not model_version:
            raise ValueError("No model version found in MLflow registry.")

        model_uri = f"models:/{model_name}/{model_version}"
        logger.info(f"🔄 Loading model from DagsHub: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("✅ Model loaded successfully from DagsHub.")
        return model, True   # True → loaded from MLflow
    except Exception as e:
        logger.error(f"⚠️ Failed to load model from DagsHub: {e}")
        emergency_path = './asthama_app/emergency_model/model.pkl'
        logger.info(f"🔁 Loading emergency backup model from {emergency_path}")
        model = load_model(emergency_path)
        return model, False  # False → loaded from emergency model


def load_data(x_path: str, y_path: str):
    """Load test features and labels."""
    try:
        logger.info("📥 Loading data for testing ...")
        x_test = pd.read_csv(x_path)
        y_test = pd.read_csv(y_path).squeeze()
        return x_test, y_test
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def model_evaluation(model, x_test, y_test):
    """Evaluate model and return metrics."""
    try:
        logger.info("🔍 Evaluating model ...")
        y_pred = model.predict(x_test)
        metrics_dict = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0)
        }
        logger.info("📊 Model evaluation completed successfully.")
        return metrics_dict
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise


def save_metrics(metrics: dict, file_path: str):
    """Save evaluation metrics to JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.info(f"📁 Metrics saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving metrics: {e}")
        raise


def save_model_info(run_id: str, model_path: str, file_path: str):
    """Save model run ID and path to JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        model_info = {"run_id": run_id, "model_path": model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.info(f"🧾 Model info saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving model info: {e}")
        raise


# ======================================================
# Main Function
# ======================================================
def main():
    mlflow.set_experiment("pipeline_2")

    with mlflow.start_run() as run:
        try:
            logger.info("🚀 Starting model evaluation process...")

            # Try loading from MLflow first
            model_name = "china_cancer_model"
            clf, loaded_from_mlflow = get_model(model_name)

            # Load test data
            x_test, y_test = load_data('./splited_data/x_test.csv', './splited_data/y_test.csv')

            # Evaluate
            metrics = model_evaluation(clf, x_test, y_test)
            save_metrics(metrics, './reports/metrics.json')

            # Always log metrics (safe)
            for name, value in metrics.items():
                mlflow.log_metric(name, value)

            if loaded_from_mlflow:
                # ✅ Only register / log model to MLflow if loaded from MLflow registry
                signature = infer_signature(x_test.head(10), clf.predict(x_test.head(10)))
                mlflow.sklearn.log_model(
                    clf,
                    artifact_path="model",
                    signature=signature,
                    input_example=x_test.head(5)
                )
                save_model_info(run.info.run_id, f"runs:/{run.info.run_id}/model", './reports/experiment_info.json')
                mlflow.log_artifact('./reports/metrics.json')
                logger.info("✅ Model evaluation + registration completed (MLflow).")
            else:
                # 🚫 Skip MLflow model registration
                logger.warning("⚠️ Skipping model registration — using emergency backup model only.")
                save_model_info(run.info.run_id, "./asthama_app/emergency_model/model.pkl", './reports/experiment_info.json')

        except Exception as e:
            logger.error(f"❌ Model evaluation failed: {e}")
            raise


# ======================================================
# Entry Point
# ======================================================
if __name__ == "__main__":
    main()
