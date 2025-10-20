import os
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "my_model"

def promote_model():
    # Load DagsHub token
    dagshub_token = os.getenv("DAGSHUB_TOKEN")  # You can rename CAPSTONE_TEST → DAGSHUB_TOKEN for clarity
    if not dagshub_token:
        raise EnvironmentError("❌ DAGSHUB_TOKEN environment variable is not set")

    # Set DagsHub authentication for MLflow
    os.environ["MLFLOW_TRACKING_USERNAME"] = "samiabdulsami122010"  # your DagsHub username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    # Define your DagsHub repo info
    dagshub_url = "https://dagshub.com"
    repo_owner = "samiabdulsami122010"
    repo_name = "china_cancer_patient_project"

    # Set the MLflow tracking URI for DagsHub
    mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

    # Optional: print to verify connection
    print("✅ MLflow Tracking URI set to:", mlflow.get_tracking_uri())

    # Initialize MLflow client
    client = MlflowClient()

    model_name = MODEL_NAME

    # ✅ Check if a Staging model exists
    versions_staging = client.get_latest_versions(model_name, stages=["Staging"])

    if not versions_staging:
        raise ValueError(f"No versions of '{model_name}' found in 'Staging' stage. Register step may not have transitioned it.")

    latest_version_staging = versions_staging[0].version

    # Archive current Production model(s)
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for version in prod_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived"
        )

    # Promote new one
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version_staging,
        stage="Production"
    )
    print(f"✅ Model version {latest_version_staging} promoted to Production!")


if __name__ == "__main__":
    promote_model()
