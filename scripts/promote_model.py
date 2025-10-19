import os
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "my_model_v2"

def promote_model():
    # ✅ Use the environment variable, not the string
    mlflow.set_tracking_uri(os.getenv("DAGSHUB_TRACKING_URI"))

    client = MlflowClient()

    model_name = MODEL_NAME
    # ✅ Check Staging models first
    versions_staging = client.get_latest_versions(model_name, stages=["Staging"])

    if not versions_staging:
        raise ValueError(f"No versions of '{model_name}' found in 'Staging' stage. Register step may not have transitioned it.")

    latest_version_staging = versions_staging[0].version

    # Archive current production model
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
