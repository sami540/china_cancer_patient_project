import os
import mlflow
from mlflow.tracking import MlflowClient
import logging

# ======================================================
# Configuration
# ======================================================
MODEL_NAME = "my_model"
DAGSHUB_URL = "https://dagshub.com"
REPO_OWNER = "samiabdulsami122010"
REPO_NAME = "china_cancer_patient_project"

# ======================================================
# Logging Setup
# ======================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ======================================================
# Promote Model Function
# ======================================================
def promote_model():
    """Promote model from Staging → Production if loaded from MLflow; skip otherwise."""
    try:
        # Load DagsHub token
        dagshub_token = os.getenv("DAGSHUB_TOKEN")
        if not dagshub_token:
            raise EnvironmentError("❌ DAGSHUB_TOKEN environment variable is not set")

        # MLflow Authentication for DagsHub
        os.environ["MLFLOW_TRACKING_USERNAME"] = REPO_OWNER
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        # Set Tracking URI
        mlflow.set_tracking_uri(f"{DAGSHUB_URL}/{REPO_OWNER}/{REPO_NAME}.mlflow")
        logger.info("✅ MLflow Tracking URI set to: %s", mlflow.get_tracking_uri())

        # Initialize MLflow Client
        client = MlflowClient()

        # Check if model exists in Staging
        versions_staging = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
        if not versions_staging:
            logger.warning(f"⚠️ No versions of '{MODEL_NAME}' found in 'Staging' stage — skipping promotion.")
            return

        latest_version_staging = versions_staging[0].version

        # Archive current Production versions
        prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        for version in prod_versions:
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=version.version,
                stage="Archived"
            )
            logger.info(f"📦 Archived previous Production version: {version.version}")

        # Promote the new Staging model to Production
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=latest_version_staging,
            stage="Production"
        )
        logger.info(f"✅ Model version {latest_version_staging} promoted to Production successfully!")

    except Exception as e:
        # Skip gracefully if MLflow unreachable or any error occurs
        logger.error(f"⚠️ Could not promote model from MLflow: {e}")
        logger.warning("⏩ Skipping model promotion step (likely using emergency model).")


# ======================================================
# Entry Point
# ======================================================
if __name__ == "__main__":
    promote_model()
