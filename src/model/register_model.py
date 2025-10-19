import mlflow
import os
import logging

MODEL_NAME = "my_model_v2"

def main():
    try:
        dagshub_uri = os.getenv("DAGSHUB_TRACKING_URI")
        mlflow.set_tracking_uri(dagshub_uri)
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_API_TOKEN")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_API_TOKEN")

        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("Default")  # or your custom name
        runs = client.search_runs(experiment_ids=[experiment.experiment_id],
                                  order_by=["attributes.start_time DESC"],
                                  max_results=1)
        latest_run = runs[0]
        run_id = latest_run.info.run_id

        model_uri = f"runs:/{run_id}/model"
        model_version = mlflow.register_model(model_uri, MODEL_NAME)

        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=model_version.version,
            stage="Staging"
        )

        print(f"✅ Model {MODEL_NAME} version {model_version.version} registered and transitioned to Staging.")

    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
