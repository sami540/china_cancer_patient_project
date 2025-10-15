import mlflow
import os
def promote_model():
    """
    Promote the latest 'Staging' model version to 'Production'
    in a local MLflow Model Registry.
    """
    # ---------------------------------------------------------------------------------
    # Set up MLflow tracking URI for local environment
    # ---------------------------------------------------------------------------------
    # Make sure this matches the tracking URI used in your app.py and training script
    mlflow.set_tracking_uri("http://localhost:5000")

    client = mlflow.MlflowClient()
    model_name = "my_model_v2"  # must match the name you used in your training script

    # ---------------------------------------------------------------------------------
    # Get the latest version in 'Staging'
    # ---------------------------------------------------------------------------------
    staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
    if not staging_versions:
        raise ValueError(f"No model found in 'Staging' stage for '{model_name}'. Please log and register one first.")

    latest_staging_version = staging_versions[0].version
    print(f"🧪 Found Staging model version: {latest_staging_version}")

    # ---------------------------------------------------------------------------------
    # Archive current 'Production' models (if any)
    # ---------------------------------------------------------------------------------
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    if prod_versions:
        for version in prod_versions:
            client.transition_model_version_stage(
                name=model_name,
                version=version.version,
                stage="Archived"
            )
            print(f"📦 Archived old Production model version: {version.version}")

    # ---------------------------------------------------------------------------------
    # Promote the latest Staging version to 'Production'
    # ---------------------------------------------------------------------------------
    client.transition_model_version_stage(
        name=model_name,
        version=latest_staging_version,
        stage="Production"
    )

    print(f"✅ Model version {latest_staging_version} promoted to 'Production' successfully!")


if __name__ == "__main__":
    promote_model()



# import os
# import mlflow

# def promote_model():
#     # Set up DagsHub credentials for MLflow tracking
#     mlflow.set_tracking_uri("http://localhost:5000")
#     client = mlflow.MlflowClient()

#     model_name = "my_model_v2"
#     # Get the latest version in staging
#     latest_version_staging = client.get_latest_versions(model_name, stages=["Staging"])[0].version

#     # Archive the current production model
#     prod_versions = client.get_latest_versions(model_name, stages=["Production"])
#     for version in prod_versions:
#         client.transition_model_version_stage(
#             name=model_name,
#             version=version.version,
#             stage="Archived"
#         )

#     # Promote the new model to production
#     client.transition_model_version_stage(
#         name=model_name,
#         version=latest_version_staging,
#         stage="Production"
#     )
#     print(f"Model version {latest_version_staging} promoted to Production")

# if __name__ == "__main__":
#     promote_model()    
