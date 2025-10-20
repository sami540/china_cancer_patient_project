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

# Setup logging
from src.logger import logging

# DAGsHub MLflow tracking URI from environment variable (CI-friendly)
mlflow.set_tracking_uri(os.getenv("DAGSHUB_TRACKING_URI"))
MODEL_NAME = "my_model_v2"

def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logging.info('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model: %s', e)
        raise

def load_data(x_path: str, y_path: str):
    """Load test features and labels."""
    try:
        logging.info('Loading data for testing ...')
        x_test = pd.read_csv(x_path)
        y_test = pd.read_csv(y_path)
        return x_test, y_test
    except Exception as e:
        logging.error(f'Error loading data: {e}')
        raise

def model_evaluation(model, x_test, y_test):
    """Evaluate model and return metrics."""
    try:
        logging.info('Evaluating model ...')
        y_pred = model.predict(x_test)

        metrics_dict = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }

        logging.info('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logging.error(f'Error during evaluation: {e}')
        raise

def save_metrics(metrics: dict, file_path: str):
    """Save evaluation metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error saving metrics: %s', e)
        raise

def save_model_info(run_id: str, model_path: str, file_path: str):
    """Save model run ID and path to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.debug('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error saving model info: %s', e)
        raise

def main():
    mlflow.set_experiment("protfolio-pipeline")
    with mlflow.start_run() as run:
        try:
            logging.info('Start evaluation')

            # Load model
            clf = load_model('./models/model.pkl')

            # Load test data
            x_test, y_test = load_data('./splited_data/x_test.csv', './splited_data/y_test.csv')

            # Evaluate model
            metrics = model_evaluation(clf, x_test, y_test)

            # Save metrics locally
            save_metrics(metrics, './reports/metrics.json')

            # Log metrics to MLflow
            for name, value in metrics.items():
                mlflow.log_metric(name, value)

            # Log model parameters to MLflow
            if hasattr(clf, 'get_params'):
                for param_name, param_value in clf.get_params().items():
                    mlflow.log_param(param_name, param_value)

            # Infer model signature to avoid Windows path issues
            signature = infer_signature(x_test, clf.predict(x_test))

            # Log model to MLflow (relative artifact path)
            mlflow.sklearn.log_model(
                clf,
                artifact_path="model",
                signature=signature,
                input_example=x_test.head(5)
            )

            # Save model info locally
            save_model_info(run.info.run_id, "model", './reports/experiment_info.json')

            # Log metrics file as artifact
            mlflow.log_artifact('./reports/metrics.json')

        except Exception as e:
            logging.error('Failed to complete the model evaluation process: %s', e)
            raise

if __name__ == '__main__':
    main()
