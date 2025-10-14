import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import mlflow
import mlflow.sklearn
import os
from src.logger import logging



mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))

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

def load_data(file_path1, file_path2) -> pd.DataFrame:
        try:
         logging.info('Loading data for testing .....')
         x_test = pd.read_csv(file_path1)
         y_test = pd.read_csv(file_path2)
         return x_test, y_test
        except Exception as e:
            logging.error(f'The error is {e}')
            raise
def model_evaluation(model, x_test, y_test):
    try:
     logging.info('evaluating model .......')
     y_pred = model.predict(x_test)

     accuracy = accuracy_score(y_test, y_pred)
     precision = precision_score(y_test, y_pred)
     recall = recall_score(y_test, y_pred)
     f1 = f1_score(y_test, y_pred)

     metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
     logging.info('Model evaluation metrics calculated')
     return metrics_dict
    except Exception as e:
        logging.error(f'the error is {e}')


def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the metrics: %s', e)
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.debug('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model info: %s', e)
        raise

def main():
    mlflow.set_experiment("local_ci_experiment")
    with mlflow.start_run() as run:
        try:
            logging.info('start evaluation')
            clf = load_model('./models/model.pkl')              
            x_test, y_test = load_data('./splited_data/x_test.csv', './splited_data/y_test.csv')            
            # x_test, y_test = load_data(r'C:\Users\sfed\Desktop\my-proj\china_cancer_patient_project\splited_data\x_test.csv', r'C:\Users\sfed\Desktop\my-proj\china_cancer_patient_project\splited_data\y_test.csv')
            
           

            metrics = model_evaluation(clf, x_test, y_test)
            
            save_metrics(metrics, 'reports/metrics.json')
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model parameters to MLflow
            if hasattr(clf, 'get_params'):
                params = clf.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(clf, "model")
            
            # Save model info
            save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')
            
            # Log the metrics file to MLflow
            mlflow.log_artifact('reports/metrics.json')

        except Exception as e:
            logging.error('Failed to complete the model evaluation process: %s', e)
            print(f"Error: {e}")

if __name__ == '__main__':
    main()
