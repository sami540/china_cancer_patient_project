import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import yaml
from src.logger import logging
import os
import pickle
import logging
import shutil



def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def split_data(df, test_size, random_state):
    try:
        logging.info('Splitting data ......')

        # Split features and target
        x = df.drop(columns=['Has_Asthma'])
        y = df['Has_Asthma']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

        # Ensure the folder exists
        folder_path = os.path.join(os.getcwd(), 'splited_data')
        os.makedirs(folder_path, exist_ok=True)

        # Save the splits to CSV
        x_train.to_csv(os.path.join(folder_path, 'x_train.csv'), index=False)
        x_test.to_csv(os.path.join(folder_path, 'x_test.csv'), index=False)
        y_train.to_csv(os.path.join(folder_path, 'y_train.csv'), index=False)
        y_test.to_csv(os.path.join(folder_path, 'y_test.csv'), index=False)

        logging.info(f'Data splits saved in {folder_path}')

        return x_train, x_test, y_train, y_test

    except Exception as e:
        logging.error(f'The error is {e}')
        raise
        
def training_model(x_train, y_train):
    try:
     logging.info('Training model ...... ')
     clf = RandomForestClassifier(n_estimators=100, random_state=42)
     clf.fit(x_train, y_train)
     return clf
    except Exception as e:
        logging.error('The error is: %s', e)
        raise


def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info('Model saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model: %s', e)
        raise

def save_model2(model, file_path: str) -> None:
    """Delete all files in the target folder and save the trained model to a file."""
    try:
        # Get the folder path from the file path
        folder = os.path.dirname(file_path)
        # If folder exists, clear it
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                file_to_delete = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_to_delete) or os.path.islink(file_to_delete):
                        os.unlink(file_to_delete)  # delete file
                    elif os.path.isdir(file_to_delete):
                        shutil.rmtree(file_to_delete)  # delete folder
                except Exception as e:
                    logging.warning("Failed to delete %s: %s", file_to_delete, e)
        else:
            os.makedirs(folder, exist_ok=True)
        # Save the new model
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info('Model saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:

        df = load_data('./data/interim/preprocessed_data_2.csv')
        # df = load_data(r'https://raw.githubusercontent.com/sami540/china_cancer_patient_project/main/data_for_github/preprocessed_data_2.csv')
        x_train, x_test, y_train, y_test = split_data(df, 0.2, 42)
        x_train, x_test, y_train, y_test = split_data(df, 0.2, 42)
        os.makedirs("models", exist_ok=True)
        clf = training_model(x_train, y_train)
        save_model(clf, './models/model.pkl')
        save_model2(clf, './asthama_app/emergency_model/model.pkl')
        logging.info('model saved successufullly !')
    except Exception as e:
        logging.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()