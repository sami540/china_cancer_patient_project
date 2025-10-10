import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import yaml
from src.logger import logging
import os


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

def main():
    try:

        df = load_data(r'C:\Users\sfed\Desktop\my-proj\china_cancer_patient_project\data\raw\preprocessed_2_data.csv')
        x_train, x_test, y_train, y_test = split_data(df, 0.2, 42)
        clf = training_model(x_train, y_train)
        save_model(clf, 'models/model.pkl')
        logging.info('model saved successufullly !')
    except Exception as e:
        logging.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()