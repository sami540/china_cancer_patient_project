# data ingestion
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import os
from sklearn.model_selection import train_test_split
import yaml
from src.logger import logging
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# def load_params(params_path: str) -> dict:
#     """Load parameters from a YAML file."""
#     try:
#         with open(params_path, 'r') as file:
#             params = yaml.safe_load(file)
#         logging.debug('Parameters retrieved from %s', params_path)
#         return params
#     except FileNotFoundError:
#         logging.error('File not found: %s', params_path)
#         raise
#     except yaml.YAMLError as e:
#         logging.error('YAML error: %s', e)
#         raise
#     except Exception as e:
#         logging.error('Unexpected error: %s', e)
#         raise

def load_data(data_url):
    try:
        logging.info('loading data ........')
        df = pd.read_csv(data_url)
        df.drop(columns=['Patient_ID', 'Asthma_Control_Level'], inplace=True)
        logging.info('dataloading completed !!')
        return df
    except Exception as e:
        logging.error(f'data ingestion gave error as {e}')
        raise


def preprocessing_first(df):
    try:
        logging.info('preprocessing ..........')
        df['Allergies'] = df['Allergies'].fillna(df['Allergies'].mode()[0])
        df['Comorbidities'] = df['Comorbidities'].fillna(df['Comorbidities'].mode()[0])

        return df
    except Exception as e:
        logging.error(f'The error is {e}')
        raise


def doing_onehotencoding(df):
    try:
        logging.info('doing one hot encoding .........')
        cols = ['Gender', 'Smoking_Status', 'Allergies', 'Comorbidities']
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # <-- fix here

        for col in cols:
            if col in df.columns:
                encoded = encoder.fit_transform(df[[col]])
                encoded_cols = encoder.get_feature_names_out([col])
                encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
                df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)

        return df
    except Exception as e:
        logging.error(f'The error is {e}')
        raise



def doing_ordinalencoding(df): 
    try:
        logging.info("Starting manual ordinal encoding...")

        ordinal_maps = {
            'Air_Pollution_Level': {'Low': 0, 'Moderate': 1, 'High': 2},
            'Physical_Activity_Level': {'Sedentary': 0, 'Moderate': 1, 'Active': 2},
            'Occupation_Type': {'Indoor': 0, 'Outdoor': 1}
        }

        for col, mapping in ordinal_maps.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
                logging.info(f"Encoded column: {col}")
            else:
                logging.warning(f"Column '{col}' not found in DataFrame. Skipping.")

        logging.info("Manual ordinal encoding completed successfully.")
        return df

    except Exception as e:
        logging.error(f"Error during manual ordinal encoding: {e}", exc_info=True)
        raise


def preprocessing(df):
    try:
        df = preprocessing_first(df)
        df = doing_onehotencoding(df)
        df = doing_ordinalencoding(df)
        logging.info('preprocessing completed !!!!')
        return df
    except Exception as e:
        logging.error(f'The error is {e}')
        raise


def save_data(df, data_path):
    try:
        logging.info('saving data ..............')
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        df.to_csv(os.path.join(raw_data_path, "preprocessed_data.csv"), index=False)
        logging.info(f'Data saved successfully at {raw_data_path}')
    except Exception as e:
        logging.error(f'Unexpected error occurred while saving the data: {e}')
        raise


def main():
    # df = load_data(r'C:\Users\sfed\Desktop\my-proj\china_cancer_patient_project\data2\raw\synthetic_asthma_dataset.csv')
    df = load_data(r'https://raw.githubusercontent.com/sami540/china_cancer_patient_project/main/data2/raw/synthetic_asthma_dataset.csv')
    df = preprocessing(df)
    save_data(df, './data')


if __name__ == '__main__':
    main()
