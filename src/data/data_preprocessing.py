from src.logger import logging
import pandas as pd
import os
pd.set_option('future.no_silent_downcasting', True)

def data_ingetion(url):
    try:
     logging.info('gathering data for for preprocessing ...........')
     df = pd.read_csv(url)
     return df
    except Exception as e:
       logging.error(f'The error is {e}')
       raise


def remove_outliers_iqr(df, columns):
    try:
     logging.info('removing outliers ......')
     for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= 10) & (df[col] <= upper_bound)]
            return df
    except Exception as e:
       logging.error(f'the error is {e}')      
       raise

def save_data(df, data_path):
    try:
        logging.info('saving data ..............')
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        df.to_csv(os.path.join(raw_data_path, "preprocessed_2_data.csv"), index=False)
        logging.info(f'Data saved successfully at {raw_data_path}')
    except Exception as e:
        logging.error(f'Unexpected error occurred while saving the data: {e}')
        raise    

def main():
   df = data_ingetion(r'C:\Users\sfed\Desktop\my-proj\china_cancer_patient_project\data\raw\preprocessed_data.csv')     
   cols = ['Age', 'BMI', 'Peak_Expiratory_Flow', 'FeNO_Level'] 
   df  = remove_outliers_iqr(df, cols)
   save_data(df, './data') 
   logging.info('data preprocessing completed !!!!')

main()   


    
    