import os
import sys
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml,load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class DataProcessor:

    def __init__(self, train_path, test_path, processed_dir,config_path):

        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config = read_yaml(config_path)
        
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def preprocess_data(self,df):
        try:
            logger.info("Starting data preprocessing")
            
            logger.info("dropping the columns")
            df.drop(columns=['Booking_ID'], inplace=True)
            df.drop_duplicates(inplace=True)

            cat_cols = self.config['data_processing']['categorical_columns']
            num_cols = self.config['data_processing']['numerical_columns']

            logger.info("Encoding categorical columns")

            labelEncoder = LabelEncoder()
            mapping = {}

            for col in cat_cols:
                df[col] = labelEncoder.fit_transform(df[col])
                mapping[col] = {label:Code for Code, label in zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_))}

            logger.info("Label Mapping:")
            for col, mapping in mapping.items():
                logger.info(f"{col}: {mapping}")

            logger.info("doing skewness handing")

            skew_threshold = self.config['data_processing']['skewness_threshold']

            skewness = df[num_cols].apply(lambda x: x.skew())

            for column in skewness[skewness > skew_threshold].index:
                df[column] = np.log1p(df[column])

            return df
        
        except Exception as e:
            logger.error(f"Error during preprocessing data: {e}")
            raise CustomException(f"Error while preprocessing data: {e}",sys)
        
    def balanced_data(self, df):
        try:
            logger.info("Balancing data using SMOTE")
            x = df.drop(columns=['booking_status'])
            y = df['booking_status']

            smote  = SMOTE(random_state=42)
            x_resampled , y_resampled = smote.fit_resample(x, y)

            balanced_df = pd.concat([x_resampled, y_resampled], axis=1)

            logger.info("Data balanced successfully")
            return balanced_df
        
        except Exception as e:
            logger.error(f"Error during balancing data: {e}")
            raise CustomException(f"Error while balancing data: {e}",sys)
        

    def select_features(self, df):
        try:
            logger.info("Selecting features selection step")
            x = df.drop(columns=['booking_status'])
            y = df['booking_status']
             
            model = RandomForestClassifier(random_state=42)
            model.fit(x, y)
            feature_importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({'feature': x.columns, 'importance': feature_importance})
            top_feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

            num_features_to_select = self.config['data_processing']['no_of_features']
            
            top_10_features = top_feature_importance_df["feature"].head(num_features_to_select).values
            logger.info(f"Top {num_features_to_select} features selected: {top_10_features}")

            top_10_df = df[top_10_features.tolist() + ['booking_status']]

            logger.info("Feature selection completed successfully")

            return top_10_df
        
        except Exception as e:
            logger.error(f"Error during feature selection: {e}")
            raise CustomException(f"Error while selecting features: {e}",sys)
        

    def save_data(self, df, file_path):
        try:
            logger.info(f"Saving processed data to {file_path}")
            df.to_csv(file_path, index=False)
            logger.info("Data saved successfully")

        except Exception as e:
            logger.error(f"Error saving data to {file_path}: {e}")
            raise CustomException(f"Error while saving data: {e}",sys)
        

    def process(self):
        try:
            logger.info("loading  data from raw directory")

            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            train_df = self.balanced_data(train_df)
            test_df = self.balanced_data(test_df)

            train_df = self.select_features(train_df)
            test_df = test_df[train_df.columns]

            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df, PROCESSED_TEST_DATA_PATH)

            logger.info("Data processing completed successfully")

        except Exception as e:
            logger.error(f"Error during data processing pipeline: {e}")
            raise CustomException(f"Error while processing data pipeline: {e}",sys)
        

if __name__ == "__main__":

    # Use the correct source data paths
    TRAIN_DATA_PATH = "artifacts/raw/train_data.csv"
    TEST_DATA_PATH = "artifacts/raw/test_data.csv"
    PROCESSED_DIR = "artifacts/processed"
    CONFIG_PATH = "config/config.yaml"
    
    processor = DataProcessor(TRAIN_DATA_PATH, TEST_DATA_PATH, PROCESSED_DIR, CONFIG_PATH)
    processor.process()