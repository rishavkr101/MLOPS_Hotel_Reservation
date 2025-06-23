import os
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report,precision_score, recall_score, f1_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml, load_data
from scipy.stats import uniform, randint
from config.model_params import LIGHTGBM_PARAMS, RANDOM_SEARCH_PARAMS

import mlflow
import mlflow.sklearn



logger = get_logger(__name__)

class ModelTraining:

    def __init__(self, train_path, test_path, model_output_path ):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path
    
        self.params_dist = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS

    def load_and_split_data(self):
        try:
            logger.info("Loading training and testing data")
            train_df = load_data(self.train_path)

            logger.info("Loading training and testing data")
            test_df = load_data(self.test_path)

            x_train = train_df.drop(columns=['booking_status'])
            y_train = train_df['booking_status']

            x_test = test_df.drop(columns=['booking_status'])
            y_test = test_df['booking_status']

            logger.info("Data loaded and split into features and target variable")
            return x_train, y_train, x_test, y_test
        
        except Exception as e:
            logger.error(f"Error loading and splitting data: {e}")
            raise CustomException(f"Error loading and splitting data: {e}")

    
    def train_lgbm(self, x_train, y_train):
        try:
            logger.info("Starting LightGBM model training")
            lgbm_model = lgb.LGBMClassifier(random_state=self.random_search_params['random_state'])

            logger.info("Performing Randomized Search for hyperparameter tuning")

            random_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.params_dist,
                n_iter=self.random_search_params['n_iter'],
                scoring=self.random_search_params['scoring'],
                cv=self.random_search_params['cv'],
                verbose=self.random_search_params['verbose'],
                random_state=self.random_search_params['random_state'],
                n_jobs=-1
            )

            logger.info("starting our hyperparameter training")

            random_search.fit(x_train, y_train)

            logger.info("hyperparameter tuning completed")

            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_
            
            logger.info(f"Best parameters found: {best_params}")

            return best_lgbm_model
        
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise CustomException(f"Error during model training: {e}") from e

    
    def evaluate_model(self, model, x_test, y_test):
        try:
            logger.info("Evaluating the model")
            y_pred = model.predict(x_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            logger.info(f"Model evaluation metrics - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise CustomException(f"failed during model evaluation: {e}") from e
        
    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            logger.info("Saving the trained model")
            joblib.dump(model, self.model_output_path)
            logger.info(f"Model saved successfully at {self.model_output_path}")
        
        except Exception as e:
            logger.error(f"Error saving the model: {e}")
            raise CustomException(f"failed to save model: {e}") from e
        

    def run(self):
        try:
            with mlflow.start_run():
                logger.info("Starting the model training pipeline")

                logger.info("starting our mlflow experimenation")

                logger.info("Logging the traning and testing dataset to mlflow")

                mlflow.log_artifact(self.train_path, artifact_path="datasets")
                mlflow.log_artifact(self.test_path, artifact_path="datasets")

                x_train, y_train, x_test, y_test = self.load_and_split_data()
                best_lgbm_model = self.train_lgbm(x_train, y_train)
                metrics = self.evaluate_model(best_lgbm_model, x_test, y_test)
                self.save_model(best_lgbm_model)


                logger.info("Logging model into MLFLOW")
                mlflow.log_artifact(self.model_output_path, artifact_path="models")
                
                logger.info("Logging model parameters and metrics to MLFLOW")
                mlflow.log_params(best_lgbm_model.get_params())
                mlflow.log_metrics(metrics)


                logger.info("Model training pipeline completed successfully")

        except Exception as e:
            logger.error(f"Error in model training pipeline: {e}")
            raise CustomException(f"failed in model training pipeline: {e}") from e
        
if __name__ == "__main__":
        
        PROCESSED_TRAIN_DATA_PATH = "artifacts/processed/processed_train.csv"
        PROCESSED_TEST_DATA_PATH = "artifacts/processed/processed_test.csv"
        MODEL_OUTPUT_PATH = "artifacts/model/lgbm_model.pkl"

        trainer = ModelTraining(
            PROCESSED_TRAIN_DATA_PATH,
            PROCESSED_TEST_DATA_PATH,
            MODEL_OUTPUT_PATH
        )
        trainer.run()