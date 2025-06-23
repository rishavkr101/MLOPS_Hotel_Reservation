import os


############# DATA INGESTION ##############

RAW_DIR = "artifacts/raw"
RAW_FILE_PATH = os.path.join(RAW_DIR, "raw_data.csv")
TRAIN_FILE_PATH = os.path.join(RAW_DIR, "train_data.csv")
TEST_FILE_PATH = os.path.join(RAW_DIR, "test_data.csv")


CONFIG_PATH = "config/config.yaml"


############### Data processing ##############

PROCESSED_DIR = "artifacts/processed"
PROCESSED_TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR, "processed_train.csv") 
PROCESSED_TEST_DATA_PATH = os.path.join(PROCESSED_DIR, "processed_test.csv")


################ Model Training ##############
MODEL_OUTPUT_PATH = "artifacts/model/lgbm_model.pkl"