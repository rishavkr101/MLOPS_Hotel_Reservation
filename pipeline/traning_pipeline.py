from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessor
from src.model_traning import ModelTraining
from utils.common_functions import read_yaml
from config.paths_config import *
from utils.common_functions import read_yaml



if __name__=="__main__":
    ### 1. DATA INGESTION
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()


    ### 2. DATA processing

    TRAIN_DATA_PATH = "artifacts/raw/train_data.csv"
    TEST_DATA_PATH = "artifacts/raw/test_data.csv"
    PROCESSED_DIR = "artifacts/processed"
    CONFIG_PATH = "config/config.yaml"
    
    processor = DataProcessor(TRAIN_DATA_PATH, TEST_DATA_PATH, PROCESSED_DIR, CONFIG_PATH)
    processor.process()



    ### 3.model Training

    trainer = ModelTraining(
            PROCESSED_TRAIN_DATA_PATH,
            PROCESSED_TEST_DATA_PATH,
            MODEL_OUTPUT_PATH
        )
    trainer.run()


