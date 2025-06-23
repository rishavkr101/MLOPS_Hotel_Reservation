import os 
import sys
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException
import yaml
import pandas as pd


logger = get_logger(__name__)


def test():
    """
    A simple test function to check if the module is working.
    """
    logger.info("Test function executed successfully.")
    return "Test successful!"   

def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"YAML file not found at: {file_path}")
        
        with open(file_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info(f"YAML file read successfully from: {file_path}")
            return config
        
    except Exception as e:
        logger.error(f"Error reading YAML file at {file_path}: {e}")
        raise CustomException(f"Error reading YAML file: {e}", sys)
    

def load_data(path):
    try:
        logger.info(f"Loading data from {path}")
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Error loading data from {path}: {e}")
        raise CustomException(f"failed to load data: {e}", sys)