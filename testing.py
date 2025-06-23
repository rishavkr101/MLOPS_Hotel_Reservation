from src.logger import get_logger
from src.custom_exception import CustomException
import sys

logger = get_logger(__name__)

def divide_numbers(num1, num2):
    try:
        result = num1 / num2
        logger.info("divide two numbers")
        return result
    except Exception as e:
        logger.error("Error occurred")
        raise CustomException("custom error zero", sys)
        


if __name__ == "__main__":
     
    try:
        logger.info("Starting the division operation")
        divide_numbers(10, 2)
    except CustomException as ce:
        logger.error(str(ce)) 