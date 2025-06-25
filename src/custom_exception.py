import traceback
import sys

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        #if error exists in predefined exception, the we will use that 
        super().__init__(error_message)
        #else we will create a new exception
        self.error_message = self.get_detailed_error_message(error_message, error_detail)
        
    @staticmethod    
    def get_detailed_error_message(error_message, error_detail:sys):
        
       _ , _ , exc_tb = traceback.sys.exc_info()
       file_name = exc_tb.tb_frame.f_code.co_filename
       line_number = exc_tb.tb_lineno

       return f"Error occurred in {file_name} , line {line_number}: {error_message}"
    
    def __str__(self):
        return self.error_message
    