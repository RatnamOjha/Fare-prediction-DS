import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from logger import logging  # Import logging from logger.py

def error_message_detail(error, error_detail: sys):
    """
    This function is used to format the error message with details.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in script: {file_name} at line number: {exc_tb.tb_lineno}, error message: {str(error)}"
    return error_message
 

class CustomException(Exception):
    """
    Custom exception class to handle exceptions with detailed error messages.
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message