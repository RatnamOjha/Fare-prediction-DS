import logging
import os
from datetime import datetime


LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"

logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(os.path.dirname(logs_path), exist_ok=True)

LOG_FILE_PATH = logs_path  # Use logs_path directly

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    filemode='w'
)

if __name__ == "__main__":
    logging.info("Logging has been set up successfully.")
    logging.warning("This is a warning message.")
    logging.error("This is an error message.")
    logging.debug("This is a debug message.")




#An f-string (formatted string literal) is a feature in Python/
# that allows you to embed expressions inside string literals using curly braces {}
#.strftime('%Y-%m-%d'):Formats the date into a string in the format YYYY-MM-DD
