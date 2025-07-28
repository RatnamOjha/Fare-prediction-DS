import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from numpy import e
import pandas as pd
import logging
from src.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exceptions import CustomException
from src.components.data_transform import DataTransformationConfig
from src.components.data_transform import DataTransformation

from src.components.model_training import ModelTrainerConfig
from src.components.model_training import ModelTrainer


@dataclass
class DataIngestionConfig:
    """
    Configuration for data ingestion.
    """
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
class DataIngestion:
    """
    Class for data ingestion.
    """
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Initiates the data ingestion process.
        """
        logging.info("Data Ingestion started")
        try:
            df = pd.read_csv("/Users/ratnamb.ojha/Downloads/uber.csv.zip")
            logging.info("Read the dataset DataFrame")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved at {self.ingestion_config.raw_data_path}")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Train and Test split completed")

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(f"Train data saved at {self.ingestion_config.train_data_path}")
            logging.info(f"Test data saved at {self.ingestion_config.test_data_path}")

            logging.info("Data Ingestion completed successfully")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )
        except Exception as e:
            raise CustomException(e, sys) from e
        

if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.initiate_data_ingestion()