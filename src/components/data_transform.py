import sys
import os
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exceptions import CustomException
from src.logger import setup_logging
import logging

from src.utils import save_object

# Set up logging and get module-specific logger
setup_logging()
logger = logging.getLogger(__name__)

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation.
        '''
        try:
            numerical_features = [
                'passenger_count', 'distance', 'is rush hour',
                'is a group trip', 'fare_per_km',
                'fare_per_passenger', 'passenger_count_log'
            ]
            categorical_features = []

            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehotencoder', OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_pipeline, numerical_features),
                    ('cat', categorical_pipeline, categorical_features)
                ]
            )

            logger.info("Preprocessing pipeline created successfully.")
            return preprocessor
        except Exception as e:
            logger.error(f"Error occurred in get_data_transformer_object: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        logger.info("Starting data transformation script.")
        transformer = DataTransformation()
        preprocessing_obj = transformer.get_data_transformer_object()
        logger.info("Preprocessing object initialized in __main__.")
    except Exception as e:
        logger.exception("Exception occurred in __main__.")
        print(f"Error occurred: {e}")

