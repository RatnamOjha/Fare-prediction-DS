import os
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        # Example: Add your data transformation logic here
        print(f"Transforming data from {train_path} and {test_path}")
        return train_path, test_path, self.config.preprocessor_obj_file_path
