import sys
import pandas as pd
import numpy as np
from src.exceptions import CustomException
from src.utils import load_object
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            
            print("Before Loading")
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            print("After Loading")
            
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 passenger_count: int,
                 distance: float,
                 is_rush_hour: int,
                 is_a_group_trip: int,
                 fare_per_km: float,
                 fare_per_passenger: float,
                 passenger_count_log: float):
        
        self.passenger_count = passenger_count
        self.distance = distance
        self.is_rush_hour = is_rush_hour
        self.is_a_group_trip = is_a_group_trip
        self.fare_per_km = fare_per_km
        self.fare_per_passenger = fare_per_passenger
        self.passenger_count_log = passenger_count_log

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "passenger_count": [self.passenger_count],
                "distance": [self.distance],
                "is rush hour": [self.is_rush_hour],  # Note: spaces as in your training data
                "is a group trip": [self.is_a_group_trip],  # Note: spaces as in your training data
                "fare_per_km": [self.fare_per_km],
                "fare_per_passenger": [self.fare_per_passenger],
                "passenger_count_log": [self.passenger_count_log]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

# Example usage:
if __name__ == "__main__":
    # Example prediction
    try:
        # Create sample data
        data = CustomData(
            passenger_count=2,
            distance=5.5,
            is_rush_hour=1,  # 1 for rush hour, 0 for not
            is_a_group_trip=0,  # 1 for group, 0 for not
            fare_per_km=2.5,
            fare_per_passenger=12.5,
            passenger_count_log=0.693  # log(2)
        )
        
        # Convert to DataFrame
        pred_df = data.get_data_as_data_frame()
        print("Input DataFrame:")
        print(pred_df)
        
        # Make prediction
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        print(f"Predicted fare amount: ${results[0]:.2f}")
        
    except Exception as e:
        print(f"Error: {e}")