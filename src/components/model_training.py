import os
from dataclasses import dataclass
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray):
        try:
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            model = LinearRegression()
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            r2 = r2_score(y_test, predictions)

            os.makedirs(os.path.dirname(self.config.trained_model_path), exist_ok=True)
            joblib.dump(model, self.config.trained_model_path)

            print(f"Model trained and saved at {self.config.trained_model_path}")
            print(f"R² Score: {r2}")

            return r2

        except Exception as e:
            raise Exception(f"Error in model training: {e}")
