import os
import sys
import dill
from sklearn.metrics import r2_score

import numpy as np
import pandas as pd


from src.exceptions import CustomException
from sklearn.model_selection import GridSearchCV
from src.logger import logging

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
        

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            model_params = param.get(model_name, {})  # Safe get

            if model_params:  # If we have params, use GridSearchCV
                gs_gridsearch = GridSearchCV(model, model_params, cv=3, n_jobs=-1, verbose=2)
                gs_gridsearch.fit(X_train, y_train)
                best_model = gs_gridsearch.best_estimator_
                logging.info(f"Best parameters for {model_name}: {gs_gridsearch.best_params_}")
            else:  # Otherwise, use default model
                model.fit(X_train, y_train)
                best_model = model
                logging.info(f"No hyperparameters tuned for {model_name}, using default.")
                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)
                train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            logging.info(f"{model_name} train score: {train_model_score:.4f}, test score: {test_model_score:.4f}")
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
        logging.info("Model evaluation completed")

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)