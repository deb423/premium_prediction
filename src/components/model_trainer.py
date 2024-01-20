import os
import sys
import numpy as np
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_numpy_array_data
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "trained_model.pkl")
    preprocessed_train_data_file_path: str = os.path.join("artifacts", "train.npz")
    preprocessed_test_data_file_path: str = os.path.join("artifacts", "test.npz")

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def initiate_model_training(self):
        try:
            # Load preprocessed data
            train_arr = load_numpy_array_data(self.config.preprocessed_train_data_file_path)
            test_arr = load_numpy_array_data(self.config.preprocessed_test_data_file_path)

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            # Define your models here
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "XGBoost Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False)
            }

            best_model = None
            best_r2_score = 0

            # Train and evaluate models
            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                logging.info(f"Model {model_name} R2 score: {r2}")
                
                if r2 > best_r2_score:
                    best_r2_score = r2
                    best_model = model

            # Save the best model
            if best_model:
                save_object(self.config.trained_model_file_path, best_model)
                logging.info(f"Best model (R2 score: {best_r2_score}) saved at {self.config.trained_model_file_path}")
            else:
                raise CustomException("No model trained successfully.")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    config = ModelTrainerConfig()
    model_trainer = ModelTrainer(config)
    model_trainer.initiate_model_training()
