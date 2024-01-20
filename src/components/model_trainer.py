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
from src.utils import save_object, evaluate_models, load_object
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "trained_model.pkl")
    preprocessed_train_data_file_path: str = os.path.join("artifacts", "train_arr.npz")
    preprocessed_test_data_file_path: str = os.path.join("artifacts", "test_arr.npz")

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def load_preprocessed_data(self, file_path: str):
        try:
            data = np.load(file_path)
            return data['arr_0']
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_training(self):
        try:
            # Load preprocessed data
            train_arr = self.load_preprocessed_data(self.config.preprocessed_train_data_file_path)
            test_arr = self.load_preprocessed_data(self.config.preprocessed_test_data_file_path)

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

            # Define model parameters for GridSearchCV
            params = {
                "Linear Regression": {},
                "Lasso": {'alpha': [0.001, 0.01, 0.1, 1]},
                "Ridge": {'alpha': [0.001, 0.01, 0.1, 1]},
                "K-Neighbors Regressor": {'n_neighbors': [3, 5, 7, 9]},
                "Decision Tree": {'max_depth': [3, 5, 7, 9]},
                "Random Forest": {'n_estimators': [10, 50, 100, 200]},
                "AdaBoost Regressor": {'n_estimators': [50, 100, 200]},
                "XGBoost Regressor": {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1]},
                "CatBoost Regressor": {'depth': [6, 8, 10], 'learning_rate': [0.01, 0.1], 'iterations': [30, 50, 100]}
            }

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            
            # Find the best model based on R2 score
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No model found with acceptable performance")
            logging.info(f"Best model: {best_model_name} with R2 score: {best_model_score}")

            # Load the best model
            best_model = models[best_model_name]

            # Save the best model
            save_object(self.config.trained_model_file_path, best_model)
            logging.info(f"Model saved at {self.config.trained_model_file_path}")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    config = ModelTrainerConfig()
    model_trainer = ModelTrainer(config)
    model_trainer.initiate_model_training()
