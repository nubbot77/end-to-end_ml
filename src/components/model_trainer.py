import os   
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts", "model.pkl"
    )

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")

            X_train,y_train,X_test,y_test = (
                train_array[:,:-1], 
                train_array[:,-1], 
                test_array[:,:-1], 
                test_array[:,-1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "XGBoost Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=0),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "KNeighbors Regressor": KNeighborsRegressor()
            }

            model_report: dict = evaluate_model(X_train= X_train,y_train= y_train,X_test= X_test,y_test= y_test,models= models)

            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy", sys)

            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            save_obj(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            r2_sc = r2_score(y_test, predicted)

            return r2_sc

        except Exception as e: 
            raise CustomException(e, sys)
        logging.info("Splitting training and testing input data")