import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    root_dir: str = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    perprocessor_obj_file_path:str = os.path.join(root_dir,'artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = ['gender','race_ethnicity', 'parental_level_of_education','lunch', 'test_preparation_course']

            num_pipline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='median')),
                ("scaler", StandardScaler())
            ])            
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encode", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info("Numerical columns: {Numerical_columns}")
            logging.info("Categorical columns: {categorical_columns}")

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])


            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        


    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = 'math_score'
            numerical_columns = ['writing_score', 'reading_score']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying preprocessing object on training dataframe and testing dataframe")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]


            logging.info("Saved preprocessing object")

            save_obj(
                file_path=self.data_transformation_config.perprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.perprocessor_obj_file_path  
            )

        except Exception as e:
            raise CustomException(e, sys)






