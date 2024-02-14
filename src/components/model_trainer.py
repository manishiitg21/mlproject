# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
import pickle
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformationConfig,DataTransformation
from src.components.data_ingestion import DataIngestionConfig,DataIngestion


@dataclass
class ModelTrainerConfig:
    model_trainer_file_path=os.path.join("artifacts","model.pkl")


# +
class ModelTrain:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
        
        
    def train(self,train_data):
        try:
            target_column='math_score'
        
            X_train=train_data.drop(columns=[target_column],axis=1)
            y_train=train_data[target_column]
            LR=LinearRegression()
            LR.fit(X_train,y_train)
            logging.info("LR model has been trained successfully")

            save_obj(
            file_path=self.model_trainer_config.model_trainer_file_path,
            obj=LR
            )

            logging.info("model has been saved successfully")
            
        except Exception as e:
            raise CustomException(e,sys)
        
        
        
        
    
# -

if __name__=="__main__":
    obj=DataIngestion()
    train_path,test_path=obj.initiate_data_ingestion()
    
    dt=DataTransformation()
    train_data,_,_=dt.initiate_data_transformation(train_path,test_path)
    
    mt=ModelTrain()
    mt.train(train_data)
    
