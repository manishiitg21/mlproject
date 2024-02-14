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
import pandas as pd
from src.utils import save_obj
from sklearn.metrics import r2_score


@dataclass
class ModelTrainerConfig:
    model_trainer_file_path=os.path.join("artifacts","model.pkl")


# +
class ModelTrain:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_arr,test_arr,preprocessor_file_path):
        try:
            logging.info("get train and tes data")
            X_train,y_train,X_test,y_test=(
            train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1]          
            
            )
            LR=LinearRegression()
            LR.fit(X_train,y_train)
            logging.info("LR model has been trained successfully")

            save_obj(
            file_path=self.model_trainer_config.model_trainer_file_path,
            obj=LR
            )

            logging.info("model has been saved successfully")
            
            y_pred=LR.predict(X_test)
            
            r2_sc=r2_score(y_test,y_pred)
            logging.info("got the r2score")
            return r2_sc
        
            
            
            
        
        
        
        except Exception as e:
            raise CustomeException(e,sys)
        
        
        

