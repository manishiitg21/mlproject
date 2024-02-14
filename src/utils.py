import pandas as pd
import pickle
from src.exception import CustomException
import os


def save_obj(file_path,obj):
    try:
        dir_path=os.path.dir_name(file_path)
        
        os.makedir(dir_path,exist_ok=True)
        
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)