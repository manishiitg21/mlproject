import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import load_obj
import pandas as pd
import os



class Predictpipeline:
    def __init__(self):
        pass
    def predict(self,feature):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            processor_path=os.path.join("artifacts","preprocessor.pkl")
            model=load_obj(model_path)
            processor=load_obj(processor_path)
            pred=model.predict(processor.transform(feature))
            logging.info("prediction is done")
            return pred
        except Exception as e:
            raise CustomException(e,sys)
            
class CustomData:
    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)          
            
            
            
        