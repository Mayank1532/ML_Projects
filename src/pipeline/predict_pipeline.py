import os
import sys

import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:

    def __init__(self):
        pass

    def predict(self, features):

        try:

            model_path = os.path.join(
                "artifacts",
                "model.pkl"
            )

            preprocessor_path = os.path.join(
                "artifacts",
                "preprocessor.pkl"
            )

            logging.info("Loading preprocessor...")
            preprocessor = load_object(preprocessor_path)

            logging.info("Loading trained model...")
            model = load_object(model_path)

            logging.info("Transforming input data...")
            data_scaled = preprocessor.transform(features)

            logging.info("Making prediction...")
            prediction = model.predict(data_scaled)

            return prediction

        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)


class CustomData:

    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: float,
        writing_score: float,
    ):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):

        try:

            custom_data_input_dict = {

                "gender": [self.gender],

                "race_ethnicity": [self.race_ethnicity],

                "parental_level_of_education": [
                    self.parental_level_of_education
                ],

                "lunch": [self.lunch],

                "test_preparation_course": [
                    self.test_preparation_course
                ],

                "reading_score": [self.reading_score],

                "writing_score": [self.writing_score]

            }

            df = pd.DataFrame(custom_data_input_dict)

            logging.info("Input dataframe created successfully")

            return df

        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)