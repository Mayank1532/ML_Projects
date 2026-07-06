import sys

from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:

    def __init__(self):
        pass

    def run_pipeline(self):

        try:

            logging.info("=" * 60)
            logging.info("TRAINING PIPELINE STARTED")
            logging.info("=" * 60)

            # Step 1: Data Ingestion
            data_ingestion = DataIngestion()

            train_path, test_path = data_ingestion.initiate_data_ingestion()

            logging.info("Data Ingestion Completed")

            # Step 2: Data Transformation
            data_transformation = DataTransformation()

            train_arr, test_arr, _ = (
                data_transformation.initiate_data_transformation(
                    train_path,
                    test_path
                )
            )

            logging.info("Data Transformation Completed")

            # Step 3: Model Training
            model_trainer = ModelTrainer()

            r2_score = model_trainer.initiate_model_trainer(
                train_arr,
                test_arr
            )

            logging.info(f"Training Completed Successfully")
            logging.info(f"Final R2 Score : {r2_score:.4f}")

            return r2_score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    pipeline = TrainPipeline()

    score = pipeline.run_pipeline()

    print(f"\nTraining Completed Successfully")
    print(f"Model R² Score : {score:.4f}")