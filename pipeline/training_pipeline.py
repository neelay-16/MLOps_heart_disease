from src.data_preprocessing import HeartDiseaseDataProcessing
from src.model_training import HeartDiseaseModelTraining
from src.feature_store import RedisFeatureStore
from src.logger import get_logger
from src.custom_exception import CustomException
import sys

logger = get_logger(__name__)


def run_training_pipeline():
    try:
        logger.info("========== Starting Heart Disease Training Pipeline ==========")

        # ====================== 1. Data Preprocessing ======================
        logger.info("Step 1: Running Data Preprocessing...")

        feature_store = RedisFeatureStore()
        data_path = r"C:\WILP 3rd sem\MLOps\Assignment_01\heart+disease\processed.cleveland.data"

        data_processor = HeartDiseaseDataProcessing(
            data_path=data_path,
            feature_store=feature_store
        )
        data_processor.run()

        logger.info("Data Preprocessing completed successfully.")

        # ====================== 2. Model Training ======================
        logger.info("Step 2: Running Model Training...")

        model_trainer = HeartDiseaseModelTraining(feature_store=feature_store)
        model_trainer.run()

        logger.info("Model Training completed successfully.")

        logger.info("========== Training Pipeline Completed Successfully ==========")

    except Exception as e:
        logger.error(f"Error in Training Pipeline: {e}")
        raise CustomException("Training Pipeline Failed", sys.exc_info())


if __name__ == "__main__":
    run_training_pipeline()