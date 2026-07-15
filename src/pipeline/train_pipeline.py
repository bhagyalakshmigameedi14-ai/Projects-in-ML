from src.components.data_ingenstion import DataIngestion
from src.components.data_transfromation import DataTransfromation
from src.components.model_trainer import ModelTrainer
from src.logger import logging


def run_training_pipeline():
    logging.info("=== Training Pipeline Started ===")

    # Step 1: Data Ingestion
    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.initiate_data_ingestion()

    # Step 2: Data Transformation
    data_transformation = DataTransfromation()
    train_arr, test_arr, preprocessor_path = (
        data_transformation.initiate_data_transfromation(train_path, test_path)
    )

    # Step 3: Model Training
    model_trainer = ModelTrainer()
    r2_score = model_trainer.initiate_model_training(train_arr, test_arr)

    logging.info(f"=== Training Pipeline Completed — Final R2 Score: {r2_score:.4f} ===")
    print(f"\n[SUCCESS] Training complete! Best model R2 Score: {r2_score:.4f}")
    return r2_score


if __name__ == "__main__":
    run_training_pipeline()
