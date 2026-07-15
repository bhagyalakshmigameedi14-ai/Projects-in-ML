import os
import sys
import dill
from src.exception import CustomException


def save_object(file_path: str, obj):
    try:
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    try:
        with open(file_path, 'rb') as file:
            return dill.load(file)
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_ingenstion import DataIngestion
    from src.components.data_transfromation import DataTransfromation

    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransfromation()
    data_transformation.initiate_data_transfromation(train_data, test_data)
