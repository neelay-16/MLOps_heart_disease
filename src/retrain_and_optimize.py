import os
import sys
import json
import shutil
import argparse
from datetime import datetime
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.keras
from src.logger import get_logger
from src.custom_exception import CustomException
from src.feature_store import RedisFeatureStore
from src.model_training import HeartDiseaseModelTraining

# ONNX & Quantization imports


from onnxruntime.quantization import quantize_dynamic, QuantType

logger = get_logger(__name__)

# ====================== PATHS ======================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "artifacts")

ONNX_LOGISTIC_PATH = os.path.join(MODEL_SAVE_PATH, "logistic_regression.onnx")
ONNX_RANDOM_FOREST_PATH = os.path.join(MODEL_SAVE_PATH, "random_forest.onnx")
ONNX_KERAS_PATH = os.path.join(MODEL_SAVE_PATH, "keras_neural_network.onnx")
QUANTIZED_KERAS_PATH = os.path.join(MODEL_SAVE_PATH, "keras_neural_network_quantized.onnx")

mlflow.set_tracking_uri("http://172.21.80.1:5000")
logger.info("MLflow tracking URI redirected cleanly to host endpoint port 5000.")


def set_mlflow_experiment():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"retrained_{timestamp}"
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment set to: {experiment_name}")
    return experiment_name


def sync_parquet_to_redis(data_path, feature_store):
    try:

        logger.info(f"Syncing latest drift data from {data_path} into Redis Feature Store...")
        df = pd.read_parquet(data_path)

        batch_data = {}
        for idx, row in df.iterrows():
            entity_id = f"prod_{idx}_{datetime.now().strftime('%M%S')}"
            row_dict = row.to_dict()

            if 'feature_json' in row_dict and row_dict['feature_json']:
                try:
                    features = json.loads(row_dict['feature_json'])
                except Exception:
                    features = row_dict
            else:
                features = row_dict

            label_val = 0
            if 'actual_label' in row_dict and pd.notna(row_dict['actual_label']):
                label_val = int(row_dict['actual_label'])
            elif 'target_binary' in features and pd.notna(features['target_binary']):
                label_val = int(features['target_binary'])
            elif 'target' in features and pd.notna(features['target']):
                label_val = int(features['target'])

            clean_features = {
                "age": float(features.get("age", 0)),
                "sex": int(features.get("sex", 0)),
                "cp": int(features.get("cp", 0)),
                "trestbps": float(features.get("trestbps", 0)),
                "chol": float(features.get("chol", 0)),
                "fbs": int(features.get("fbs", 0)),
                "restecg": int(features.get("restecg", 0)),
                "thalach": float(features.get("thalach", 0)),
                "exang": int(features.get("exang", 0)),
                "oldpeak": float(features.get("oldpeak", 0)),
                "slope": int(features.get("slope", 0)),
                "ca": float(features.get("ca", 0.0)),
                "thal": float(features.get("thal", 0.0)),
                "target": int(label_val),
                "target_binary": int(label_val)
            }
            batch_data[entity_id] = clean_features

        feature_store.store_batch_features(batch_data)
        logger.info(f"Successfully loaded {len(batch_data)} latest drift records into Redis!")

    except Exception as e:
        logger.error(f"Error while seeding Redis Feature Store: {e}")
        raise CustomException(str(e), sys.exc_info())


def convert_models_to_onnx():
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import onnx
        import tensorflow as tf
        import tf2onnx
        logger.info("Converting models to ONNX...")

        # Logistic Regression
        lr_path = os.path.join(MODEL_SAVE_PATH, "modelslogistic_regression.pkl")
        with open(lr_path, "rb") as f:
            lr_data = __import__('pickle').load(f)
        lr_model = lr_data['model']
        scaler = lr_data['scaler']
        n_features = scaler.n_features_in_
        initial_type = [('float_input', FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(lr_model, initial_types=initial_type, target_opset=13)
        onnx.save_model(onnx_model, ONNX_LOGISTIC_PATH)

        # Random Forest
        rf_path = os.path.join(MODEL_SAVE_PATH, "modelsrandom_forest.pkl")
        with open(rf_path, "rb") as f:
            rf_model = __import__('pickle').load(f)
        n_features = rf_model.n_features_in_
        initial_type = [('float_input', FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(rf_model, initial_types=initial_type, target_opset=13)
        onnx.save_model(onnx_model, ONNX_RANDOM_FOREST_PATH)

        # Keras Model
        keras_path = os.path.join(MODEL_SAVE_PATH, "modelskeras_neural_network.keras")
        model = tf.keras.models.load_model(keras_path)
        input_spec = (tf.TensorSpec((None, 13), tf.float32, name="input_layer"),)

        @tf.function
        def model_func(x):
            return model(x)

        onnx_model, _ = tf2onnx.convert.from_function(model_func, input_signature=input_spec, opset=13)
        onnx.save_model(onnx_model, ONNX_KERAS_PATH)

        logger.info("All models converted to ONNX successfully.")

    except Exception as e:
        logger.error(f"ONNX conversion failed: {e}")
        raise CustomException(str(e), sys.exc_info())


def quantize_keras_model():
    try:
        logger.info("Quantizing Keras model...")
        quantize_dynamic(
            model_input=ONNX_KERAS_PATH,
            model_output=QUANTIZED_KERAS_PATH,
            weight_type=QuantType.QInt8
        )
        logger.info(f"Quantized model saved at: {QUANTIZED_KERAS_PATH}")
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        raise CustomException(str(e), sys.exc_info())


def main(data_path=None):
    try:
        if mlflow.active_run():
            mlflow.end_run()

        experiment_name = set_mlflow_experiment()

        with mlflow.start_run(run_name=experiment_name):

            # ====================== DATA LOADING ======================
            if data_path and os.path.exists(data_path):
                logger.info(f"Loading balanced data directly from: {data_path}")
                df = pd.read_parquet(data_path)
                
                # Use the balanced data directly (no Redis involved)
                from sklearn.model_selection import train_test_split
                train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
                
                X_train = train_df.drop(['target', 'target_binary'], axis=1, errors='ignore')
                X_test = test_df.drop(['target', 'target_binary'], axis=1, errors='ignore')
                y_train = train_df['target_binary']
                y_test = test_df['target_binary']
                
                logger.info(f"Using balanced data → Train: {X_train.shape}, Test: {X_test.shape}")

            else:
                logger.info("No data_path provided. Loading from Redis...")
                feature_store = RedisFeatureStore()
                trainer = HeartDiseaseModelTraining(feature_store=feature_store)
                X_train, X_test, y_train, y_test = trainer.prepare_data()

            # ====================== MODEL TRAINING ======================
            trainer = HeartDiseaseModelTraining(feature_store=RedisFeatureStore())
            
            trainer.train_logistic_regression(X_train, y_train, X_test, y_test)
            trainer.train_random_forest(X_train, y_train, X_test, y_test)
            trainer.train_keras_model(X_train, y_train, X_test, y_test)
            trainer.save_models()

            # ====================== ONNX + QUANTIZATION ======================
            convert_models_to_onnx()
            quantize_keras_model()

            # ====================== LOG TO MLFLOW ======================
            mlflow.log_artifact(ONNX_LOGISTIC_PATH)
            mlflow.log_artifact(ONNX_RANDOM_FOREST_PATH)
            mlflow.log_artifact(ONNX_KERAS_PATH)
            if os.path.exists(QUANTIZED_KERAS_PATH):
                mlflow.log_artifact(QUANTIZED_KERAS_PATH)

            logger.info("Retraining and optimization completed successfully!")

    except Exception as e:
        if mlflow.active_run():
            mlflow.end_run()
        logger.error(f"Pipeline failed: {e}")
        raise CustomException(str(e), sys.exc_info())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None, help="Path to preprocessed data (parquet)")
    args = parser.parse_args()

    main(data_path=args.data_path)
