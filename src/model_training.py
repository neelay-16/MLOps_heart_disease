from src.logger import get_logger
from src.custom_exception import CustomException
import pandas as pd
import numpy as np
import os
import pickle
import sys

from src.feature_store import RedisFeatureStore
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

import mlflow
import mlflow.sklearn
import mlflow.keras

logger = get_logger(__name__)



import os

class HeartDiseaseModelTraining:
    def __init__(self, feature_store: RedisFeatureStore, model_save_path=r"C:\Users\ramba\Desktop\MLOps Assignment\artifacts\models"):
        self.feature_store = feature_store
        self.model_save_path = model_save_path
        self.models = {}  # To store trained models

        os.makedirs(self.model_save_path, exist_ok=True)
        logger.info("Heart Disease Model Training initialized...")

    # ====================== REDIS DATA LOADING ======================
    def load_data_from_redis(self, entity_ids):
        try:
            logger.info("Extracting data from Redis")
            data = []
            for entity_id in entity_ids:
                features = self.feature_store.get_features(entity_id)
                if features:
                    data.append(features)
                else:
                    logger.warning(f"Feature not found for entity_id: {entity_id}")
            return data
        except Exception as e:
            logger.error(f"Error while loading data from Redis: {e}")
            raise CustomException(str(e), sys.exc_info())

    def prepare_data(self):
        try:
            logger.info("Preparing data from Redis...")

            entity_ids = self.feature_store.get_all_entity_ids()

            if not entity_ids:
                raise CustomException("No data found in Redis. Please run data_preprocessing.py first.")

            logger.info(f"Total records found in Redis: {len(entity_ids)}")

            train_entity_ids, test_entity_ids = train_test_split(
                entity_ids, test_size=0.2, random_state=42
            )

            train_data = self.load_data_from_redis(train_entity_ids)
            test_data = self.load_data_from_redis(test_entity_ids)

            train_df = pd.DataFrame(train_data)
            test_df = pd.DataFrame(test_data)

            # ====================== IMPORTANT: Handle NaN in target ======================
            # Drop rows where target_binary is missing
            train_df = train_df.dropna(subset=['target_binary'])
            test_df = test_df.dropna(subset=['target_binary'])

            # Convert to integer (in case it's float due to NaN)
            train_df['target_binary'] = train_df['target_binary'].astype(int)
            test_df['target_binary'] = test_df['target_binary'].astype(int)

            # Drop target columns
            X_train = train_df.drop(['target', 'target_binary'], axis=1, errors='ignore')
            X_test = test_df.drop(['target', 'target_binary'], axis=1, errors='ignore')

            y_train = train_df['target_binary']
            y_test = test_df['target_binary']

            logger.info(f"Data prepared successfully. Train: {X_train.shape}, Test: {X_test.shape}")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error(f"Error while preparing data: {e}")
            raise CustomException(str(e), sys.exc_info())

    # ====================== MODEL 1: LOGISTIC REGRESSION ======================
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        try:
            logger.info("Training Logistic Regression...")

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(X_train_scaled, y_train)

            y_pred = lr.predict(X_test_scaled)
            y_prob = lr.predict_proba(X_test_scaled)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)

            logger.info(f"Logistic Regression - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")

            self.models['LogisticRegression'] = {'model': lr, 'scaler': scaler}
            return lr, scaler

        except Exception as e:
            logger.error(f"Error training Logistic Regression: {e}")
            raise CustomException(str(e), sys.exc_info())

    # ====================== MODEL 2: RANDOM FOREST ======================
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        try:
            logger.info("Training Random Forest with Hyperparameter Tuning...")

            param_distributions = {
                'n_estimators': [100, 200, 300],
                'max_depth': [8, 10, 15, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            }

            rf = RandomForestClassifier(random_state=42, class_weight='balanced')

            # ====================== IMPORTANT FIX ======================
            random_search = RandomizedSearchCV(
                rf, 
                param_distributions, 
                n_iter=15, 
                cv=3, 
                scoring='roc_auc', 
                random_state=42,
                n_jobs=1          # ← Changed from -1 to 1 (fixes Windows issue)
            )
            # ===========================================================

            random_search.fit(X_train, y_train)

            best_rf = random_search.best_estimator_
            logger.info(f"Best Random Forest Params: {random_search.best_params_}")

            y_pred = best_rf.predict(X_test)
            y_prob = best_rf.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)

            logger.info(f"Random Forest - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")

            self.models['RandomForest'] = best_rf
            return best_rf

        except Exception as e:
            logger.error(f"Error training Random Forest: {e}")
            raise CustomException(str(e), sys.exc_info())
    # ====================== MODEL 3: KERAS NEURAL NETWORK ======================
    def train_keras_model(self, X_train, y_train, X_test, y_test):
        try:
            logger.info("Training Keras Neural Network...")

            from tensorflow.keras.layers import Input

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # ====================== FIXED: Use Input layer ======================
            model = Sequential([
                Input(shape=(X_train.shape[1],)),           # ← Fixed here
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dropout(0.3),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            # ===================================================================

            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
            )

            early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

            model.fit(
                X_train_scaled, y_train,
                validation_split=0.2,
                epochs=100,
                batch_size=16,
                callbacks=[early_stop],
                verbose=0
            )

            y_prob = model.predict(X_test_scaled, verbose=0).ravel()
            y_pred = (y_prob > 0.5).astype(int)

            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)

            logger.info(f"Keras Neural Network - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")

            self.models['KerasNN'] = {'model': model, 'scaler': scaler}
            return model, scaler

        except Exception as e:
            logger.error(f"Error training Keras model: {e}")
            raise CustomException(str(e), sys.exc_info())

    # ====================== SAVE MODELS ======================
    def save_models(self):
        try:
            saved = []

            # Save Logistic Regression + Scaler
            if 'LogisticRegression' in self.models:
                with open(f"{self.model_save_path}logistic_regression.pkl", 'wb') as f:
                    pickle.dump(self.models['LogisticRegression'], f)
                saved.append("Logistic Regression")

            # Save Random Forest
            if 'RandomForest' in self.models:
                with open(f"{self.model_save_path}random_forest.pkl", 'wb') as f:
                    pickle.dump(self.models['RandomForest'], f)
                saved.append("Random Forest")

            # Save Keras Model + Scaler
            if 'KerasNN' in self.models:
                self.models['KerasNN']['model'].save(f"{self.model_save_path}keras_neural_network.keras")
                with open(f"{self.model_save_path}keras_scaler.pkl", 'wb') as f:
                    pickle.dump(self.models['KerasNN']['scaler'], f)
                saved.append("Keras Neural Network")

            if saved:
                logger.info(f"Models saved successfully: {', '.join(saved)}")
            else:
                logger.warning("No models were saved.")

        except Exception as e:
            logger.error(f"Error while saving models: {e}")
            raise CustomException(str(e), sys.exc_info())
        
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """
        Evaluates a model and returns key metrics.
        Works for both sklearn and Keras models.
        """
        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                # For Keras model
                y_prob = model.predict(X_test, verbose=0).ravel()

            y_pred = (y_prob > 0.5).astype(int)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_prob)
            }

            logger.info(f"{model_name} Evaluation Metrics: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Error while evaluating {model_name}: {e}")
            raise CustomException(str(e), sys.exc_info())
    
    def run(self):
        try:
            import mlflow
            import mlflow.sklearn
            import mlflow.keras

            logger.info("Starting Heart Disease Model Training with MLflow...")

            # ====================== MLflow Setup ======================
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            mlflow.set_experiment("Heart_Disease_Prediction")

            with mlflow.start_run(run_name="Heart_Disease_Training"):

                X_train, X_test, y_train, y_test = self.prepare_data()

                best_model = None
                best_roc_auc = 0
                best_model_name = ""

                mlflow.log_param("train_samples", X_train.shape[0])
                mlflow.log_param("test_samples", X_test.shape[0])
                mlflow.log_param("features", X_train.shape[1])

                # ====================== Train All Models ======================

                # 1. Logistic Regression
                try:
                    lr_model, _ = self.train_logistic_regression(X_train, y_train, X_test, y_test)
                    lr_metrics = self.evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
                    mlflow.log_metrics({f"LR_{k}": v for k, v in lr_metrics.items()})

                    if lr_metrics["roc_auc"] > best_roc_auc:
                        best_roc_auc = lr_metrics["roc_auc"]
                        best_model = lr_model
                        best_model_name = "Logistic_Regression"
                except Exception as e:
                    logger.error(f"Logistic Regression failed: {e}")

                # 2. Random Forest
                try:
                    rf_model = self.train_random_forest(X_train, y_train, X_test, y_test)
                    rf_metrics = self.evaluate_model(rf_model, X_test, y_test, "Random Forest")
                    mlflow.log_metrics({f"RF_{k}": v for k, v in rf_metrics.items()})

                    if rf_metrics["roc_auc"] > best_roc_auc:
                        best_roc_auc = rf_metrics["roc_auc"]
                        best_model = rf_model
                        best_model_name = "Random_Forest"
                except Exception as e:
                    logger.error(f"Random Forest failed: {e}")

                # 3. Keras Neural Network
                try:
                    keras_model, _ = self.train_keras_model(X_train, y_train, X_test, y_test)
                    keras_metrics = self.evaluate_model(keras_model, X_test, y_test, "Keras Neural Network")
                    mlflow.log_metrics({f"Keras_{k}": v for k, v in keras_metrics.items()})

                    if keras_metrics["roc_auc"] > best_roc_auc:
                        best_roc_auc = keras_metrics["roc_auc"]
                        best_model = keras_model
                        best_model_name = "Keras_Neural_Network"
                except Exception as e:
                    logger.error(f"Keras model failed: {e}")

                # ====================== Log Best Model ======================
                if best_model is not None:
                    mlflow.log_param("best_model", best_model_name)
                    mlflow.log_metric("best_roc_auc", best_roc_auc)

                    if best_model_name == "Logistic_Regression":
                        mlflow.sklearn.log_model(best_model, name="best_model")
                    elif best_model_name == "Random_Forest":
                        mlflow.sklearn.log_model(best_model, name="best_model")
                    elif best_model_name == "Keras_Neural_Network":
                        mlflow.keras.log_model(best_model, name="best_model")

                    logger.info(f"Best Model: {best_model_name} with ROC-AUC = {best_roc_auc:.4f}")

                logger.info("Model Training completed successfully with MLflow tracking.")

        except Exception as e:
            logger.error(f"Error in Model Training: {e}")
            raise CustomException(str(e), sys.exc_info())
        
if __name__ == "__main__":
    feature_store = RedisFeatureStore()
    model_trainer = HeartDiseaseModelTraining(feature_store)
    model_trainer.run()