import psycopg2
import json
import uuid
from datetime import datetime
from config.database_config import DB_CONFIG
from src.logger import get_logger

logger = get_logger(__name__)

def log_prediction(patient_id, features, prediction, probability, model_version="v1"):
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            database=DB_CONFIG['database'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO prediction_logs 
            (request_id, patient_id, features, prediction, probability, model_version, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            str(uuid.uuid4()),
            str(patient_id) if patient_id else None,
            json.dumps(features),
            int(prediction),
            float(probability),
            model_version,
            datetime.now()
        ))

        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"Prediction logged for patient_id: {patient_id}")

    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")