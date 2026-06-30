import redis
import json
from src.logger import get_logger
from src.custom_exception import CustomException
import sys

logger = get_logger(__name__)


class OnlineFeatureStore:
    """
    Online Feature Store for real-time predictions
    """
    def __init__(self, host="localhost", port=6379, db=0, password=None):
        try:
            self.client = redis.StrictRedis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
                socket_timeout=5
            )
            # Test connection
            self.client.ping()
            logger.info("✅ Connected to Online Feature Store (Redis)")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise CustomException("Redis connection failed", sys.exc_info())

    def store_patient_features(self, patient_id: str, features: dict):
        """Store latest features for a patient (used during data ingestion)"""
        try:
            key = f"patient:{patient_id}:features"
            self.client.set(key, json.dumps(features))
            # Optional: Set expiration (e.g., 30 days)
            self.client.expire(key, 30 * 24 * 60 * 60)
            logger.info(f"Stored features for patient {patient_id}")
        except Exception as e:
            logger.error(f"Failed to store patient features: {e}")

    def get_patient_features(self, patient_id: str):
        """Get latest features for real-time prediction"""
        try:
            key = f"patient:{patient_id}:features"
            features = self.client.get(key)
            if features:
                return json.loads(features)
            else:
                logger.warning(f"No features found for patient {patient_id}")
                return None
        except Exception as e:
            logger.error(f"Error fetching patient features: {e}")
            return None

    def get_all_patient_ids(self):
        """Get all patients currently in online store"""
        try:
            keys = self.client.keys("patient:*:features")
            patient_ids = []
            for key in keys:
                pid = key.split(":")[1]
                patient_ids.append(pid)
            return patient_ids
        except Exception as e:
            logger.error(f"Error getting patient IDs: {e}")
            return []