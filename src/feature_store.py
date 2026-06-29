import redis
import json
from src.logger import get_logger

logger = get_logger(__name__)

class RedisFeatureStore:
    def __init__(self, host = "localhost", port = 6379, db = 0):
        self.client = redis.StrictRedis(host = host, port = port, db = db, decode_responses = True)

    def store_features(self, entity_id, features):              #entity_id means your rows
        key = f"entity: {entity_id}: features"
        self.client.set(key, json.dumps(features))

    def get_features(self, entity_id):
        key = f"entity: {entity_id}: features"
        features = self.client.get(key)
        if features:
            return json.loads(features)
        else:
            return None
        
    def store_batch_features(self, batch_data):
        for entity_id, features in batch_data.items():
            self.store_features(entity_id, features)

    def get_batch_features(self, entity_ids):
        batch_features = {}
        for entity_id in entity_ids:
            batch_features[entity_id] = self.get_features(entity_id)
        return batch_features
    
    def get_all_entity_ids(self):
        """
        Returns all entity IDs stored in Redis.
        """
        try:
            # Use self.client instead of self.redis_client
            keys = self.client.keys()
            
            entity_ids = []
            for key in keys:
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                
                # Extract entity_id from key pattern "entity: {id}: features"
                if key.startswith("entity:") and key.endswith(": features"):
                    entity_id = key.replace("entity: ", "").replace(": features", "")
                    entity_ids.append(entity_id)
                else:
                    # If keys are stored directly as entity_id (fallback)
                    entity_ids.append(key)
            
            return entity_ids
        except Exception as e:
            logger.error(f"Error while getting all entity IDs: {e}")
            return []
                    
