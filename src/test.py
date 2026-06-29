from src.feature_store import RedisFeatureStore

fs = RedisFeatureStore()
entity_ids = fs.get_all_entity_ids()

print(f"Total entity IDs found: {len(entity_ids)}")
print(f"First 10 entity IDs: {entity_ids[:10]}")