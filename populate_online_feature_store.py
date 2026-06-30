import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.online_feature_store import OnlineFeatureStore
from src.logger import get_logger

logger = get_logger(__name__)

def populate_online_store():
    try:
        # Load your processed data (from CSV or your preprocessing output)
        # Option A: From CSV (recommended for now)
        df = pd.read_csv("artifacts/processed/X_train.csv")   # or wherever your processed data is
        
        # If you also saved target, merge it
        # df['target_binary'] = ... 

        online_store = OnlineFeatureStore()

        print(f"Populating Online Feature Store with {len(df)} records...")

        for idx, row in df.iterrows():
            features = row.to_dict()
            
            # Convert numpy types to native Python types (important for Redis)
            features = {k: float(v) if isinstance(v, (int, float)) else v for k, v in features.items()}
            
            # Store with patient_id as string of index
            patient_id = str(idx)
            online_store.store_patient_features(patient_id, features)

        print("✅ Online Feature Store populated successfully!")
        print(f"Total patients stored: {len(df)}")
        print(f"You can now use patient_ids from 0 to {len(df)-1}")

    except Exception as e:
        logger.error(f"Failed to populate online store: {e}")
        raise

if __name__ == "__main__":
    populate_online_store()