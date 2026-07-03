import pandas as pd
import numpy as np
import redis
import json
import os
from dotenv import load_dotenv

# Load environment variables (for local testing)
if os.getenv("KUBERNETES_SERVICE_HOST") is None:
    load_dotenv()

redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT_NUMBER", 6379))

print(f"Connecting to Redis at {redis_host}:{redis_port}...")

client = redis.StrictRedis(
    host=redis_host,
    port=redis_port,
    db=0,
    decode_responses=True
)

try:
    client.ping()
    print("✅ Connected to Redis successfully!")
except Exception as e:
    print(f"❌ Redis connection failed: {e}")
    exit(1)

# ====================== LOAD DATA ======================
train_path = "artifacts/processed/heart_disease_train.csv"

if os.path.exists(train_path):
    print(f"Loading data from: {train_path}")
    df = pd.read_csv(train_path)
else:
    print("Processed CSV not found. Downloading original UCI Heart Disease data...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    
    df = pd.read_csv(url, header=None, names=columns)
    df = df.replace('?', np.nan).apply(pd.to_numeric, errors='coerce')
    df['ca'] = df['ca'].fillna(df['ca'].median())
    df['thal'] = df['thal'].fillna(df['thal'].median())
    df['target_binary'] = (df['target'] > 0).astype(int)
    print(f"Downloaded and preprocessed {len(df)} records from UCI.")

# ====================== POPULATE REDIS ======================
print("Populating Redis with patient data...")

count = 0
for idx, row in df.iterrows():
    # Prepare features (exclude target columns)
    features = {}
    for col in df.columns:
        if col not in ['target', 'target_binary']:
            val = row[col]
            features[col] = float(val) if pd.notna(val) else 0.0
    
    # Add target info
    features['target'] = int(row.get('target', 0))
    features['target_binary'] = int(row.get('target_binary', 0))
    
    # Store with correct key format (matches OnlineFeatureStore)
    key = f"patient:{idx}:features"
    client.set(key, json.dumps(features))
    count += 1

print(f"✅ Successfully populated {count} patient records into Redis!")
print("You can now use patient_ids from 0 to", count - 1)