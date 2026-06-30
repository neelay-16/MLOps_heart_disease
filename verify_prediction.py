import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.online_feature_store import OnlineFeatureStore

def verify_prediction(patient_id="5"):
    print("="*80)
    print(f"VERIFYING PREDICTION FOR patient_id = {patient_id}")
    print("="*80)

    # 1. Get features from Online Feature Store (Redis)
    online_store = OnlineFeatureStore()
    features = online_store.get_patient_features(patient_id)

    if not features:
        print(f"❌ No features found for patient_id = {patient_id}")
        return

    print("\n📌 Features stored in Online Feature Store:")
    for k, v in features.items():
        print(f"   {k:12} : {v}")

    # 2. Load original training data to find true label
    try:
        X_train = pd.read_csv("artifacts/processed/X_train.csv")
        y_train = pd.read_csv("artifacts/processed/y_train.csv")

        idx = int(patient_id)

        if idx >= len(X_train):
            print(f"\n❌ patient_id {patient_id} is out of range. Max index = {len(X_train)-1}")
            return

        true_label = int(y_train.iloc[idx].values[0])

        print("\n" + "-"*80)
        print(f"📌 Original Row Index in Training Data : {idx}")
        print(f"📌 True Label (Ground Truth)           : {true_label} "
              f"({'Heart Disease' if true_label == 1 else 'No Heart Disease'})")
        print("-"*80)

        # 3. Show what the model predicted (you already have this from API)
        print("\n📌 Model Prediction (from your API call):")
        print("   Final Prediction : 0 (LOW RISK - No Heart Disease)")
        print("   Average Probability : 0.3891")

        # 4. Final Verdict
        print("\n" + "="*80)
        if true_label == 0:
            print("✅ CORRECT PREDICTION! The model correctly predicted LOW RISK.")
        else:
            print("❌ INCORRECT PREDICTION! The model predicted LOW RISK but patient actually has Heart Disease.")
        print("="*80)

    except FileNotFoundError:
        print("\n⚠️  Could not find artifacts/processed/X_train.csv or y_train.csv")
        print("Please make sure you have saved the processed data.")

if __name__ == "__main__":
    verify_prediction(patient_id="5")