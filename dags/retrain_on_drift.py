from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
import pandas as pd
import requests
import os
import sys

# Add project path
sys.path.append("/usr/local/airflow")

#from src.retrain_and_optimize import main as retrain_main

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='retrain_on_drift_detection',
    default_args=default_args,
    description='Retrains models on latest data when drift is detected',
    schedule='0 */6 * * *',   # Runs every 6 hours
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['mlops', 'drift', 'retraining'],
) as dag:

    def check_drift(**context):
        """Check Prometheus for Data Drift or Concept Drift with explicit list indexing"""
        import traceback
        prometheus_url = "http://host.docker.internal:9090"

        # Initialize explicit fallback values
        data_drift = 0.0
        concept_drift = 0.0

        try:
            print("Querying production drift vector states from Prometheus...")
            
            # Fetch data from Prometheus
            data_drift_resp = requests.get(
                f"{prometheus_url}/api/v1/query",
                params={"query": "ml_data_drift_detected"},
                timeout=10
            ).json()

            concept_drift_resp = requests.get(
                f"{prometheus_url}/api/v1/query",
                params={"query": "ml_concept_drift_score"},
                timeout=10
            ).json()

            # Navigate down to the result arrays safely
            data_results = data_drift_resp.get('data', {}).get('result', [])
            concept_results = concept_drift_resp.get('data', {}).get('result', [])

            # Safe extraction for Data Drift
            if data_results and len(data_results) > 0:
                value_array = data_results[0].get('value', [])
                if len(value_array) >= 2:
                    data_drift = float(value_array[1])  # Index 1 contains the string "1" or "0"

            # Safe extraction for Concept Drift
            if concept_results and len(concept_results) > 0:
                value_array = concept_results[0].get('value', [])
                if len(value_array) >= 2:
                    concept_drift = float(value_array[1])

        except Exception as e:
            print(f"CRITICAL ERROR inside try block: {e}")
            print(traceback.format_exc())
            data_drift, concept_drift = 0.0, 0.0

        print(f"Final Parsed Values -> Data Drift: {data_drift}, Concept Drift: {concept_drift}")

        # Pivot to the correct branch
        if data_drift == 1.0 or concept_drift == 1.0:
            print("Drift verified! Routing to extract_and_preprocess_data task pipeline...")
            return 'extract_and_preprocess_data'
        else:
            print("No active drift found. Skipping training optimizations...")
            return 'no_retraining_needed'


    def no_retraining_needed(**context):
        print("No significant drift detected. Skipping retraining.")

    def extract_and_preprocess_data(**context):
        """Extract last 30 days labeled data + balance classes"""
        hook = PostgresHook(postgres_conn_id='heart_disease_db')
        
        query = """
            SELECT * FROM prediction_logs 
            WHERE actual_label IS NOT NULL 
            AND created_at >= NOW() - INTERVAL '30 days'
        """
        df = hook.get_pandas_df(query)
        print(f"Class distribution:\n{df['actual_label'].value_counts(normalize=True)}")
        print(f"Before balancing → {df['actual_label'].value_counts().to_dict()}")

        # Balance classes (important to avoid bias)
        min_count = df['actual_label'].value_counts().min()
        df_balanced = df.groupby('actual_label').sample(n=min_count, random_state=42)
        
        print(f"After balancing → {df_balanced['actual_label'].value_counts().to_dict()}")

        # Save with timestamp
        today = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"/tmp/retraining_data/{today}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "retraining_data.parquet")
        
        df_balanced.to_parquet(output_path, index=False)
        print(f"Preprocessed data saved at: {output_path}")

        # Push to XCom
        context['ti'].xcom_push(key='retraining_data_path', value=output_path)
        context['ti'].xcom_push(key='retraining_date', value=today)

        return output_path   # Important for XCom

    def push_data_to_dvc(**context):
        """Push preprocessed training data to DVC (GCS)"""
        import os
        import shutil
        import subprocess
        from datetime import datetime

        data_path = context['ti'].xcom_pull(task_ids='extract_and_preprocess_data')

        if not data_path or not os.path.exists(data_path):
            print("No valid data path received. Skipping DVC push.")
            return None

        # Create dated file
        today = datetime.now().strftime("%Y-%m-%d")
        dvc_dir = "/usr/local/airflow/dags/data/retraining"
        os.makedirs(dvc_dir, exist_ok=True)

        dvc_filename = f"retraining_data_{today}.parquet"
        dvc_file_path = os.path.join(dvc_dir, dvc_filename)

        # Copy the balanced data
        shutil.copy(data_path, dvc_file_path)
        print(f"Data copied to DVC folder: {dvc_file_path}")

        try:
            # Add to DVC tracking
            subprocess.run(["dvc", "add", dvc_file_path], check=True, cwd="/usr/local/airflow/dags")

            # Push to GCS remote
            subprocess.run(["dvc", "push"], check=True, cwd="/usr/local/airflow/dags")

            print(f"✅ Successfully pushed {dvc_filename} to DVC remote (gs://heart_disease_dvc/)")
            return dvc_file_path

        except subprocess.CalledProcessError as e:
            print(f"❌ DVC push failed: {e}")
            raise e

    def run_retrain_and_optimize(**context):
        """Run retraining using data extracted from prediction_logs"""
        data_path = context['ti'].xcom_pull(
            task_ids='extract_and_preprocess_data'
        )
        
        print(f"Data path received: {data_path}")

        if data_path and os.path.exists(data_path):
            from src.retrain_and_optimize import main
            main(data_path=data_path)
        else:
            print("ERROR: Data path is missing or file does not exist!")
            raise ValueError("retraining_data_path is None or invalid.")

    # ====================== TASK DEFINITIONS ======================
    check_drift_task = BranchPythonOperator(
        task_id='check_drift_from_prometheus',
        python_callable=check_drift,
    )

    no_retrain_task = PythonOperator(
        task_id='no_retraining_needed',
        python_callable=no_retraining_needed,
    )

    extract_task = PythonOperator(
        task_id='extract_and_preprocess_data',
        python_callable=extract_and_preprocess_data,
    )

    dvc_task = PythonOperator(
        task_id='push_data_to_dvc',
        python_callable=push_data_to_dvc,
    )

    retrain_task = PythonOperator(
        task_id='run_retrain_and_optimize',
        python_callable=run_retrain_and_optimize,
    )

    # ====================== TASK DEPENDENCIES ======================
    check_drift_task >> [extract_task, no_retrain_task]
    extract_task >> dvc_task >> retrain_task