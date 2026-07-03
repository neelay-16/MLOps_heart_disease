from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import psycopg2
from config.database_config import DB_CONFIG
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
import json

# ====================== CONFIGURATION ======================
PUSHGATEWAY_URL = "http://prometheus-kube-prometheus-prometheus-pushgateway.monitoring.svc.cluster.local:9091"
REFERENCE_DATA_PATH = "artifacts/X_train.csv"

default_args = {
    'owner': 'mlops-team',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# ====================== HELPER FUNCTIONS ======================

def get_db_connection():
    """Create and return PostgreSQL connection"""
    return psycopg2.connect(**DB_CONFIG)


def extract_recent_predictions():
    """Extract predictions from last 30 days"""
    conn = get_db_connection()
    query = """
        SELECT features 
        FROM prediction_logs 
        WHERE timestamp >= NOW() - INTERVAL '30 days'
    """
    df = pd.read_sql(query, conn)
    conn.close()

    if df.empty:
        print("No recent predictions found in the last 30 days.")
        return pd.DataFrame()

    # Convert JSONB column back to DataFrame
    features_list = df['features'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    current_data = pd.DataFrame(features_list.tolist())
    
    print(f"Extracted {len(current_data)} recent predictions for drift analysis.")
    return current_data


def detect_data_drift(**context):
    """Detect Data Drift using Evidently AI"""
    current_data = extract_recent_predictions()

    if current_data.empty:
        print("Skipping drift detection due to insufficient data.")
        context['ti'].xcom_push(key='drift_detected', value=False)
        context['ti'].xcom_push(key='drift_share', value=0.0)
        return {"drift_detected": False, "drift_share": 0.0}

    # Load Reference (Training) Data
    try:
        reference_data = pd.read_csv(REFERENCE_DATA_PATH)
    except Exception as e:
        raise Exception(f"Failed to load reference data from {REFERENCE_DATA_PATH}: {e}")

    # Define column mapping for Evidently
    column_mapping = ColumnMapping()
    column_mapping.numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    column_mapping.categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    # Run Evidently Data Drift Report
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_data, 
        current_data=current_data, 
        column_mapping=column_mapping
    )

    # Get results
    result_dict = report.as_dict()
    dataset_drift = result_dict['metrics'][0]['result']['dataset_drift']
    drift_share = result_dict['metrics'][0]['result']['drift_share']
    number_of_drifted_columns = result_dict['metrics'][0]['result']['number_of_drifted_columns']

    print(f"\n{'='*60}")
    print(f"Data Drift Detection Results (Last 30 days)")
    print(f"{'='*60}")
    print(f"Drift Detected     : {dataset_drift}")
    print(f"Drift Share        : {drift_share:.4f}")
    print(f"Drifted Features   : {number_of_drifted_columns}")
    print(f"{'='*60}\n")

    # Push metrics to Prometheus
    push_metrics_to_prometheus(dataset_drift, drift_share, number_of_drifted_columns)

    # Push results to XCom (can be used by downstream tasks)
    context['ti'].xcom_push(key='drift_detected', value=dataset_drift)
    context['ti'].xcom_push(key='drift_share', value=drift_share)

    return {
        "drift_detected": dataset_drift,
        "drift_share": drift_share,
        "drifted_features": number_of_drifted_columns
    }


def push_metrics_to_prometheus(drift_detected, drift_share, drifted_features):
    """Push drift metrics to Prometheus Pushgateway"""
    registry = CollectorRegistry()

    # Define metrics
    drift_detected_gauge = Gauge(
        'data_drift_detected', 
        'Whether data drift was detected (1 = Yes, 0 = No)', 
        registry=registry
    )
    drift_share_gauge = Gauge(
        'data_drift_share', 
        'Proportion of features that drifted', 
        registry=registry
    )
    drifted_features_gauge = Gauge(
        'data_drifted_features_count', 
        'Number of features that showed drift', 
        registry=registry
    )

    # Set values
    drift_detected_gauge.set(1 if drift_detected else 0)
    drift_share_gauge.set(drift_share)
    drifted_features_gauge.set(drifted_features)

    # Push to Pushgateway
    push_to_gateway(PUSHGATEWAY_URL, job='drift_detection', registry=registry)
    print("Drift metrics successfully pushed to Prometheus.")


# ====================== DAG DEFINITION ======================

with DAG(
    dag_id='drift_detection_weekly',
    default_args=default_args,
    schedule_interval='@weekly',           # Runs every Sunday
    start_date=datetime(2026, 1, 1),
    catchup=False,
    description='Weekly Data Drift Detection using Evidently AI',
    tags=['drift', 'evidently', 'monitoring', 'mlops']
) as dag:

    extract_task = PythonOperator(
        task_id='extract_recent_predictions',
        python_callable=extract_recent_predictions
    )

    drift_detection_task = PythonOperator(
        task_id='detect_data_drift',
        python_callable=detect_data_drift
    )

    extract_task >> drift_detection_task