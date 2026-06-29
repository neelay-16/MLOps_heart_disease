from airflow import DAG
from airflow.providers.google.cloud.transfers.gcs_to_local import GCSToLocalFilesystemOperator
from airflow.providers.google.cloud.operators.gcs import GCSListObjectsOperator
from airflow.operators.python import PythonOperator
from airflow.hooks.base import BaseHook
from datetime import datetime
import pandas as pd
import sqlalchemy
import urllib.parse

#### TRANSFORM STEP....
def load_to_sql(file_path):
    conn = BaseHook.get_connection('postgres')  
    
    # Safely encode the password to handle any special characters
    safe_password = urllib.parse.quote_plus(conn.password)
    
    # Use the safe password in your f-string
    engine = sqlalchemy.create_engine(f"postgresql+psycopg2://{conn.login}:{safe_password}@ml-ops-assignment_7fc493-postgres-1:{conn.port}/{conn.schema}")
    
    df = pd.read_csv(file_path)
    df.to_sql(name="heart_disease", con=engine, if_exists="replace", index=False)

# Define the DAG
with DAG(
    dag_id="extract_heart_disease_data",
    schedule=None, 
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    # Extract STEP...
    list_files = GCSListObjectsOperator(
        task_id="list_files",
        bucket="heart_disease_bucket", 
    )

    download_file = GCSToLocalFilesystemOperator(
        task_id="download_file",
        bucket="heart_disease_bucket", 
        object_name="processed.cleveland.data", 
        filename="/tmp/processed.cleveland.data", 
    )
    
    ### TRANSFORM AND LOAD....
    load_data = PythonOperator(
        task_id="load_to_sql",
        python_callable=load_to_sql,
        op_kwargs={"file_path": "/tmp/processed.cleveland.data"}
    )

    list_files >> download_file >> load_data
