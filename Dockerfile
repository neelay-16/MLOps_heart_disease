FROM astrocrpublic.azurecr.io/runtime:3.2-4

RUN pip install apache-airflow-providers-google

RUN pip install apache-airflow-providers-postgres


