from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import logging

logger = logging.getLogger(__name__)

default_args = {
    "owner": "compagnon-immobilier",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

def collect_data(**context):
    """Simule la collecte des données DVF."""
    logger.info("Étape 1 : Collecte des données DVF...")
    logger.info("Données disponibles dans /data/raw/")
    logger.info("collect_data terminé avec succès")
    return {"status": "ok", "rows": 150000}

def preprocess_data(**context):
    """Simule le prétraitement des données."""
    ti = context["ti"]
    collect_result = ti.xcom_pull(task_ids="collect_data")
    logger.info(f"Étape 2 : Prétraitement de {collect_result['rows']} lignes...")
    logger.info("Nettoyage, feature engineering, split train/test...")
    logger.info("preprocess_data terminé avec succès")
    return {"status": "ok", "features": 31}

def train_model(**context):
    """Simule le réentraînement du modèle XGBoost."""
    ti = context["ti"]
    preprocess_result = ti.xcom_pull(task_ids="preprocess_data")
    logger.info(f"Étape 3 : Entraînement XGBoost avec {preprocess_result['features']} features...")
    logger.info("Logging vers MLflow : http://mlflow:5000")
    logger.info("train_model terminé avec succès")
    return {"status": "ok", "model": "xgboost_single", "r2": 0.7956}

with DAG(
    dag_id="compagnon_immo_pipeline",
    default_args=default_args,
    description="Pipeline MLOps : collect → preprocess → train",
    schedule="@weekly",
    start_date=datetime(2026, 6, 1),
    catchup=False,
    tags=["immobilier", "mlops", "xgboost"],
) as dag:

    t1 = PythonOperator(
        task_id="collect_data",
        python_callable=collect_data,
    )

    t2 = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    t3 = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    t1 >> t2 >> t3
