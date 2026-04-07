"""
metrics.py — SageMaker Experiments e métricas consolidadas.
"""

import json
from datetime import datetime

import boto3

from .config import BRT, logger


def create_experiment(session, project):
    """Cria (ou reutiliza) um Experiment no SageMaker para rastrear tudo."""
    experiment_name = f"{project}-experiment"
    logger.info(f"SageMaker Experiment: {experiment_name}")
    return experiment_name


def save_metrics(bucket, region, experiment_name, estimator, tuner,
                 warm_start_params, autopilot_job, autopilot_results,
                 use_pipeline, skip_feature_store):
    """Salva métricas consolidadas no S3 como JSON."""
    metrics = {
        "experiment_name": experiment_name,
        "pipeline_mode": use_pipeline,
        "feature_store_enabled": not skip_feature_store,
        "ga_training_job": estimator.latest_training_job.name if estimator else None,
        "ga_model_s3_uri": str(estimator.model_data) if estimator else None,
        "hpo_tuning_job": tuner.latest_tuning_job.name if tuner else None,
        "hpo_warm_start_params": warm_start_params,
        "autopilot_job": autopilot_job,
        "autopilot_results": autopilot_results,
        "timestamp": datetime.now(BRT).isoformat(),
    }
    s3 = boto3.client("s3", region_name=region)
    s3.put_object(
        Bucket=bucket,
        Key="output/training_metrics.json",
        Body=json.dumps(metrics, indent=2, default=str),
        ContentType="application/json",
    )
    logger.info(f"Métricas salvas em s3://{bucket}/output/training_metrics.json")
    return metrics
