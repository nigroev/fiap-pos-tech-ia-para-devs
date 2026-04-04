"""
pipeline — Módulos do orquestrador de treinamento e deploy via SageMaker
"""

from .config import BRT, logger, get_train_source_dir
from .data_ingestion import load_nhanes_data, preprocess_data, upload_dataset_to_s3
from .feature_store import (
    create_or_get_feature_group,
    ingest_features,
    read_features_from_store,
)
from .autopilot import run_autopilot_job, wait_for_autopilot
from .training import run_hpo_tuning_job, run_training_job
from .sagemaker_pipeline import create_sagemaker_pipeline
from .deployment import deploy_sagemaker_endpoint, deploy_autopilot_endpoint
from .metrics import create_experiment, save_metrics
