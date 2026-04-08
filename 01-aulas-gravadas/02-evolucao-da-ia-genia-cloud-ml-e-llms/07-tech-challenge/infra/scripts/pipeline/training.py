"""
training.py — HPO Tuning Job e GA Training Job via SageMaker.
"""

import json
from datetime import datetime

from sagemaker.experiments import Run
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.tuner import HyperparameterTuner, IntegerParameter

from .config import BRT, logger, get_train_source_dir


def run_hpo_tuning_job(data_s3_uri, bucket, region, project, role_arn, session,
                       training_instance_type, max_run, max_spot_wait,
                       hpo_max_jobs=20, hpo_max_parallel_jobs=4,
                       experiment_name=None):
    """Executa HPO Tuning Job para encontrar os melhores hiperparâmetros do RF.

    Os resultados servem como warm start (ponto de partida) para o GA.
    """
    logger.info("Lançando SageMaker Hyperparameter Tuning Job...")

    checkpoint_s3_uri = f"s3://{bucket}/checkpoints/{project}-hpo"

    # Estimator em modo HPO — treina um único RF e emite métrica
    hpo_estimator = SKLearn(
        entry_point="train.py",
        source_dir=get_train_source_dir(),
        role=role_arn,
        instance_type=training_instance_type,
        instance_count=1,
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=session,
        hyperparameters={
            "mode": "hpo",
        },
        output_path=f"s3://{bucket}/models/hpo",
        base_job_name=f"{project}-hpo",
        use_spot_instances=True,
        max_run=max_run,
        max_wait=max_spot_wait,
        checkpoint_s3_uri=checkpoint_s3_uri,
        environment={"EXPERIMENT_NAME": experiment_name} if experiment_name else {},
    )

    # Espaço de hiperparâmetros a otimizar
    hyperparameter_ranges = {
        "n-estimators": IntegerParameter(50, 300),
        "max-depth": IntegerParameter(3, 20),
        "min-samples-split": IntegerParameter(2, 20),
        "min-samples-leaf": IntegerParameter(1, 10),
    }

    tuner = HyperparameterTuner(
        estimator=hpo_estimator,
        objective_metric_name="fbeta_1_5",
        hyperparameter_ranges=hyperparameter_ranges,
        metric_definitions=[
            {"Name": "fbeta_1_5", "Regex": r"fbeta_1_5=([0-9\\.]+)"},
            {"Name": "roc_auc", "Regex": r"roc_auc=([0-9\\.]+)"},
        ],
        max_jobs=hpo_max_jobs,
        max_parallel_jobs=hpo_max_parallel_jobs,
        strategy="Bayesian",
        objective_type="Maximize",
    )

    logger.info(f"HPO: {hpo_max_jobs} jobs, {hpo_max_parallel_jobs} paralelos, Bayesian")

    tuner.fit({"training": data_s3_uri})

    logger.info(f"HPO Tuning Job concluído: {tuner.latest_tuning_job.name}")

    # Extrair os top-N melhores hiperparâmetros para warm start do GA
    tuner_analytics = tuner.analytics()
    df_results = tuner_analytics.dataframe()

    # Ordenar por métrica objetiva (FinalObjectiveValue)
    df_top = df_results.nlargest(5, "FinalObjectiveValue")
    logger.info(f"Top 5 resultados do HPO:\n{df_top[['TrainingJobName', 'FinalObjectiveValue']].to_string()}")

    warm_start_params = []
    for _, row in df_top.iterrows():
        warm_start_params.append({
            "n_estimators": int(row.get("n-estimators", 100)),
            "max_depth": int(row.get("max-depth", 10)),
            "min_samples_split": int(row.get("min-samples-split", 5)),
            "min_samples_leaf": int(row.get("min-samples-leaf", 2)),
        })

    best = warm_start_params[0]
    logger.info(f"Melhor HPO: {best} (fbeta={df_top.iloc[0]['FinalObjectiveValue']:.4f})")

    return tuner, warm_start_params


def run_training_job(
    data_s3_uri, bucket, region, project, role_arn, session,
    training_instance_type, ga_pop, ga_gen,
    max_run=7200, max_spot_wait=10800,
    warm_start_params=None,
    experiment_name=None,
):
    """Lança Training Job com GA usando warm start dos melhores hiperparâmetros do HPO."""
    logger.info("Lançando SageMaker Training Job (GA + warm start do HPO)...")

    checkpoint_s3_uri = f"s3://{bucket}/checkpoints/{project}-ga"

    hyperparameters = {
        "mode": "full",
        "ga-pop": ga_pop,
        "ga-gen": ga_gen,
    }

    # Passar warm start como JSON string
    if warm_start_params:
        hyperparameters["warm-start-params"] = json.dumps(warm_start_params)
        logger.info(f"Warm start do HPO: {len(warm_start_params)} configurações")

    estimator = SKLearn(
        entry_point="train.py",
        source_dir=get_train_source_dir(),
        role=role_arn,
        instance_type=training_instance_type,
        instance_count=1,
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=session,
        hyperparameters=hyperparameters,
        output_path=f"s3://{bucket}/models",
        base_job_name=f"{project}-ga-training",
        # Managed Spot Training
        use_spot_instances=True,
        max_run=max_run,
        max_wait=max_spot_wait,
        checkpoint_s3_uri=checkpoint_s3_uri,
        environment={"EXPERIMENT_NAME": experiment_name} if experiment_name else {},
    )

    # SageMaker Experiments — associar Run ao Training Job
    if experiment_name:
        run_name = f"ga-training-{datetime.now(BRT).strftime('%Y%m%d%H%M%S')}"
        with Run(
            experiment_name=experiment_name,
            run_name=run_name,
            sagemaker_session=session,
        ) as run:
            run.log_parameter("mode", "full")
            run.log_parameter("ga_pop", ga_pop)
            run.log_parameter("ga_gen", ga_gen)
            run.log_parameter("warm_start", str(warm_start_params is not None))
            estimator.fit({"training": data_s3_uri})
    else:
        estimator.fit({"training": data_s3_uri})

    logger.info(f"Training Job concluído: {estimator.latest_training_job.name}")
    logger.info(f"Artefato do modelo: {estimator.model_data}")
    return estimator
