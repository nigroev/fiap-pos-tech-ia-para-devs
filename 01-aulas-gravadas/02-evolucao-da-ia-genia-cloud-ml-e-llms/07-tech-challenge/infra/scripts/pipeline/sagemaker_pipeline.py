"""
sagemaker_pipeline.py — Criação de SageMaker Pipeline (TuningStep + TrainingStep).
"""

from sagemaker.sklearn.estimator import SKLearn
from sagemaker.tuner import HyperparameterTuner, IntegerParameter
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, TuningStep

from .config import logger, get_train_source_dir


def create_sagemaker_pipeline(
    bucket, region, project, role_arn, session,
    training_instance_type, endpoint_instance_type,
    ga_pop, ga_gen, max_run, max_spot_wait,
    hpo_max_jobs, hpo_max_parallel_jobs,
    data_s3_uri, experiment_name,
):
    """Cria um SageMaker Pipeline com TuningStep + TrainingStep.

    O Pipeline gerencia a execução sequencial:
      1. TuningStep — HPO Bayesian (otimiza hiperparâmetros do RF)
      2. TrainingStep — GA Training com warm start (pós-HPO manual)

    Nota: O Autopilot roda fora do Pipeline (assíncrono) porque não é
    suportado como step nativo. Feature Store ingestion também ocorre
    antes do Pipeline (dados precisam estar no S3 antes da execução).
    """
    logger.info("Criando SageMaker Pipeline...")

    # Parâmetros do Pipeline (configuráveis em runtime)
    param_instance_type = ParameterString(
        name="TrainingInstanceType", default_value=training_instance_type
    )
    param_ga_pop = ParameterInteger(name="GAPop", default_value=ga_pop)
    param_ga_gen = ParameterInteger(name="GAGen", default_value=ga_gen)
    param_hpo_max_jobs = ParameterInteger(name="HPOMaxJobs", default_value=hpo_max_jobs)
    param_hpo_max_parallel = ParameterInteger(
        name="HPOMaxParallelJobs", default_value=hpo_max_parallel_jobs
    )

    checkpoint_s3_uri = f"s3://{bucket}/checkpoints/{project}-hpo"

    # --- Step 1: HPO Tuning ---
    hpo_estimator = SKLearn(
        entry_point="train.py",
        source_dir=get_train_source_dir(),
        role=role_arn,
        instance_type=param_instance_type,
        instance_count=1,
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=session,
        hyperparameters={"mode": "hpo"},
        output_path=f"s3://{bucket}/models/hpo",
        base_job_name=f"{project}-hpo",
        use_spot_instances=True,
        max_run=max_run,
        max_wait=max_spot_wait,
        checkpoint_s3_uri=checkpoint_s3_uri,
        environment={"EXPERIMENT_NAME": experiment_name} if experiment_name else {},
    )

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
        max_jobs=param_hpo_max_jobs,
        max_parallel_jobs=param_hpo_max_parallel,
        strategy="Bayesian",
        objective_type="Maximize",
    )

    tuning_step = TuningStep(
        name=f"{project}-HPO-Tuning",
        step_args=tuner.fit({"training": data_s3_uri}),
    )

    # --- Step 2: GA Training (warm start é passado manualmente após HPO) ---
    ga_estimator = SKLearn(
        entry_point="train.py",
        source_dir=get_train_source_dir(),
        role=role_arn,
        instance_type=param_instance_type,
        instance_count=1,
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=session,
        hyperparameters={
            "mode": "full",
            "ga-pop": param_ga_pop,
            "ga-gen": param_ga_gen,
        },
        output_path=f"s3://{bucket}/models",
        base_job_name=f"{project}-ga-training",
        use_spot_instances=True,
        max_run=max_run,
        max_wait=max_spot_wait,
        checkpoint_s3_uri=f"s3://{bucket}/checkpoints/{project}-ga",
        environment={"EXPERIMENT_NAME": experiment_name} if experiment_name else {},
    )

    training_step = TrainingStep(
        name=f"{project}-GA-Training",
        step_args=ga_estimator.fit({"training": data_s3_uri}),
        depends_on=[tuning_step],
    )

    # --- Montar Pipeline ---
    pipeline_name = f"{project}-pipeline"
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            param_instance_type,
            param_ga_pop,
            param_ga_gen,
            param_hpo_max_jobs,
            param_hpo_max_parallel,
        ],
        steps=[tuning_step, training_step],
        sagemaker_session=session,
    )

    logger.info(f"Pipeline criado: {pipeline_name}")
    return pipeline, ga_estimator
