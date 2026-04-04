"""
train_and_deploy.py — Orquestrador de treinamento e deploy via SageMaker
=========================================================================
Projeto: AVC Stroke Prediction (NHANES)

Este script executa:
  1. Download e merge dos dados NHANES (mesma lógica do notebook)
  2. Pré-processamento e feature engineering
  3. SageMaker Feature Store — persistência das features offline
  4. SageMaker Autopilot — AutoML (feature engineering, tuning, training, ranking)
  5. SageMaker Hyperparameter Tuning Job — otimiza hiperparâmetros do RF
  6. Training Job com GA usando warm start dos melhores hiperparâmetros do HPO
  7. Comparação dos resultados e deploy do melhor modelo

Integrações SageMaker:
  - Experiments: Rastreamento de todos os experimentos com Run/Experiment
  - Feature Store: Feature Group offline para features pré-processadas
  - Pipelines: Orquestração via SageMaker Pipelines (opcional com --use-pipeline)
"""

import argparse
import json
import logging
import math
import os
import shutil
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone, timedelta

# Fuso horário de São Paulo (UTC-3)
BRT = timezone(timedelta(hours=-3))

import boto3
import numpy as np
import pandas as pd
import sagemaker
from sagemaker import AutoML, AutoMLInput
from sagemaker.experiments import Run
from sagemaker.feature_store.feature_definition import (
    FeatureDefinition,
    FeatureTypeEnum,
)
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.tuner import HyperparameterTuner, IntegerParameter
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import TrainingStep, TuningStep

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.Formatter.converter = lambda *args: datetime.now(BRT).timetuple()
logger = logging.getLogger(__name__)

# Diretório temporário com apenas train.py (sem requirements.txt pesado)
# O SKLearn container já possui sklearn, pandas, numpy, joblib.
_TRAIN_SOURCE_DIR = None


def _get_train_source_dir():
    """Retorna diretório temporário contendo apenas train.py.

    Evita que o SageMaker instale dependências pesadas do requirements.txt
    (sagemaker[feature-store], pyarrow, etc.) dentro do container de treinamento.
    """
    global _TRAIN_SOURCE_DIR
    if _TRAIN_SOURCE_DIR is None:
        _TRAIN_SOURCE_DIR = tempfile.mkdtemp(prefix="sagemaker_train_src_")
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
        shutil.copy2(src, os.path.join(_TRAIN_SOURCE_DIR, "train.py"))
        logger.info(f"Training source dir criado: {_TRAIN_SOURCE_DIR}")
    return _TRAIN_SOURCE_DIR


# ==============================================================================
# 1. COLETA DE DADOS NHANES
# ==============================================================================


def _s3_key_for_xpt(name, letter):
    """Retorna a chave S3 para um arquivo XPT do NHANES."""
    return f"data/raw/nhanes/{name}{letter}.parquet"


def _s3_object_exists(s3, bucket, key):
    """Verifica se um objeto existe no S3."""
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


def load_nhanes_data(bucket, region):
    """Carrega e faz merge dos dados NHANES de múltiplos ciclos.

    Para cada arquivo XPT de cada ciclo:
      1. Verifica se já existe em cache no S3 (parquet).
      2. Se existir, lê direto do S3.
      3. Se não existir, baixa da API do CDC, salva no S3 como parquet e usa.
    """
    logger.info("Iniciando carga dos dados NHANES (com cache S3)...")

    s3 = boto3.client("s3", region_name=region)

    cycle_map = {
        "2005-2006": "_D",
        "2007-2008": "_E",
        "2009-2010": "_F",
        "2011-2012": "_G",
        "2013-2014": "_H",
        "2015-2016": "_I",
        "2017-2018": "_J",
    }

    modules = {
        "demo": "DEMO{}.XPT",
        "bpx": "BPX{}.XPT",
        "bpq": "BPQ{}.XPT",
        "ocq": "OCQ{}.XPT",
        "ghb": "GHB{}.XPT",
        "bmx": "BMX{}.XPT",
        "smq": "SMQ{}.XPT",
        "mcq": "MCQ{}.XPT",
    }

    dfs = {name: [] for name in modules}

    for cycle, letter in cycle_map.items():
        year = cycle.split("-")[0]
        base_url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/"
        for name, file_template in modules.items():
            file = file_template.format(letter)
            s3_key = _s3_key_for_xpt(name, letter)

            # 1) Tentar ler do cache S3
            if _s3_object_exists(s3, bucket, s3_key):
                logger.info(f"  [S3 cache] {file} ({cycle}) -> s3://{bucket}/{s3_key}")
                with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                    s3.download_file(bucket, s3_key, tmp.name)
                    df_part = pd.read_parquet(tmp.name)
                    os.unlink(tmp.name)
                dfs[name].append(df_part)
                continue

            # 2) Baixar da API do CDC
            url = base_url + file
            logger.info(f"  [CDC API] {file} ({cycle}) -> baixando de {url}")
            try:
                df_part = pd.read_sas(url)
            except Exception as e:
                logger.warning(f"  Falha ao carregar {file}: {e}")
                continue

            # 3) Salvar no S3 como parquet para reusar nas próximas execuções
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                df_part.to_parquet(tmp.name, index=False)
                s3.upload_file(tmp.name, bucket, s3_key)
                os.unlink(tmp.name)
            logger.info(f"  [S3 upload] {file} -> s3://{bucket}/{s3_key}")

            dfs[name].append(df_part)

    # Concatenar ciclos por módulo
    concat = {name: pd.concat(frames, ignore_index=True) for name, frames in dfs.items() if frames}

    # Merge em SEQN
    df = concat["demo"]
    for name in ["bpx", "bpq", "ocq", "ghb", "bmx", "smq", "mcq"]:
        if name in concat:
            df = df.merge(concat[name], on="SEQN", how="left")

    logger.info(f"Dataset combinado: {df.shape[0]} linhas, {df.shape[1]} colunas")
    return df


# ==============================================================================
# 2. PRÉ-PROCESSAMENTO
# ==============================================================================


def preprocess_data(df):
    """Seleciona features, trata missing values e binariza variáveis."""
    logger.info("Pré-processando dados...")

    # Renomear colunas de interesse
    rename = {
        "RIDAGEYR": "RIDAGEYR_age",
        "BPXSY1": "BPXSY1_sbp",
        "LBXGH": "LBXGH_hba1c",
        "BMXBMI": "BMXBMI_bmi",
        "RIAGENDR": "RIAGENDR_gender",
        "DMDMARTL": "DMDMARTL_marital",
        "BPQ020": "BPQ020_high_bp",
        "MCQ160B": "MCQ160B_chf",
        "OCQ260": "OCQ260_occupation",
        "SMQ020": "SMQ020_smoking",
        "MCQ160F": "MCQ160F_stroke",
    }
    for old, new in rename.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    # Manter apenas colunas de interesse
    cols_of_interest = list(rename.values())
    available_cols = [c for c in cols_of_interest if c in df.columns]
    df = df[available_cols].copy()

    # Binarizar features categóricas (1=Yes, 2=No -> 1, 0)
    categorical_features = [
        "RIAGENDR_gender",
        "BPQ020_high_bp",
        "MCQ160B_chf",
        "SMQ020_smoking",
    ]
    for col in categorical_features:
        if col in df.columns:
            df = df[df[col].isin([1, 2])].copy()
            binary_col = f"{col}_bin"
            df[binary_col] = df[col].map({1: 1, 2: 0})
            df = df.drop(columns=[col])

    # Marital status -> binary (já/nunca casou)
    if "DMDMARTL_marital" in df.columns:
        df = df[df["DMDMARTL_marital"].isin([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])].copy()
        df["DMDMARTL_married_bin"] = (df["DMDMARTL_marital"] != 5.0).astype(int)
        df = df.drop(columns=["DMDMARTL_marital"])

    # Target: stroke binary
    target_col = "MCQ160F_stroke"
    if target_col in df.columns:
        df = df[df[target_col].isin([1, 2])].copy()
        df["MCQ160F_stroke_bin"] = df[target_col].map({1: 1, 2: 0})
        df = df.drop(columns=[target_col])

    logger.info(f"Dataset pré-processado: {df.shape[0]} linhas, {df.shape[1]} colunas")
    logger.info(f"Distribuição do target:\n{df['MCQ160F_stroke_bin'].value_counts()}")
    return df


# ==============================================================================
# 3. UPLOAD DO DATASET PARA S3
# ==============================================================================


def upload_dataset_to_s3(df, bucket, region):
    """Salva o dataset processado no S3 para ser usado pelo Training Job."""
    logger.info("Salvando dataset processado no S3...")
    s3 = boto3.client("s3", region_name=region)

    s3_key = "data/nhanes_stroke_processed.csv"
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        df.to_csv(tmp.name, index=False)
        s3.upload_file(tmp.name, bucket, s3_key)
        os.unlink(tmp.name)

    s3_uri = f"s3://{bucket}/{s3_key}"
    logger.info(f"Dataset salvo em {s3_uri}")
    return s3_uri


# ==============================================================================
# 3.1 SAGEMAKER FEATURE STORE — Persistência de features offline
# ==============================================================================

FEATURE_GROUP_NAME_PREFIX = "nhanes-stroke-features"

FEATURE_DEFINITIONS = [
    FeatureDefinition("record_id", FeatureTypeEnum.STRING),
    FeatureDefinition("event_time", FeatureTypeEnum.STRING),
    FeatureDefinition("RIDAGEYR_age", FeatureTypeEnum.FRACTIONAL),
    FeatureDefinition("BPXSY1_sbp", FeatureTypeEnum.FRACTIONAL),
    FeatureDefinition("LBXGH_hba1c", FeatureTypeEnum.FRACTIONAL),
    FeatureDefinition("BMXBMI_bmi", FeatureTypeEnum.FRACTIONAL),
    FeatureDefinition("RIAGENDR_gender_bin", FeatureTypeEnum.INTEGRAL),
    FeatureDefinition("DMDMARTL_married_bin", FeatureTypeEnum.INTEGRAL),
    FeatureDefinition("BPQ020_high_bp_bin", FeatureTypeEnum.INTEGRAL),
    FeatureDefinition("MCQ160B_chf_bin", FeatureTypeEnum.INTEGRAL),
    FeatureDefinition("OCQ260_occupation", FeatureTypeEnum.FRACTIONAL),
    FeatureDefinition("SMQ020_smoking_bin", FeatureTypeEnum.INTEGRAL),
    FeatureDefinition("MCQ160F_stroke_bin", FeatureTypeEnum.INTEGRAL),
]


def create_or_get_feature_group(session, role_arn, bucket, project):
    """Cria (ou obtém) o Feature Group para as features pré-processadas."""
    feature_group_name = f"{project}-{FEATURE_GROUP_NAME_PREFIX}"

    fg = FeatureGroup(name=feature_group_name, sagemaker_session=session)

    sm_client = session.sagemaker_client
    try:
        sm_client.describe_feature_group(FeatureGroupName=feature_group_name)
        logger.info(f"Feature Group já existe: {feature_group_name}")
        return fg, feature_group_name
    except sm_client.exceptions.ResourceNotFound:
        pass

    logger.info(f"Criando Feature Group: {feature_group_name}")

    fg.feature_definitions = FEATURE_DEFINITIONS

    fg.create(
        s3_uri=f"s3://{bucket}/feature-store/",
        record_identifier_name="record_id",
        event_time_feature_name="event_time",
        role_arn=role_arn,
        enable_online_store=False,
    )

    # Aguardar criação
    while True:
        status = sm_client.describe_feature_group(
            FeatureGroupName=feature_group_name
        )["FeatureGroupStatus"]
        if status == "Created":
            break
        elif status == "CreateFailed":
            raise RuntimeError(f"Falha ao criar Feature Group: {feature_group_name}")
        logger.info(f"  Feature Group status: {status}...")
        time.sleep(10)

    logger.info(f"Feature Group criado: {feature_group_name}")
    return fg, feature_group_name


def ingest_features(df, feature_group, session):
    """Ingere os dados pré-processados no Feature Store."""
    logger.info(f"Ingerindo {len(df)} registros no Feature Store...")

    # Adicionar colunas obrigatórias
    df_fs = df.copy()
    df_fs["record_id"] = [str(uuid.uuid4()) for _ in range(len(df_fs))]
    df_fs["event_time"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Converter tipos para compatibilidade com Feature Store
    int_cols = [
        "RIAGENDR_gender_bin", "DMDMARTL_married_bin", "BPQ020_high_bp_bin",
        "MCQ160B_chf_bin", "SMQ020_smoking_bin", "MCQ160F_stroke_bin",
    ]
    float_cols = [
        "RIDAGEYR_age", "BPXSY1_sbp", "LBXGH_hba1c", "BMXBMI_bmi",
        "OCQ260_occupation",
    ]
    for col in int_cols:
        if col in df_fs.columns:
            df_fs[col] = df_fs[col].fillna(0).astype(int)
    for col in float_cols:
        if col in df_fs.columns:
            df_fs[col] = df_fs[col].astype(float)

    feature_group.ingest(data_frame=df_fs, max_workers=4, wait=True)
    logger.info(f"Feature Store: {len(df_fs)} registros ingeridos com sucesso")
    return df_fs


def read_features_from_store(feature_group_name, session, bucket, region):
    """Lê features do Feature Store offline (S3) via Athena query.

    Retorna o S3 URI do dataset resultante para uso no Training Job.
    """
    logger.info(f"Lendo features do Feature Store: {feature_group_name}")

    sm_client = session.sagemaker_client
    fg_desc = sm_client.describe_feature_group(FeatureGroupName=feature_group_name)

    # O offline store salva em formato parquet no S3
    offline_config = fg_desc.get("OfflineStoreConfig", {})
    s3_prefix = offline_config.get("S3StorageConfig", {}).get("ResolvedOutputS3Uri", "")

    if s3_prefix:
        logger.info(f"Feature Store offline S3: {s3_prefix}")
        return s3_prefix

    # Fallback: usar o dataset CSV já no S3
    logger.warning("Feature Store offline URI não encontrada, usando dataset CSV")
    return f"s3://{bucket}/data/nhanes_stroke_processed.csv"


# ==============================================================================
# 4. SAGEMAKER EXPERIMENTS — Rastreamento de experimentos
# ==============================================================================


def create_experiment(session, project):
    """Cria (ou reutiliza) um Experiment no SageMaker para rastrear tudo."""
    experiment_name = f"{project}-experiment"
    logger.info(f"SageMaker Experiment: {experiment_name}")
    return experiment_name


# ==============================================================================
# 5. SAGEMAKER AUTOPILOT — AutoML
# ==============================================================================


def run_autopilot_job(data_s3_uri, bucket, region, project, role_arn, session,
                      max_candidates=20):
    """Lança um job SageMaker Autopilot (AutoML) de forma assíncrona.

    O Autopilot executa automaticamente:
      - Feature engineering
      - Seleção e tuning de algoritmos
      - Treinamento e ranking de modelos candidatos
    """
    logger.info("Lançando SageMaker Autopilot (AutoML)...")

    timestamp = datetime.now(BRT).strftime("%m%d%H%M%S")
    # AutoML job name: max 32 chars, pattern [a-zA-Z0-9](-*[a-zA-Z0-9]){0,31}
    job_name = f"avc-ap-{timestamp}"

    auto_ml = AutoML(
        role=role_arn,
        sagemaker_session=session,
        target_attribute_name="MCQ160F_stroke_bin",
        output_path=f"s3://{bucket}/autopilot-output",
        problem_type="BinaryClassification",
        max_candidates=max_candidates,
        job_objective={"MetricName": "F1"},
    )

    # Lançar de forma assíncrona (wait=False) para rodar em paralelo com HPO+GA
    sm_client = session.sagemaker_client
    try:
        auto_ml.fit(
            inputs=data_s3_uri,
            job_name=job_name,
            wait=False,
        )
        logger.info(f"Autopilot job lançado: {job_name} (max_candidates={max_candidates})")
    except sm_client.exceptions.ResourceLimitExceeded:
        logger.warning(
            "ResourceLimitExceeded: limite de AutoML jobs concorrentes atingido. "
            "Procurando job em andamento para reutilizar..."
        )
        # Find the most recent InProgress AutoML job to reuse
        paginator = sm_client.get_paginator("list_auto_ml_jobs")
        running_job = None
        for page in paginator.paginate(StatusEquals="InProgress", SortBy="CreationTime", SortOrder="Descending"):
            jobs = page.get("AutoMLJobSummaries", [])
            if jobs:
                running_job = jobs[0]["AutoMLJobName"]
                break
        if running_job:
            job_name = running_job
            logger.info(f"Reutilizando Autopilot job em andamento: {job_name}")
        else:
            logger.error("Nenhum AutoML job em andamento encontrado. Pulando Autopilot.")
            return None, None

    return auto_ml, job_name


def wait_for_autopilot(auto_ml, job_name, session, timeout_minutes=0):
    """Aguarda a conclusão do Autopilot e retorna os resultados.

    Se timeout_minutes > 0, desiste após esse tempo e retorna None.
    """
    logger.info(f"Aguardando conclusão do Autopilot job: {job_name}...")
    if timeout_minutes > 0:
        logger.info(f"  Timeout configurado: {timeout_minutes} minutos")

    sm_client = session.sagemaker_client
    start_wait = time.time()

    while True:
        response = sm_client.describe_auto_ml_job(AutoMLJobName=job_name)
        status = response["AutoMLJobStatus"]
        secondary = response.get("AutoMLJobSecondaryStatus", "")
        elapsed_min = (time.time() - start_wait) / 60
        logger.info(f"  Autopilot status: {status} ({secondary}) [{elapsed_min:.0f}min]")

        if status == "Completed":
            break
        elif status in ("Failed", "Stopped"):
            logger.warning(f"Autopilot job {status}: {response.get('FailureReason', 'N/A')}")
            return None

        # Timeout — seguir sem Autopilot
        if timeout_minutes > 0 and elapsed_min >= timeout_minutes:
            logger.warning(
                f"Autopilot timeout atingido ({timeout_minutes}min). "
                f"Status atual: {status} ({secondary}). Continuando sem resultado do Autopilot."
            )
            return None

        time.sleep(60)

    # Obter o melhor candidato
    best = auto_ml.best_candidate(job_name=job_name)
    best_metric = best.get("FinalAutoMLJobObjectiveMetric", {})

    logger.info(f"Autopilot concluído!")
    logger.info(f"  Melhor candidato: {best.get('CandidateName', 'N/A')}")
    logger.info(f"  Métrica ({best_metric.get('MetricName', 'F1')}): "
                f"{best_metric.get('Value', 'N/A')}")

    return {
        "candidate_name": best.get("CandidateName"),
        "metric_name": best_metric.get("MetricName", "F1"),
        "metric_value": best_metric.get("Value", 0.0),
        "inference_containers": best.get("InferenceContainers", []),
    }


# ==============================================================================
# 6. SAGEMAKER HYPERPARAMETER TUNING JOB
# ==============================================================================


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
        source_dir=_get_train_source_dir(),
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


# ==============================================================================
# 7. TREINAMENTO GA COM WARM START VIA SAGEMAKER TRAINING JOB
# ==============================================================================


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
        source_dir=_get_train_source_dir(),
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


# ==============================================================================
# 8. DEPLOY DO ENDPOINT DE INFERÊNCIA
# ==============================================================================


def _cleanup_endpoint(sm_client, endpoint_name):
    """Remove endpoint, endpoint-config e models antigos para evitar conflitos."""
    # Deletar endpoint
    try:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        logger.info(f"Deletando endpoint antigo '{endpoint_name}'...")
        sm_client.delete_endpoint(EndpointName=endpoint_name)
        # Aguardar deleção
        import botocore.exceptions
        for _ in range(30):
            try:
                sm_client.describe_endpoint(EndpointName=endpoint_name)
                time.sleep(10)
            except sm_client.exceptions.ClientError:
                break
        logger.info(f"Endpoint '{endpoint_name}' deletado.")
    except sm_client.exceptions.ClientError:
        pass

    # Deletar endpoint-config
    try:
        sm_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        logger.info(f"Endpoint config '{endpoint_name}' deletado.")
    except sm_client.exceptions.ClientError:
        pass


def _wait_for_endpoint(sm_client, endpoint_name, timeout=900, poll_interval=30):
    """Aguarda endpoint ficar InService com timeout configurável (default 15min)."""
    elapsed = 0
    while elapsed < timeout:
        resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = resp["EndpointStatus"]
        if status == "InService":
            return True
        if status == "Failed":
            reason = resp.get("FailureReason", "desconhecido")
            logger.warning(f"Endpoint '{endpoint_name}' falhou: {reason}")
            return False
        logger.info(f"  Endpoint '{endpoint_name}': {status} ({elapsed}s/{timeout}s)")
        time.sleep(poll_interval)
        elapsed += poll_interval
    logger.warning(f"Timeout ({timeout}s) aguardando endpoint '{endpoint_name}'")
    return False


def deploy_sagemaker_endpoint(estimator, project, endpoint_instance_type, max_retries=2):
    """Faz deploy do modelo treinado (GA) como endpoint de inferência com retry."""
    logger.info("Fazendo deploy do endpoint no SageMaker...")

    endpoint_name = f"{project}-endpoint"
    sm_client = estimator.sagemaker_session.sagemaker_client

    # Verificar que inference_src existe
    inference_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference_src")
    if not os.path.isfile(os.path.join(inference_src, "inference.py")):
        raise FileNotFoundError(f"inference.py não encontrado em {inference_src}")

    for attempt in range(1, max_retries + 1):
        # Limpar recursos antigos para evitar conflito de nomes
        _cleanup_endpoint(sm_client, endpoint_name)

        logger.info(f"Criando endpoint '{endpoint_name}' (tentativa {attempt}/{max_retries}, instance: {endpoint_instance_type})...")
        logger.info(f"Model data: {estimator.model_data}")

        estimator.deploy(
            initial_instance_count=1,
            instance_type=endpoint_instance_type,
            endpoint_name=endpoint_name,
            entry_point="inference.py",
            source_dir=inference_src,
            wait=False,
        )

        if _wait_for_endpoint(sm_client, endpoint_name, timeout=900):
            logger.info(f"Endpoint ativo: {endpoint_name}")
            return endpoint_name

        if attempt < max_retries:
            logger.warning(f"Health check falhou. Retry {attempt + 1}/{max_retries}...")
        else:
            logger.error(f"Endpoint falhou após {max_retries} tentativas.")
            raise RuntimeError(f"Endpoint '{endpoint_name}' não passou no health check após {max_retries} tentativas")

    return endpoint_name


def deploy_autopilot_endpoint(auto_ml, job_name, session, project, endpoint_instance_type, max_retries=2):
    """Faz deploy do melhor modelo do Autopilot como endpoint separado com retry."""
    logger.info("Fazendo deploy do endpoint Autopilot...")

    endpoint_name = f"{project}-autopilot-endpoint"
    sm_client = session.sagemaker_client

    for attempt in range(1, max_retries + 1):
        _cleanup_endpoint(sm_client, endpoint_name)

        logger.info(f"Criando endpoint Autopilot '{endpoint_name}' (tentativa {attempt}/{max_retries})...")

        auto_ml.deploy(
            initial_instance_count=1,
            instance_type=endpoint_instance_type,
            endpoint_name=endpoint_name,
            sagemaker_session=session,
            wait=False,
        )

        if _wait_for_endpoint(sm_client, endpoint_name, timeout=900):
            logger.info(f"Endpoint Autopilot ativo: {endpoint_name}")
            return endpoint_name

        if attempt < max_retries:
            logger.warning(f"Autopilot health check falhou. Retry {attempt + 1}/{max_retries}...")
        else:
            logger.error(f"Endpoint Autopilot falhou após {max_retries} tentativas.")
            raise RuntimeError(f"Endpoint Autopilot '{endpoint_name}' não passou no health check após {max_retries} tentativas")

    return endpoint_name


# ==============================================================================
# 9. SAGEMAKER PIPELINES — Orquestração como Pipeline gerenciado
# ==============================================================================


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
        source_dir=_get_train_source_dir(),
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
        source_dir=_get_train_source_dir(),
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


# ==============================================================================
# MAIN — Orquestra o pipeline completo
# ==============================================================================

_TOTAL_PHASES = 8
_phase_start_time: dict = {}


def _phase(n: int, label: str, end: bool = False) -> None:
    """Loga um banner de início ou fim de fase com timestamp e contador."""
    bar = "─" * 56
    ts = datetime.now(BRT).strftime("%H:%M:%S")
    if end:
        elapsed = ""
        if n in _phase_start_time:
            secs = time.time() - _phase_start_time[n]
            elapsed = f"  ({math.floor(secs // 60)}m{int(secs % 60)}s)"
        logger.info(f"╰{bar}╯")
        logger.info(f"  ✓ FASE {n}/{_TOTAL_PHASES} concluída{elapsed}  [{ts}]")
        logger.info("")
    else:
        _phase_start_time[n] = time.time()
        logger.info("")
        logger.info(f"╭{bar}╮")
        logger.info(f"  ▶ FASE {n}/{_TOTAL_PHASES}: {label}  [{ts}]")
        logger.info(f"╰{bar}╯")


def main():
    parser = argparse.ArgumentParser(
        description="Orquestrador de treinamento e deploy AVC/Stroke via SageMaker"
    )
    parser.add_argument("--bucket", required=True, help="Nome do bucket S3")
    parser.add_argument("--region", required=True, help="Região AWS")
    parser.add_argument("--project", required=True, help="Nome do projeto")
    parser.add_argument("--skip-deploy", action="store_true", help="Pular deploy dos endpoints")
    parser.add_argument("--skip-autopilot", action="store_true", help="Pular Autopilot AutoML")
    parser.add_argument("--skip-feature-store", action="store_true", help="Pular Feature Store")
    parser.add_argument(
        "--use-pipeline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Usar SageMaker Pipelines (padrão: ativado). Use --no-use-pipeline para modo manual.",
    )
    parser.add_argument("--ga-pop", type=int, default=10, help="Tamanho da população do GA")
    parser.add_argument("--ga-gen", type=int, default=5, help="Número de gerações do GA")
    parser.add_argument(
        "--training-instance-type", type=str, default="ml.m5.large",
        help="Tipo de instância para Training Jobs"
    )
    parser.add_argument(
        "--endpoint-instance-type", type=str, default="ml.t2.medium",
        help="Tipo de instância para Endpoints de inferência"
    )
    parser.add_argument(
        "--max-run", type=int, default=1800,
        help="Tempo máximo de execução por Training Job (segundos)"
    )
    parser.add_argument(
        "--max-spot-wait", type=int, default=3600,
        help="Tempo máximo de espera por instância spot (segundos, >= max-run)"
    )
    parser.add_argument(
        "--hpo-max-jobs", type=int, default=6,
        help="Número máximo de jobs no HPO Tuning"
    )
    parser.add_argument(
        "--hpo-max-parallel-jobs", type=int, default=3,
        help="Número máximo de jobs paralelos no HPO Tuning"
    )
    parser.add_argument(
        "--autopilot-max-candidates", type=int, default=5,
        help="Número máximo de modelos candidatos do Autopilot"
    )
    parser.add_argument(
        "--autopilot-timeout", type=int, default=0,
        help="Timeout em minutos para aguardar Autopilot (0=sem limite)"
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help=(
            "Modo desenvolvimento: cada fase ≤25min. "
            "Equivale a: --hpo-max-jobs 3 --hpo-max-parallel-jobs 3 "
            "--ga-pop 4 --ga-gen 3 --autopilot-max-candidates 5 "
            "--autopilot-timeout 25 --max-run 900 --max-spot-wait 1200"
        ),
    )
    args = parser.parse_args()

    # Aplicar overrides do modo --dev antes de qualquer uso dos args
    if args.dev:
        args.hpo_max_jobs = 3          # 3 jobs HPO — 1 batch paralelo ~5min
        args.hpo_max_parallel_jobs = 3
        args.ga_pop = 4               # GA leve: 4*3*3cv = 36 fits (~3min)
        args.ga_gen = 3
        args.autopilot_max_candidates = 3
        args.autopilot_timeout = 20        # máx 20min, depois segue sem ele
        args.max_run = 600            # 10 min por job
        args.max_spot_wait = 900      # 15 min de espera spot

    bar = "═" * 58
    logger.info(f"╔{bar}╗")
    logger.info("║  Orquestrador — AVC Stroke Prediction (SageMaker)       ║")
    logger.info("║  Experiments · Feature Store · Pipelines                ║")
    logger.info("║  Autopilot · HPO · GA                                   ║")
    if args.dev:
        logger.info("║  *** MODO DEV — parâmetros reduzidos para validação ***  ║")
    logger.info(f"╚{bar}╝")
    logger.info(f"  Início: {datetime.now(BRT).strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Bucket: {args.bucket} | Região: {args.region} | Projeto: {args.project}")
    logger.info(f"  Modo: {'Pipeline' if args.use_pipeline else 'Manual'} | "
                f"Autopilot: {'sim' if not args.skip_autopilot else 'não'} | "
                f"FeatureStore: {'sim' if not args.skip_feature_store else 'não'}"
                + (" | [DEV]" if args.dev else ""))
    if args.dev:
        logger.info(
            f"  [DEV] HPO jobs={args.hpo_max_jobs} parallel={args.hpo_max_parallel_jobs} | "
            f"GA pop={args.ga_pop} gen={args.ga_gen} | "
            f"max-run={args.max_run}s | "
            f"autopilot-timeout={args.autopilot_timeout}min"
        )

    # ================================================================
    # 1. Setup — Role, Session, Experiment
    # ================================================================
    _phase(1, "Setup — Role, Session, SageMaker Experiment")
    iam = boto3.client("iam", region_name=args.region)
    role_name = f"{args.project}-dev-sagemaker-role"
    role_arn = iam.get_role(RoleName=role_name)["Role"]["Arn"]
    logger.info(f"Usando role: {role_arn}")

    session = sagemaker.Session(
        boto_session=boto3.Session(region_name=args.region),
        default_bucket=args.bucket,
    )
    pipeline_session = PipelineSession(
        boto_session=boto3.Session(region_name=args.region),
        default_bucket=args.bucket,
    )

    # Criar Experiment para rastrear todos os jobs
    experiment_name = create_experiment(session, args.project)
    _phase(1, "", end=True)

    # ================================================================
    # 2. Carregar e pré-processar dados
    # ================================================================
    _phase(2, "Ingestão e pré-processamento de dados (NHANES)")
    df_raw = load_nhanes_data(bucket=args.bucket, region=args.region)
    df = preprocess_data(df_raw)

    # ================================================================
    # 3. Feature Store — persistir features pré-processadas
    # ================================================================
    _phase(3, "Feature Store — ingestão de features pré-processadas")
    if not args.skip_feature_store:
        fg, fg_name = create_or_get_feature_group(
            session=session,
            role_arn=role_arn,
            bucket=args.bucket,
            project=args.project,
        )
        ingest_features(df, fg, session)
        logger.info(f"Features ingeridas no Feature Store: {fg_name}")
    else:
        logger.info("Feature Store ignorado (--skip-feature-store).")
    _phase(3, "", end=True)

    # Upload CSV para S3 (usado como canal de treinamento)
    data_s3_uri = upload_dataset_to_s3(df, args.bucket, args.region)

    # Log de dados no Experiment
    _phase(2, "", end=True)
    with Run(
        experiment_name=experiment_name,
        run_name=f"data-prep-{datetime.now(BRT).strftime('%Y%m%d%H%M%S')}",
        sagemaker_session=session,
    ) as run:
        run.log_parameter("dataset_rows", df.shape[0])
        run.log_parameter("dataset_cols", df.shape[1])
        run.log_parameter("positive_rate", float(df["MCQ160F_stroke_bin"].mean()))
        run.log_parameter("feature_store_enabled", str(not args.skip_feature_store))

    # ================================================================
    # 4. Lançar Autopilot AutoML (assíncrono — roda em paralelo)
    # ================================================================
    _phase(4, "Autopilot AutoML — lançamento assíncrono")
    auto_ml = None
    autopilot_job = None
    if not args.skip_autopilot:
        result = run_autopilot_job(
            data_s3_uri=data_s3_uri,
            bucket=args.bucket,
            region=args.region,
            project=args.project,
            role_arn=role_arn,
            session=session,
            max_candidates=args.autopilot_max_candidates,
        )
        if result is not None and result != (None, None):
            auto_ml, autopilot_job = result
        else:
            logger.warning("Autopilot não pôde ser lançado; continuando sem ele.")
    else:
        logger.info("Autopilot ignorado (--skip-autopilot).")
    _phase(4, "", end=True)

    # ================================================================
    # 5. Execução — Pipeline ou manual
    # ================================================================
    _phase(5, f"Treinamento — {'SageMaker Pipelines (HPO + GA)' if args.use_pipeline else 'Modo manual (HPO + GA)'}")
    if args.use_pipeline:
        # ----- Modo Pipeline -----
        logger.info("Executando via SageMaker Pipelines...")

        pipeline, ga_estimator = create_sagemaker_pipeline(
            bucket=args.bucket,
            region=args.region,
            project=args.project,
            role_arn=role_arn,
            session=pipeline_session,
            training_instance_type=args.training_instance_type,
            endpoint_instance_type=args.endpoint_instance_type,
            ga_pop=args.ga_pop,
            ga_gen=args.ga_gen,
            max_run=args.max_run,
            max_spot_wait=args.max_spot_wait,
            hpo_max_jobs=args.hpo_max_jobs,
            hpo_max_parallel_jobs=args.hpo_max_parallel_jobs,
            data_s3_uri=data_s3_uri,
            experiment_name=experiment_name,
        )

        # Criar/atualizar e executar o Pipeline
        pipeline.upsert(role_arn=role_arn)
        execution = pipeline.start(
            execution_display_name=f"exec-{datetime.now(BRT).strftime('%Y%m%d%H%M%S')}",
            # Passando parâmetros explicitamente garante que dev mode seja respeitado
            # independente da definição em cache no SageMaker
            parameters={
                "TrainingInstanceType": args.training_instance_type,
                "GAPop": args.ga_pop,
                "GAGen": args.ga_gen,
                "HPOMaxJobs": args.hpo_max_jobs,
                "HPOMaxParallelJobs": args.hpo_max_parallel_jobs,
            },
        )
        logger.info(f"Pipeline execution ARN: {execution.arn}")
        logger.info(
            f"Parâmetros: HPO max_jobs={args.hpo_max_jobs} parallel={args.hpo_max_parallel_jobs} | "
            f"GA pop={args.ga_pop} gen={args.ga_gen} | "
            f"instance={args.training_instance_type}"
        )
        logger.info("Acompanhando execução do Pipeline... (poll a cada 60s)")

        sm_client = session.sagemaker_client
        poll_interval = 60
        last_step_statuses = {}

        while True:
            exec_desc = execution.describe()
            exec_status = exec_desc["PipelineExecutionStatus"]

            # Listar steps e logar mudanças de status
            steps = execution.list_steps()
            total_steps = len(steps)
            done_steps = sum(1 for s in steps if s["StepStatus"] in ("Succeeded", "Failed", "Stopped"))

            _STEP_DESC = {
                "TuningStep": "HPO Bayesian (otimização de hiperparâmetros)",
                "TrainingStep": "GA Training (treinamento com warm start)",
            }

            for idx, step in enumerate(steps, 1):
                step_name = step["StepName"]
                step_status = step["StepStatus"]
                step_type = step.get("StepType", "")
                desc = _STEP_DESC.get(step_type, step_type)
                prev = last_step_statuses.get(step_name)
                if step_status != prev:
                    logger.info(
                        f"  [Step {idx}/{total_steps}] {step_name} ({desc}): "
                        f"{prev or 'Pendente'} → {step_status}"
                    )
                    last_step_statuses[step_name] = step_status

                # Detalhar progresso do HPO Tuning
                if step_status == "Executing" and step.get("Metadata", {}).get("TuningJob"):
                    tuning_job_name = step["Metadata"]["TuningJob"]["Arn"].split("/")[-1]
                    try:
                        tj = sm_client.describe_hyper_parameter_tuning_job(
                            HyperParameterTuningJobName=tuning_job_name
                        )
                        counts = tj.get("TrainingJobStatusCounters", {})
                        best = tj.get("BestTrainingJob", {})
                        best_metric = best.get("FinalHyperParameterTuningJobObjectiveMetric", {})
                        logger.info(
                            f"    HPO '{tuning_job_name}': "
                            f"Completed={counts.get('Completed', 0)}, "
                            f"InProgress={counts.get('InProgress', 0)}, "
                            f"Failed={counts.get('Failed', 0)} | "
                            f"BestMetric={best_metric.get('MetricName', '-')}="
                            f"{best_metric.get('Value', '-')}"
                        )
                    except Exception:
                        pass

                # Detalhar progresso do Training Job
                if step_status == "Executing" and step.get("Metadata", {}).get("TrainingJob"):
                    training_job_arn = step["Metadata"]["TrainingJob"]["Arn"]
                    training_job_name = training_job_arn.split("/")[-1]
                    try:
                        tj = sm_client.describe_training_job(TrainingJobName=training_job_name)
                        tj_status = tj["TrainingJobStatus"]
                        secondary = tj.get("SecondaryStatus", "")
                        elapsed = ""
                        if "TrainingStartTime" in tj:
                            secs = (datetime.now(tj["TrainingStartTime"].tzinfo) - tj["TrainingStartTime"]).total_seconds()
                            elapsed = f" | Elapsed={math.floor(secs//60)}m{int(secs%60)}s"
                        logger.info(
                            f"    Training '{training_job_name}': {tj_status} ({secondary}){elapsed}"
                        )
                    except Exception:
                        pass

            if exec_status in ("Succeeded", "Failed", "Stopped"):
                logger.info(f"Pipeline finalizado com status: {exec_status}")
                break

            logger.info(f"  Pipeline: {exec_status} | steps {done_steps}/{total_steps} concluídos — aguardando {poll_interval}s...")
            time.sleep(poll_interval)

        # Para deploy, recuperar o training job do Pipeline e recriar estimator com session normal
        estimator = ga_estimator
        if exec_status == "Succeeded":
            try:
                # Buscar o TrainingStep do Pipeline para obter o job name real
                ga_step = next(
                    (s for s in execution.list_steps()
                     if s.get("Metadata", {}).get("TrainingJob")),
                    None
                )
                if ga_step:
                    training_job_name = ga_step["Metadata"]["TrainingJob"]["Arn"].split("/")[-1]
                    from sagemaker.sklearn.estimator import SKLearn as _SKLearn
                    estimator = _SKLearn.attach(training_job_name, sagemaker_session=session)
                    logger.info(f"Estimator recuperado do Pipeline: {training_job_name}")
            except Exception as e:
                logger.warning(f"Não foi possível recuperar estimator do Pipeline: {e}")
        tuner = None
        warm_start_params = None
        _phase(5, "", end=True)

    else:
        # ----- Modo manual (sem Pipeline) -----
        logger.info("Executando em modo manual (sem SageMaker Pipelines)...")

        # HPO Tuning Job
        tuner, warm_start_params = run_hpo_tuning_job(
            data_s3_uri=data_s3_uri,
            bucket=args.bucket,
            region=args.region,
            project=args.project,
            role_arn=role_arn,
            session=session,
            training_instance_type=args.training_instance_type,
            max_run=args.max_run,
            max_spot_wait=args.max_spot_wait,
            hpo_max_jobs=args.hpo_max_jobs,
            hpo_max_parallel_jobs=args.hpo_max_parallel_jobs,
            experiment_name=experiment_name,
        )

        # Logar HPO results no Experiment
        with Run(
            experiment_name=experiment_name,
            run_name=f"hpo-results-{datetime.now(BRT).strftime('%Y%m%d%H%M%S')}",
            sagemaker_session=session,
        ) as run:
            run.log_parameter("hpo_tuning_job", tuner.latest_tuning_job.name)
            run.log_parameter("hpo_max_jobs", args.hpo_max_jobs)
            for i, params in enumerate(warm_start_params):
                for k, v in params.items():
                    run.log_parameter(f"hpo_top{i+1}_{k}", v)

        # GA Training Job com warm start
        estimator = run_training_job(
            data_s3_uri=data_s3_uri,
            bucket=args.bucket,
            region=args.region,
            project=args.project,
            role_arn=role_arn,
            session=session,
            training_instance_type=args.training_instance_type,
            ga_pop=args.ga_pop,
            ga_gen=args.ga_gen,
            max_run=args.max_run,
            max_spot_wait=args.max_spot_wait,
            warm_start_params=warm_start_params,
            experiment_name=experiment_name,
        )
        _phase(5, "", end=True)

    # ================================================================
    # 6. Aguardar Autopilot e comparar resultados
    # ================================================================
    _phase(6, "Aguardar Autopilot e comparar resultados")
    autopilot_results = None
    if auto_ml and autopilot_job:
        autopilot_results = wait_for_autopilot(
            auto_ml, autopilot_job, session,
            timeout_minutes=getattr(args, 'autopilot_timeout', 0)
        )

    # Logar comparação no Experiment
    with Run(
        experiment_name=experiment_name,
        run_name=f"comparison-{datetime.now(BRT).strftime('%Y%m%d%H%M%S')}",
        sagemaker_session=session,
    ) as run:
        run.log_parameter("ga_training_job", estimator.latest_training_job.name)
        run.log_parameter("ga_model_s3_uri", str(estimator.model_data))
        run.log_parameter("pipeline_mode", str(args.use_pipeline))
        if autopilot_results:
            run.log_parameter("autopilot_candidate", autopilot_results["candidate_name"])
            run.log_metric("autopilot_f1", autopilot_results["metric_value"])
        if tuner:
            run.log_parameter("hpo_tuning_job", tuner.latest_tuning_job.name)

    _phase(6, "", end=True)
    logger.info("  ┌─ COMPARAÇÃO DE RESULTADOS ─────────────────────────────┐")
    logger.info(f"  │  GA Job  : {estimator.latest_training_job.name}")
    logger.info(f"  │  GA Model: {estimator.model_data}")
    if autopilot_results:
        logger.info(f"  │  Autopilot candidato : {autopilot_results['candidate_name']}")
        logger.info(f"  │  Autopilot {autopilot_results['metric_name']}: "
                    f"{autopilot_results['metric_value']:.4f}")
    logger.info("  └──────────────────────────────────────────────────────────┘")

    # ================================================================
    # 7. Salvar métricas consolidadas no S3
    # ================================================================
    _phase(7, "Salvar métricas consolidadas no S3")
    metrics = {
        "experiment_name": experiment_name,
        "pipeline_mode": args.use_pipeline,
        "feature_store_enabled": not args.skip_feature_store,
        "ga_training_job": estimator.latest_training_job.name,
        "ga_model_s3_uri": str(estimator.model_data),
        "hpo_tuning_job": tuner.latest_tuning_job.name if tuner else None,
        "hpo_warm_start_params": warm_start_params,
        "autopilot_job": autopilot_job,
        "autopilot_results": autopilot_results,
        "timestamp": datetime.now(BRT).isoformat(),
    }
    s3 = boto3.client("s3", region_name=args.region)
    s3.put_object(
        Bucket=args.bucket,
        Key="output/training_metrics.json",
        Body=json.dumps(metrics, indent=2, default=str),
        ContentType="application/json",
    )
    logger.info(f"Métricas salvas em s3://{args.bucket}/output/training_metrics.json")
    _phase(7, "", end=True)

    # ================================================================
    # 8. Deploy dos endpoints
    # ================================================================
    _phase(8, "Deploy dos endpoints de inferência")
    if not args.skip_deploy:
        # Deploy do modelo GA (endpoint principal com inference.py customizado)
        endpoint_name = deploy_sagemaker_endpoint(
            estimator, args.project, args.endpoint_instance_type
        )
        logger.info(f"Endpoint GA ativo: {endpoint_name}")

        # Deploy do Autopilot (endpoint separado para comparação)
        if autopilot_results and auto_ml:
            autopilot_endpoint = deploy_autopilot_endpoint(
                auto_ml, autopilot_job, session,
                args.project, args.endpoint_instance_type
            )
            logger.info(f"Endpoint Autopilot ativo: {autopilot_endpoint}")
    else:
        logger.info("Deploy dos endpoints ignorado (--skip-deploy).")
    _phase(8, "", end=True)

    bar = "═" * 58
    logger.info(f"╔{bar}╗")
    logger.info("║  ✓ Pipeline concluído com sucesso!                      ║")
    logger.info(f"║  Fim: {datetime.now(BRT).strftime('%Y-%m-%d %H:%M:%S')}                               ║")
    logger.info(f"╚{bar}╝")


if __name__ == "__main__":
    main()
