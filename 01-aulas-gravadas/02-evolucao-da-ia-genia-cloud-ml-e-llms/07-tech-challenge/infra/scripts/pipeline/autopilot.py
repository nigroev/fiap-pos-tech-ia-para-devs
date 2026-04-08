"""
autopilot.py — SageMaker Autopilot (AutoML): lançamento e polling.
"""

import time
from datetime import datetime

from .config import BRT, logger


def run_autopilot_job(data_s3_uri, bucket, region, project, role_arn, session,
                      max_candidates=20, max_runtime_per_job=600,
                      max_total_runtime=2700):
    """Lança um job SageMaker Autopilot via boto3 de forma assíncrona.

    Usa Mode=ENSEMBLING para execução mais rápida e eficiente.
    max_runtime_per_job: segundos máximos por candidato (padrão: 600s = 10min)
    max_total_runtime:   segundos máximos no total   (padrão: 2700s = 45min)
    """
    logger.info("Lançando SageMaker Autopilot (AutoML)...")

    timestamp = datetime.now(BRT).strftime("%m%d%H%M%S")
    # AutoML job name: max 32 chars, pattern [a-zA-Z0-9](-*[a-zA-Z0-9]){0,31}
    job_name = f"avc-ap-{timestamp}"

    sm_client = session.sagemaker_client
    try:
        # Usar AutoMLConfig (Autopilot V2) para melhor suporte a buckets e segurança
        sm_client.create_auto_ml_job(
            AutoMLJobName=job_name,
            AutoMLJobConfig={
                "CompletionCriteria": {
                    "MaxCandidates": max_candidates,
                    "MaxRuntimePerTrainingJobInSeconds": max_runtime_per_job,
                    "MaxAutoMLJobRuntimeInSeconds": max_total_runtime,
                },
                "Mode": "ENSEMBLING",
                "DataSplitConfig": {"ValidationFraction": 0.2},
                "SecurityConfig": {
                    "EnableInterContainerTrafficEncryption": False,
                },
            },
            InputDataConfig=[{
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": data_s3_uri,
                    }
                },
                "TargetAttributeName": "MCQ160F_stroke_bin",
            }],
            OutputDataConfig={
                "S3OutputPath": f"s3://{bucket}/autopilot-output",
            },
            RoleArn=role_arn,
        )
        logger.info(
            f"Autopilot job lançado: {job_name} "
            f"(mode=ENSEMBLING, max_candidates={max_candidates}, "
            f"max_runtime_per_job={max_runtime_per_job}s, max_total={max_total_runtime}s)"
        )
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

    return None, job_name


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

    # Obter o melhor candidato direto do describe (sem precisar do SDK AutoML)
    response = sm_client.describe_auto_ml_job(AutoMLJobName=job_name)
    best = response.get("BestCandidate", {})
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
