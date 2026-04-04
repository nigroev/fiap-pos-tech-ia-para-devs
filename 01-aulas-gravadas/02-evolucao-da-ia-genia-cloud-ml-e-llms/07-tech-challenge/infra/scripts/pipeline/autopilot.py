"""
autopilot.py — SageMaker Autopilot (AutoML): lançamento e polling.
"""

import time
from datetime import datetime

from sagemaker import AutoML

from .config import BRT, logger


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
