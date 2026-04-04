"""
deployment.py — Deploy e gestão de endpoints SageMaker.
"""

import os
import time

from .config import logger


def _cleanup_endpoint(sm_client, endpoint_name):
    """Remove endpoint, endpoint-config e models antigos para evitar conflitos."""
    # Deletar endpoint
    try:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        logger.info(f"Deletando endpoint antigo '{endpoint_name}'...")
        sm_client.delete_endpoint(EndpointName=endpoint_name)
        # Aguardar deleção
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
    # inference_src está em scripts/ (mesmo nível do diretório pipeline/)
    scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    inference_src = os.path.join(scripts_dir, "inference_src")
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
