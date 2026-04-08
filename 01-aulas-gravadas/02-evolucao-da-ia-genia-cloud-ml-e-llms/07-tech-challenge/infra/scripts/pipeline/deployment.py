"""
deployment.py — Deploy e gestão de endpoints SageMaker.
"""

import io
import os
import tarfile
import time

from sagemaker import get_execution_role
from sagemaker import image_uris

from .config import logger


def _cleanup_endpoint(sm_client, endpoint_name, model_name=None):
    """Remove endpoint, endpoint-config e model antigos para evitar conflitos."""
    # Deletar endpoint
    try:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        logger.info(f"Deletando endpoint antigo '{endpoint_name}'...")
        sm_client.delete_endpoint(EndpointName=endpoint_name)
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

    # Deletar model (nome fixo baseado no endpoint)
    target_model = model_name or f"{endpoint_name}-model"
    try:
        sm_client.delete_model(ModelName=target_model)
        logger.info(f"Model '{target_model}' deletado.")
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


def deploy_sagemaker_endpoint(estimator, project, endpoint_instance_type, role_arn=None, max_retries=2):
    """Faz deploy do modelo treinado (GA) via boto3 direto, sem depender do SDK model.deploy().

    Usa boto3 para create_model / create_endpoint_config / create_endpoint,
    evitando bugs de versão do SageMaker Python SDK onde role=None é passado
    internamente para endpoint_from_production_variants.
    """
    logger.info("Fazendo deploy do endpoint no SageMaker...")

    sm_session = estimator.sagemaker_session
    sm_client = sm_session.sagemaker_client
    region = sm_session.boto_region_name
    bucket = sm_session.default_bucket()
    endpoint_name = f"{project}-endpoint"
    model_name = f"{endpoint_name}-model"

    effective_role = role_arn or estimator.role
    if not effective_role:
        try:
            effective_role = get_execution_role(sm_session)
            logger.warning(f"role_arn e estimator.role são None; usando get_execution_role: {effective_role}")
        except Exception:
            raise RuntimeError(
                "Não foi possível determinar a IAM role para o deploy. "
                "Passe --role-arn ou garanta que o estimator tenha .role preenchido."
            )

    # Verificar que inference_src existe
    scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    inference_src = os.path.join(scripts_dir, "inference_src")
    if not os.path.isfile(os.path.join(inference_src, "inference.py")):
        raise FileNotFoundError(f"inference.py não encontrado em {inference_src}")

    # Empacotar inference_src como sourcedir.tar.gz e fazer upload para S3
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w:gz") as tar:
        for fname in os.listdir(inference_src):
            fpath = os.path.join(inference_src, fname)
            if os.path.isfile(fpath):
                tar.add(fpath, arcname=fname)
    tar_buf.seek(0)

    sourcedir_key = f"{project}/inference-src/sourcedir.tar.gz"
    s3_client = sm_session.boto_session.client("s3", region_name=region)
    s3_client.upload_fileobj(tar_buf, bucket, sourcedir_key)
    sourcedir_s3 = f"s3://{bucket}/{sourcedir_key}"
    logger.info(f"inference_src enviado para {sourcedir_s3}")

    # Obter imagem do container SKLearn
    sklearn_image = image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="1.2-1",
        py_version="py3",
        instance_type=endpoint_instance_type,
    )
    logger.info(f"Container image: {sklearn_image}")

    for attempt in range(1, max_retries + 1):
        _cleanup_endpoint(sm_client, endpoint_name, model_name=model_name)

        logger.info(f"Criando endpoint '{endpoint_name}' (tentativa {attempt}/{max_retries}, instance: {endpoint_instance_type})...")
        logger.info(f"Model data: {estimator.model_data}")

        try:
            # 1. Registrar model no SageMaker via boto3 (sem depender do SDK)
            sm_client.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    "Image": sklearn_image,
                    "ModelDataUrl": estimator.model_data,
                    "Environment": {
                        "SAGEMAKER_PROGRAM": "inference.py",
                        "SAGEMAKER_SUBMIT_DIRECTORY": sourcedir_s3,
                        "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                        "SAGEMAKER_REGION": region,
                    },
                },
                ExecutionRoleArn=effective_role,
            )

            # 2. Criar endpoint config
            sm_client.create_endpoint_config(
                EndpointConfigName=endpoint_name,
                ProductionVariants=[{
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "InitialInstanceCount": 1,
                    "InstanceType": endpoint_instance_type,
                }],
            )

            # 3. Criar endpoint
            sm_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_name,
            )
        except Exception as exc:
            logger.warning(f"Erro ao criar recursos de endpoint (tentativa {attempt}): {exc}")
            if attempt >= max_retries:
                raise
            continue

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
