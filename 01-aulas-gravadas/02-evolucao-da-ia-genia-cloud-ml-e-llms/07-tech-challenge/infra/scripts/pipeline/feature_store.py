"""
feature_store.py — SageMaker Feature Store: criação, ingestão e leitura.
"""

import time
import uuid
from datetime import datetime, timezone

from sagemaker.feature_store.feature_definition import (
    FeatureDefinition,
    FeatureTypeEnum,
)
from sagemaker.feature_store.feature_group import FeatureGroup

from .config import logger

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
        desc = sm_client.describe_feature_group(FeatureGroupName=feature_group_name)

        # Verificar se o bucket do offline store ainda é válido
        offline_cfg = desc.get("OfflineStoreConfig", {})
        resolved_uri = offline_cfg.get("S3StorageConfig", {}).get(
            "ResolvedOutputS3Uri", ""
        )
        expected_prefix = f"s3://{bucket}/"

        if resolved_uri and not resolved_uri.startswith(expected_prefix):
            logger.warning(
                f"Feature Group aponta para bucket diferente: {resolved_uri}"
            )
            logger.info(f"Deletando Feature Group antigo: {feature_group_name}")
            sm_client.delete_feature_group(FeatureGroupName=feature_group_name)
            for _ in range(30):
                try:
                    sm_client.describe_feature_group(
                        FeatureGroupName=feature_group_name
                    )
                    time.sleep(10)
                except sm_client.exceptions.ResourceNotFound:
                    break
            logger.info(f"Feature Group deletado, recriando com bucket correto")
        else:
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
