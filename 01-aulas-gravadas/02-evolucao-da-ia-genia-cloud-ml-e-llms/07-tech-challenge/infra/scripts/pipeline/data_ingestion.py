"""
data_ingestion.py — Coleta NHANES, pré-processamento e upload para S3.
"""

import os
import tempfile

import boto3
import pandas as pd

from .config import logger


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
