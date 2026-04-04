"""
config.py — Configuração global: logging, timezone, diretório de treino.
"""

import logging
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone

# Fuso horário de São Paulo (UTC-3)
BRT = timezone(timedelta(hours=-3))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.Formatter.converter = lambda *args: datetime.now(BRT).timetuple()
logger = logging.getLogger("pipeline")


def add_cloudwatch_handler(log_group, region):
    """Adiciona handler CloudWatch Logs ao logger (se watchtower disponível).

    Envia os mesmos logs que vão para stdout também para o CloudWatch,
    permitindo consulta em tempo real via AWS Console sem SSH no notebook.
    """
    try:
        import boto3
        import watchtower

        stream_name = f"exec-{datetime.now(BRT).strftime('%Y%m%d-%H%M%S')}"
        cw_handler = watchtower.CloudWatchLogHandler(
            log_group_name=log_group,
            log_stream_name=stream_name,
            boto3_session=boto3.Session(region_name=region),
            send_interval=10,
            create_log_group=False,
        )
        cw_handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        cw_handler.formatter.converter = lambda *args: datetime.now(BRT).timetuple()

        logger.addHandler(cw_handler)
        # Adicionar ao root logger também para capturar logs de bibliotecas
        logging.getLogger().addHandler(cw_handler)
        logger.info(f"CloudWatch Logs ativado: {log_group}/{stream_name}")
    except ImportError:
        logger.warning("watchtower não instalado — logs apenas em stdout/arquivo")
    except Exception as e:
        logger.warning(f"Falha ao configurar CloudWatch Logs: {e}")


# Diretório temporário com apenas train.py (sem requirements.txt pesado)
# O SKLearn container já possui sklearn, pandas, numpy, joblib.
_TRAIN_SOURCE_DIR = None


def get_train_source_dir():
    """Retorna diretório temporário contendo apenas train.py.

    Evita que o SageMaker instale dependências pesadas do requirements.txt
    (sagemaker[feature-store], pyarrow, etc.) dentro do container de treinamento.
    """
    global _TRAIN_SOURCE_DIR
    if _TRAIN_SOURCE_DIR is None:
        _TRAIN_SOURCE_DIR = tempfile.mkdtemp(prefix="sagemaker_train_src_")
        # train.py está em scripts/ (mesmo nível do diretório pipeline/)
        scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = os.path.join(scripts_dir, "train.py")
        shutil.copy2(src, os.path.join(_TRAIN_SOURCE_DIR, "train.py"))
        logger.info(f"Training source dir criado: {_TRAIN_SOURCE_DIR}")
    return _TRAIN_SOURCE_DIR
