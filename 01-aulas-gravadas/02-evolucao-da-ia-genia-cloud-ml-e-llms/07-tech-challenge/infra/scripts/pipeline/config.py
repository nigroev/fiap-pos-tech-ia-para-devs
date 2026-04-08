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
