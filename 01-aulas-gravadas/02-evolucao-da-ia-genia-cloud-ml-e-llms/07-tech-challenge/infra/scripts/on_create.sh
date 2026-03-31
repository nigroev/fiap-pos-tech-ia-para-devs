#!/bin/bash
# ==============================================================================
# SageMaker Notebook Lifecycle — on_create
# Executado UMA VEZ quando o notebook é criado.
# Instala dependências e baixa os scripts do S3.
# ==============================================================================
set -e

BUCKET_NAME="${bucket_name}"
REGION="${region}"
PROJECT="${project}"

echo "=============================="
echo " on_create: Configurando notebook"
echo "=============================="

# Diretório de trabalho no notebook
WORK_DIR="/home/ec2-user/SageMaker/$${PROJECT}"
mkdir -p "$${WORK_DIR}"

# Baixar scripts do S3
aws s3 cp "s3://$${BUCKET_NAME}/scripts/train_and_deploy.py" "$${WORK_DIR}/train_and_deploy.py"
aws s3 cp "s3://$${BUCKET_NAME}/scripts/train.py" "$${WORK_DIR}/train.py"
aws s3 cp "s3://$${BUCKET_NAME}/scripts/requirements.txt" "$${WORK_DIR}/requirements.txt"
aws s3 cp "s3://$${BUCKET_NAME}/scripts/inference_src/" "$${WORK_DIR}/inference_src/" --recursive

# Instalar dependências no ambiente conda padrão do SageMaker
source /home/ec2-user/anaconda3/bin/activate python3
pip install -r "$${WORK_DIR}/requirements.txt"
source /home/ec2-user/anaconda3/bin/deactivate

echo "on_create: Configuração concluída."
