#!/bin/bash
# ==============================================================================
# SageMaker Notebook Lifecycle — on_start
# Executado TODA VEZ que o notebook é iniciado.
# Dispara o treinamento e deploy do modelo em background.
# ==============================================================================
set -e

BUCKET_NAME="${bucket_name}"
REGION="${region}"
PROJECT="${project}"

echo "=============================="
echo " on_start: Iniciando pipeline"
echo "=============================="

WORK_DIR="/home/ec2-user/SageMaker/$${PROJECT}"

# Atualizar scripts do S3 (caso tenham sido alterados)
aws s3 cp "s3://$${BUCKET_NAME}/scripts/train_and_deploy.py" "$${WORK_DIR}/train_and_deploy.py"
aws s3 cp "s3://$${BUCKET_NAME}/scripts/requirements.txt" "$${WORK_DIR}/requirements.txt"

# Executar o pipeline de treinamento e deploy em background
# para não bloquear a inicialização do notebook
nohup bash -c "
  source /home/ec2-user/anaconda3/bin/activate python3
  cd $${WORK_DIR}
  python train_and_deploy.py \
    --bucket $${BUCKET_NAME} \
    --region $${REGION} \
    --project $${PROJECT} \
    > /home/ec2-user/SageMaker/train_and_deploy.log 2>&1
  source /home/ec2-user/anaconda3/bin/deactivate
" &

echo "on_start: Pipeline disparado em background. Veja /home/ec2-user/SageMaker/train_and_deploy.log"
