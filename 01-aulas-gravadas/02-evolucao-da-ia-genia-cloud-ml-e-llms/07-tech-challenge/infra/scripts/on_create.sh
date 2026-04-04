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

# Aguardar scripts no S3 (depends_on no Terraform garante ordem,
# mas adicionamos retry como safety net para eventual consistency)
echo "on_create: Aguardando scripts no S3..."
for i in 1 2 3 4 5; do
  aws s3 ls "s3://$${BUCKET_NAME}/scripts/train_and_deploy.py" && break
  echo "  Tentativa $i — scripts ainda não disponíveis, aguardando 10s..."
  sleep 10
done

# Baixar scripts do S3
aws s3 cp "s3://$${BUCKET_NAME}/scripts/train_and_deploy.py" "$${WORK_DIR}/train_and_deploy.py"
aws s3 cp "s3://$${BUCKET_NAME}/scripts/train.py" "$${WORK_DIR}/train.py"
aws s3 cp "s3://$${BUCKET_NAME}/scripts/requirements.txt" "$${WORK_DIR}/requirements.txt"
aws s3 cp "s3://$${BUCKET_NAME}/scripts/inference_src/" "$${WORK_DIR}/inference_src/" --recursive
aws s3 cp "s3://$${BUCKET_NAME}/scripts/pipeline/" "$${WORK_DIR}/pipeline/" --recursive

# Instalar dependências em background (pip install pesado > 5min do lifecycle timeout)
# O on_start.sh também faz pip install dentro do nohup, garantindo que deps
# estarão prontas antes de rodar o Python.
nohup bash -c "
  source /home/ec2-user/anaconda3/bin/activate python3
  pip install -q -r \"$${WORK_DIR}/requirements.txt\"
  source /home/ec2-user/anaconda3/bin/deactivate
  touch \"$${WORK_DIR}/.deps_installed\"
" > /home/ec2-user/SageMaker/on_create_install.log 2>&1 &

echo "on_create: Configuração concluída (pip install em background)."
