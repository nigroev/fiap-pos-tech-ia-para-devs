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
SKIP_DEPLOY="${skip_deploy}"
TRAINING_INSTANCE_TYPE="${training_instance_type}"
ENDPOINT_INSTANCE_TYPE="${endpoint_instance_type}"
HPO_MAX_JOBS="${hpo_max_jobs}"
HPO_MAX_PARALLEL_JOBS="${hpo_max_parallel_jobs}"
GA_POP="${ga_pop}"
GA_GEN="${ga_gen}"
AUTOPILOT_MAX_CANDIDATES="${autopilot_max_candidates}"
AUTOPILOT_TIMEOUT="${autopilot_timeout}"
MAX_RUN="${max_run}"
MAX_SPOT_WAIT="${max_spot_wait}"

echo "=============================="
echo " on_start: Iniciando pipeline"
echo "=============================="

WORK_DIR="/home/ec2-user/SageMaker/$${PROJECT}"

# Atualizar scripts do S3 (caso tenham sido alterados)
aws s3 cp "s3://$${BUCKET_NAME}/scripts/train_and_deploy.py" "$${WORK_DIR}/train_and_deploy.py"
aws s3 cp "s3://$${BUCKET_NAME}/scripts/requirements.txt" "$${WORK_DIR}/requirements.txt"
mkdir -p "$${WORK_DIR}/inference_src"
aws s3 cp "s3://$${BUCKET_NAME}/scripts/inference_src/" "$${WORK_DIR}/inference_src/" --recursive
mkdir -p "$${WORK_DIR}/pipeline"
aws s3 cp "s3://$${BUCKET_NAME}/scripts/pipeline/" "$${WORK_DIR}/pipeline/" --recursive

# Executar o pipeline de treinamento e deploy em background
# para não bloquear a inicialização do notebook
nohup bash -c "
  source /home/ec2-user/anaconda3/bin/activate python3
  cd $${WORK_DIR}
  pip install -q -r requirements.txt
  SKIP_DEPLOY_FLAG=\"\"
  if [ \"$${SKIP_DEPLOY}\" = \"true\" ]; then SKIP_DEPLOY_FLAG=\"--skip-deploy\"; fi
  python train_and_deploy.py \
    --bucket $${BUCKET_NAME} \
    --region $${REGION} \
    --project $${PROJECT} \
    --use-pipeline \
    --training-instance-type $${TRAINING_INSTANCE_TYPE} \
    --endpoint-instance-type $${ENDPOINT_INSTANCE_TYPE} \
    --hpo-max-jobs $${HPO_MAX_JOBS} \
    --hpo-max-parallel-jobs $${HPO_MAX_PARALLEL_JOBS} \
    --ga-pop $${GA_POP} \
    --ga-gen $${GA_GEN} \
    --autopilot-max-candidates $${AUTOPILOT_MAX_CANDIDATES} \
    --autopilot-timeout $${AUTOPILOT_TIMEOUT} \
    --max-run $${MAX_RUN} \
    --max-spot-wait $${MAX_SPOT_WAIT} \
    $${SKIP_DEPLOY_FLAG} \
    > /home/ec2-user/SageMaker/train_and_deploy.log 2>&1
  source /home/ec2-user/anaconda3/bin/deactivate
" &

echo "on_start: Pipeline disparado em background. Veja /home/ec2-user/SageMaker/train_and_deploy.log"
