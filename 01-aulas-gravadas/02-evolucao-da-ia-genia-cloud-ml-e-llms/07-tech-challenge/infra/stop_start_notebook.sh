#!/usr/bin/env bash
# ==============================================================================
# stop_start_notebook.sh — Para ou inicia o notebook SageMaker e dependências
#
# Uso:
#   cd infra/
#   bash stop_start_notebook.sh stop    # para notebook + endpoints + jobs
#   bash stop_start_notebook.sh start   # inicia notebook (on_start.sh roda automaticamente)
#   bash stop_start_notebook.sh status  # mostra estado atual
# ==============================================================================
set -euo pipefail

ACTION="${1:-}"

# ── Parâmetros lidos do terraform.tfvars ──────────────────────────────────────
PROJECT=$(grep 'project_name' terraform.tfvars | awk -F'"' '{print $2}')
ENV=$(grep 'environment'      terraform.tfvars | awk -F'"' '{print $2}')
REGION=$(grep 'aws_region'    terraform.tfvars | awk -F'"' '{print $2}')

NOTEBOOK_NAME="${PROJECT}-${ENV}-notebook"
ENDPOINT_PREFIX="${PROJECT}"

SM="aws sagemaker --region ${REGION}"

# ── Funções auxiliares ────────────────────────────────────────────────────────

print_header() {
  echo ""
  echo "════════════════════════════════════════════════════"
  echo "  $1"
  echo "  Notebook : ${NOTEBOOK_NAME}"
  echo "  Região   : ${REGION}"
  echo "════════════════════════════════════════════════════"
  echo ""
}

get_notebook_status() {
  ${SM} describe-notebook-instance \
    --notebook-instance-name "${NOTEBOOK_NAME}" \
    --query "NotebookInstanceStatus" \
    --output text 2>/dev/null || echo "NOT_FOUND"
}

wait_for_status() {
  local target="$1"
  local timeout="${2:-300}"
  local elapsed=0

  echo "  Aguardando status '${target}' (timeout ${timeout}s)..."
  while true; do
    local status
    status=$(get_notebook_status)
    if [ "${status}" = "${target}" ]; then
      echo "  ✓ Notebook em status: ${status}"
      return 0
    fi
    if [ "${elapsed}" -ge "${timeout}" ]; then
      echo "  ✗ Timeout! Status atual: ${status}"
      return 1
    fi
    sleep 10
    elapsed=$((elapsed + 10))
    echo "  ... ${status} (${elapsed}s)"
  done
}

stop_endpoints() {
  echo "▶ Endpoints..."
  local endpoints
  endpoints=$(${SM} list-endpoints \
    --name-contains "${ENDPOINT_PREFIX}" \
    --query "Endpoints[?EndpointStatus!='Deleting'].EndpointName" \
    --output text 2>/dev/null || true)

  for EP in $endpoints; do
    echo "  Deletando endpoint: ${EP}"
    ${SM} delete-endpoint --endpoint-name "${EP}" || true
  done
  [ -z "${endpoints}" ] && echo "  Nenhum endpoint encontrado."

  # Limpar endpoint configs órfãos
  echo ""
  echo "▶ Endpoint configs..."
  local configs
  configs=$(${SM} list-endpoint-configs \
    --name-contains "${ENDPOINT_PREFIX}" \
    --query "EndpointConfigs[].EndpointConfigName" \
    --output text 2>/dev/null || true)

  for CFG in $configs; do
    echo "  Deletando config: ${CFG}"
    ${SM} delete-endpoint-config --endpoint-config-name "${CFG}" || true
  done
  [ -z "${configs}" ] && echo "  Nenhum endpoint config encontrado."
}

stop_jobs() {
  echo "▶ Training jobs (InProgress)..."
  local training_jobs
  training_jobs=$(${SM} list-training-jobs \
    --status-equals InProgress \
    --query "TrainingJobSummaries[?starts_with(TrainingJobName, '${PROJECT}') || starts_with(TrainingJobName, 'sagemaker-scikit')].TrainingJobName" \
    --output text 2>/dev/null || true)

  for JOB in $training_jobs; do
    echo "  Parando: ${JOB}"
    ${SM} stop-training-job --training-job-name "${JOB}" || true
  done
  [ -z "${training_jobs}" ] && echo "  Nenhum."

  echo ""
  echo "▶ HPO Tuning jobs (InProgress)..."
  local hpo_jobs
  hpo_jobs=$(${SM} list-hyper-parameter-tuning-jobs \
    --status-equals InProgress \
    --query "HyperParameterTuningJobSummaries[?starts_with(HyperParameterTuningJobName, '${PROJECT}') || starts_with(HyperParameterTuningJobName, 'sagemaker-scikit')].HyperParameterTuningJobName" \
    --output text 2>/dev/null || true)

  for JOB in $hpo_jobs; do
    echo "  Parando: ${JOB}"
    ${SM} stop-hyper-parameter-tuning-job --hyper-parameter-tuning-job-name "${JOB}" || true
  done
  [ -z "${hpo_jobs}" ] && echo "  Nenhum."

  echo ""
  echo "▶ AutoML jobs (InProgress)..."
  local automl_jobs
  automl_jobs=$(${SM} list-auto-ml-jobs \
    --status-equals InProgress \
    --query "AutoMLJobSummaries[?starts_with(AutoMLJobName, '${PROJECT}') || starts_with(AutoMLJobName, 'avc-')].AutoMLJobName" \
    --output text 2>/dev/null || true)

  for JOB in $automl_jobs; do
    echo "  Parando: ${JOB}"
    ${SM} stop-auto-ml-job --auto-ml-job-name "${JOB}" || true
  done
  [ -z "${automl_jobs}" ] && echo "  Nenhum."

  echo ""
  echo "▶ Pipeline executions (Executing)..."
  local pipelines
  pipelines=$(${SM} list-pipelines \
    --pipeline-name-prefix "${PROJECT}" \
    --query "PipelineSummaries[].PipelineName" \
    --output text 2>/dev/null || true)

  for PIPELINE in $pipelines; do
    local executions
    executions=$(${SM} list-pipeline-executions \
      --pipeline-name "${PIPELINE}" \
      --query "PipelineExecutionSummaries[?PipelineExecutionStatus=='Executing'].PipelineExecutionArn" \
      --output text 2>/dev/null || true)
    for EXEC_ARN in $executions; do
      echo "  Parando: ${EXEC_ARN}"
      ${SM} stop-pipeline-execution --pipeline-execution-arn "${EXEC_ARN}" || true
    done
  done
  [ -z "${pipelines}" ] && echo "  Nenhuma pipeline encontrada."
}

# ── Comandos ──────────────────────────────────────────────────────────────────

do_stop() {
  print_header "STOP — Parando notebook e dependências"

  local status
  status=$(get_notebook_status)
  echo "Status atual do notebook: ${status}"
  echo ""

  # 1. Parar jobs em andamento
  stop_jobs

  # 2. Deletar endpoints (cobram por hora mesmo parado)
  echo ""
  stop_endpoints

  # 3. Parar o notebook
  echo ""
  echo "▶ Parando notebook instance..."
  if [ "${status}" = "InService" ]; then
    ${SM} stop-notebook-instance --notebook-instance-name "${NOTEBOOK_NAME}"
    wait_for_status "Stopped" 300
  elif [ "${status}" = "Stopped" ]; then
    echo "  Notebook já está parado."
  elif [ "${status}" = "Stopping" ]; then
    wait_for_status "Stopped" 300
  else
    echo "  Status inesperado: ${status}. Nenhuma ação tomada."
  fi

  echo ""
  echo "✓ Stop concluído. Todos os recursos que geram custo foram parados/removidos."
}

do_start() {
  print_header "START — Iniciando notebook"

  local status
  status=$(get_notebook_status)
  echo "Status atual do notebook: ${status}"
  echo ""

  if [ "${status}" = "Stopped" ]; then
    echo "▶ Iniciando notebook instance..."
    ${SM} start-notebook-instance --notebook-instance-name "${NOTEBOOK_NAME}"
    wait_for_status "InService" 600
    echo ""
    echo "✓ Notebook iniciado. O lifecycle on_start.sh será executado automaticamente."
    echo "  Acompanhe o log em: JupyterLab → Terminal → tail -f /home/ec2-user/SageMaker/train_and_deploy.log"
  elif [ "${status}" = "InService" ]; then
    echo "  Notebook já está rodando."
  elif [ "${status}" = "Pending" ] || [ "${status}" = "Updating" ]; then
    echo "  Notebook está em transição (${status}). Aguardando InService..."
    wait_for_status "InService" 600
  else
    echo "  Status inesperado: ${status}. Não é possível iniciar."
    exit 1
  fi
}

do_status() {
  print_header "STATUS — Estado atual dos recursos"

  # Notebook
  local nb_status
  nb_status=$(get_notebook_status)
  echo "▶ Notebook: ${nb_status}"

  # Endpoints
  echo ""
  echo "▶ Endpoints:"
  ${SM} list-endpoints \
    --name-contains "${ENDPOINT_PREFIX}" \
    --query "Endpoints[].{Nome:EndpointName,Status:EndpointStatus}" \
    --output table 2>/dev/null || echo "  Nenhum."

  # Training jobs em andamento
  echo ""
  echo "▶ Training jobs (InProgress):"
  ${SM} list-training-jobs \
    --status-equals InProgress \
    --query "TrainingJobSummaries[?starts_with(TrainingJobName, '${PROJECT}')].{Nome:TrainingJobName,Status:TrainingJobStatus}" \
    --output table 2>/dev/null || echo "  Nenhum."

  # HPO jobs em andamento
  echo ""
  echo "▶ HPO Tuning jobs (InProgress):"
  ${SM} list-hyper-parameter-tuning-jobs \
    --status-equals InProgress \
    --query "HyperParameterTuningJobSummaries[?starts_with(HyperParameterTuningJobName, '${PROJECT}')].{Nome:HyperParameterTuningJobName,Status:HyperParameterTuningJobStatus}" \
    --output table 2>/dev/null || echo "  Nenhum."

  # AutoML jobs
  echo ""
  echo "▶ AutoML jobs (InProgress):"
  ${SM} list-auto-ml-jobs \
    --status-equals InProgress \
    --query "AutoMLJobSummaries[?starts_with(AutoMLJobName, '${PROJECT}')].{Nome:AutoMLJobName,Status:AutoMLJobStatus}" \
    --output table 2>/dev/null || echo "  Nenhum."
}

# ── Main ──────────────────────────────────────────────────────────────────────

case "${ACTION}" in
  stop)
    do_stop
    ;;
  start)
    do_start
    ;;
  status)
    do_status
    ;;
  *)
    echo "Uso: bash stop_start_notebook.sh {stop|start|status}"
    echo ""
    echo "  stop    — Para jobs, deleta endpoints e para o notebook"
    echo "  start   — Inicia o notebook (on_start.sh roda automaticamente)"
    echo "  status  — Mostra estado atual de todos os recursos"
    exit 1
    ;;
esac
