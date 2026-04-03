#!/usr/bin/env bash
# ==============================================================================
# bootstrap.sh — Zera jobs/pipelines que consomem quotas e sobe a infra
#
# Uso:
#   cd infra/
#   bash bootstrap.sh          # zera recursos + terraform apply
#   bash bootstrap.sh --only-cleanup   # só zera, sem apply
# ==============================================================================
set -euo pipefail

ONLY_CLEANUP="${1:-}"

# ── Parâmetros lidos do terraform.tfvars ───────────────────────────────────────
PROJECT=$(grep 'project_name' terraform.tfvars | awk -F'"' '{print $2}')
REGION=$(grep 'aws_region'    terraform.tfvars | awk -F'"' '{print $2}')

echo "════════════════════════════════════════════════════"
echo "  Cleanup de quotas SageMaker"
echo "  Projeto : ${PROJECT}"
echo "  Região  : ${REGION}"
echo "════════════════════════════════════════════════════"

SM="aws sagemaker --region ${REGION} --output text"

# ── 1. Parar AutoML jobs em andamento ─────────────────────────────────────────
echo ""
echo "▶ AutoML jobs (InProgress)..."
AUTOML_JOBS=$(aws sagemaker list-auto-ml-jobs \
  --region "${REGION}" \
  --status-equals InProgress \
  --query "AutoMLJobSummaries[?starts_with(AutoMLJobName, '${PROJECT}') || starts_with(AutoMLJobName, 'avc-')].AutoMLJobName" \
  --output text 2>/dev/null || true)

for JOB in $AUTOML_JOBS; do
  echo "  Parando AutoML job: ${JOB}"
  aws sagemaker stop-auto-ml-job --auto-ml-job-name "${JOB}" --region "${REGION}" || true
done
[ -z "${AUTOML_JOBS}" ] && echo "  Nenhum AutoML job em andamento."

# ── 2. Parar HPO Tuning jobs em andamento ─────────────────────────────────────
echo ""
echo "▶ HPO Tuning jobs (InProgress)..."
HPO_JOBS=$(aws sagemaker list-hyper-parameter-tuning-jobs \
  --region "${REGION}" \
  --status-equals InProgress \
  --query "HyperParameterTuningJobSummaries[?starts_with(HyperParameterTuningJobName, '${PROJECT}') || starts_with(HyperParameterTuningJobName, 'sagemaker-scikit')].HyperParameterTuningJobName" \
  --output text 2>/dev/null || true)

for JOB in $HPO_JOBS; do
  echo "  Parando HPO job: ${JOB}"
  aws sagemaker stop-hyper-parameter-tuning-job --hyper-parameter-tuning-job-name "${JOB}" --region "${REGION}" || true
done
[ -z "${HPO_JOBS}" ] && echo "  Nenhum HPO job em andamento."

# ── 3. Parar Training jobs em andamento ───────────────────────────────────────
echo ""
echo "▶ Training jobs (InProgress)..."
TRAINING_JOBS=$(aws sagemaker list-training-jobs \
  --region "${REGION}" \
  --status-equals InProgress \
  --query "TrainingJobSummaries[?starts_with(TrainingJobName, '${PROJECT}') || starts_with(TrainingJobName, 'sagemaker-scikit')].TrainingJobName" \
  --output text 2>/dev/null || true)

for JOB in $TRAINING_JOBS; do
  echo "  Parando Training job: ${JOB}"
  aws sagemaker stop-training-job --training-job-name "${JOB}" --region "${REGION}" || true
done
[ -z "${TRAINING_JOBS}" ] && echo "  Nenhum Training job em andamento."

# ── 4. Parar execuções de Pipeline em andamento ───────────────────────────────
echo ""
echo "▶ Pipeline executions (Executing)..."
PIPELINES=$(aws sagemaker list-pipelines \
  --region "${REGION}" \
  --pipeline-name-prefix "${PROJECT}" \
  --query "PipelineSummaries[].PipelineName" \
  --output text 2>/dev/null || true)

for PIPELINE in $PIPELINES; do
  EXECUTIONS=$(aws sagemaker list-pipeline-executions \
    --region "${REGION}" \
    --pipeline-name "${PIPELINE}" \
    --query "PipelineExecutionSummaries[?PipelineExecutionStatus=='Executing'].PipelineExecutionArn" \
    --output text 2>/dev/null || true)
  for EXEC_ARN in $EXECUTIONS; do
    echo "  Parando execução: ${EXEC_ARN}"
    aws sagemaker stop-pipeline-execution --pipeline-execution-arn "${EXEC_ARN}" --region "${REGION}" || true
  done
done
[ -z "${PIPELINES}" ] && echo "  Nenhuma pipeline encontrada."

echo ""
echo "✓ Cleanup concluído."

# ── Aguardar jobs pararem (até 60s) ───────────────────────────────────────────
if [ -n "${AUTOML_JOBS}${HPO_JOBS}${TRAINING_JOBS}" ]; then
  echo "  Aguardando jobs encerrarem (até 60s)..."
  sleep 20
fi

# ── 5. Terraform apply ────────────────────────────────────────────────────────
if [ "${ONLY_CLEANUP}" != "--only-cleanup" ]; then
  echo ""
  echo "════════════════════════════════════════════════════"
  echo "  terraform apply"
  echo "════════════════════════════════════════════════════"
  terraform apply -auto-approve
fi

