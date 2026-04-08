#!/usr/bin/env bash
# ==============================================================================
# bootstrap.sh — Zera jobs/pipelines que consomem quotas e sobe a infra
#
# Uso:
#   cd infra/
#   bash bootstrap.sh > bootstrap.log                 # zera recursos + terraform apply
#   bash bootstrap.sh --only-cleanup> bootstrap.log   # só zera, sem apply
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

# ── 5. Limpar Experiments (trial components) para evitar ResourceLimitExceeded ─
echo ""
echo "▶ Limpando SageMaker Experiments (trials e trial-components)..."
EXPERIMENT="${PROJECT}-experiment"
# Listar trials do experiment
TRIALS=$(aws sagemaker list-trials \
  --region "${REGION}" \
  --experiment-name "${EXPERIMENT}" \
  --query "TrialSummaries[].TrialName" \
  --output text 2>/dev/null || true)
for TRIAL in $TRIALS; do
  # Desassociar todos os trial components do trial
  TCS=$(aws sagemaker list-trial-components \
    --region "${REGION}" \
    --trial-name "${TRIAL}" \
    --query "TrialComponentSummaries[].TrialComponentName" \
    --output text 2>/dev/null || true)
  for TC in $TCS; do
    echo "  Desassociando ${TC} de ${TRIAL}"
    aws sagemaker disassociate-trial-component \
      --trial-name "${TRIAL}" \
      --trial-component-name "${TC}" \
      --region "${REGION}" 2>/dev/null || true
    echo "  Deletando trial component: ${TC}"
    aws sagemaker delete-trial-component \
      --trial-component-name "${TC}" \
      --region "${REGION}" 2>/dev/null || true
  done
  echo "  Deletando trial: ${TRIAL}"
  aws sagemaker delete-trial --trial-name "${TRIAL}" --region "${REGION}" 2>/dev/null || true
done
echo "  Deletando experiment: ${EXPERIMENT}"
aws sagemaker delete-experiment --experiment-name "${EXPERIMENT}" --region "${REGION}" 2>/dev/null || true
echo "  Experiment cleanup concluído."

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

  # Remove aws_s3_object entries from state before applying.
  # When the bucket is destroyed and recreated, Terraform's state refresh tries
  # to read those objects from the (now empty) bucket and fails with
  # "couldn't find resource". Removing them from state lets Terraform recreate
  # them cleanly on the next apply (etag ensures idempotency when unchanged).
  echo "▶ Resetando estado de objetos S3..."
  S3_STATE=$(terraform state list 2>/dev/null | grep '^aws_s3_object\.' || true)
  if [ -n "$S3_STATE" ]; then
    # xargs passes each resource as a separate argument to a single state rm call
    echo "$S3_STATE" | xargs terraform state rm 2>/dev/null || true
  fi

  terraform apply -auto-approve
fi

