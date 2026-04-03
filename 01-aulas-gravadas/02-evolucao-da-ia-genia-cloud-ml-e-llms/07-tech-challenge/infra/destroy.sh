#!/usr/bin/env bash
# ==============================================================================
# destroy.sh — Zera jobs/pipelines que consomem quotas e remove a infra
#
# Uso:
#   cd infra/
#   bash destroy.sh          # terraform destroy
# ==============================================================================

terraform destroy -auto-approve