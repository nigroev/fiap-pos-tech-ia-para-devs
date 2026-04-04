# ==============================================================================
# CloudWatch Log Group — Logs do orquestrador train_and_deploy.py
# ==============================================================================

resource "aws_cloudwatch_log_group" "train_and_deploy" {
  name              = "/aws/sagemaker/${local.name_prefix}/train-and-deploy"
  retention_in_days = 30

  tags = {
    Name = "${local.name_prefix}-train-and-deploy-logs"
  }
}
