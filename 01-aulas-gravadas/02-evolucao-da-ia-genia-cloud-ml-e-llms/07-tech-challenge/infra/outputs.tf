# ==============================================================================
# Outputs
# ==============================================================================

output "s3_bucket_name" {
  description = "Nome do bucket S3 para datasets e modelos"
  value       = aws_s3_bucket.data_bucket.id
}

output "s3_bucket_arn" {
  description = "ARN do bucket S3"
  value       = aws_s3_bucket.data_bucket.arn
}

output "sagemaker_role_arn" {
  description = "ARN da IAM Role usada pelo SageMaker"
  value       = aws_iam_role.sagemaker_execution_role.arn
}

output "notebook_instance_name" {
  description = "Nome da instância de notebook do SageMaker"
  value       = aws_sagemaker_notebook_instance.notebook.name
}

output "notebook_instance_url" {
  description = "URL do notebook do SageMaker"
  value       = aws_sagemaker_notebook_instance.notebook.url
}

output "data_s3_uri" {
  description = "URI S3 para o diretório de dados"
  value       = "s3://${aws_s3_bucket.data_bucket.id}/data/"
}

output "models_s3_uri" {
  description = "URI S3 para o diretório de modelos"
  value       = "s3://${aws_s3_bucket.data_bucket.id}/models/"
}

output "cloudwatch_log_group" {
  description = "CloudWatch Log Group para logs do orquestrador"
  value       = aws_cloudwatch_log_group.train_and_deploy.name
}
