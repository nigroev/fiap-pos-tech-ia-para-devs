# ==============================================================================
# SageMaker Notebook Instance
# ==============================================================================

resource "aws_sagemaker_notebook_instance" "notebook" {
  name                  = "${local.name_prefix}-notebook"
  instance_type         = var.notebook_instance_type
  role_arn              = aws_iam_role.sagemaker_execution_role.arn
  volume_size           = var.notebook_volume_size
  lifecycle_config_name = aws_sagemaker_notebook_instance_lifecycle_configuration.notebook_lc.name

  direct_internet_access = "Enabled"

  tags = {
    Name = "${local.name_prefix}-notebook"
  }
}

# ==============================================================================
# Lifecycle Configuration — on-create e on-start
# ==============================================================================

resource "aws_sagemaker_notebook_instance_lifecycle_configuration" "notebook_lc" {
  name = "${local.name_prefix}-lc"

  on_create = base64encode(templatefile("${path.module}/scripts/on_create.sh", {
    bucket_name = local.bucket_name
    region      = var.aws_region
    project     = var.project_name
  }))

  on_start = base64encode(templatefile("${path.module}/scripts/on_start.sh", {
    bucket_name = local.bucket_name
    region      = var.aws_region
    project     = var.project_name
    dev_mode    = var.dev_mode ? "true" : "false"
    skip_deploy = var.skip_deploy ? "true" : "false"
  }))
}
