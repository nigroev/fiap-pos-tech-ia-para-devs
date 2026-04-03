# ==============================================================================
# S3 Bucket — Terraform Remote State
# ==============================================================================
#
# BOOTSTRAP: Este bucket precisa existir ANTES de configurar o backend remoto.
# Fluxo:
#   1. terraform apply (com backend local — versions.tf sem bloco backend)
#   2. Descomente o bloco backend em versions.tf
#   3. terraform init -migrate-state
#
# ==============================================================================

locals {
  state_bucket_name = "${var.project_name}-${var.environment}-tfstate-${local.account_id}"
}

resource "aws_s3_bucket" "terraform_state" {
  bucket        = local.state_bucket_name
  force_destroy = false # Protege contra destruição acidental do state

  lifecycle {
    prevent_destroy = false
  }
}

resource "aws_s3_bucket_versioning" "terraform_state_versioning" {
  bucket = aws_s3_bucket.terraform_state.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "terraform_state_encryption" {
  bucket = aws_s3_bucket.terraform_state.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "terraform_state_public_access" {
  bucket = aws_s3_bucket.terraform_state.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}