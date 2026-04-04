# ==============================================================================
# S3 Bucket — Armazenamento de datasets, modelos e artefatos
# ==============================================================================

resource "aws_s3_bucket" "data_bucket" {
  bucket        = local.bucket_name
  force_destroy = var.s3_force_destroy
}

resource "aws_s3_bucket_versioning" "data_bucket_versioning" {
  bucket = aws_s3_bucket.data_bucket.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data_bucket_encryption" {
  bucket = aws_s3_bucket.data_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "data_bucket_public_access" {
  bucket = aws_s3_bucket.data_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ==============================================================================
# Upload do script de treinamento e deploy para o S3
# ==============================================================================

resource "aws_s3_object" "training_script" {
  bucket = aws_s3_bucket.data_bucket.id
  key    = "scripts/train_and_deploy.py"
  source = "${path.module}/scripts/train_and_deploy.py"
  etag   = filemd5("${path.module}/scripts/train_and_deploy.py")
}

resource "aws_s3_object" "train_script" {
  bucket = aws_s3_bucket.data_bucket.id
  key    = "scripts/train.py"
  source = "${path.module}/scripts/train.py"
  etag   = filemd5("${path.module}/scripts/train.py")
}

resource "aws_s3_object" "requirements" {
  bucket = aws_s3_bucket.data_bucket.id
  key    = "scripts/requirements.txt"
  source = "${path.module}/scripts/requirements.txt"
  etag   = filemd5("${path.module}/scripts/requirements.txt")
}

resource "aws_s3_object" "inference_script" {
  bucket = aws_s3_bucket.data_bucket.id
  key    = "scripts/inference_src/inference.py"
  source = "${path.module}/scripts/inference_src/inference.py"
  etag   = filemd5("${path.module}/scripts/inference_src/inference.py")
}

# ==============================================================================
# Upload do pacote pipeline/ (módulos do orquestrador)
# ==============================================================================

resource "aws_s3_object" "pipeline_init" {
  bucket = aws_s3_bucket.data_bucket.id
  key    = "scripts/pipeline/__init__.py"
  source = "${path.module}/scripts/pipeline/__init__.py"
  etag   = filemd5("${path.module}/scripts/pipeline/__init__.py")
}

resource "aws_s3_object" "pipeline_config" {
  bucket = aws_s3_bucket.data_bucket.id
  key    = "scripts/pipeline/config.py"
  source = "${path.module}/scripts/pipeline/config.py"
  etag   = filemd5("${path.module}/scripts/pipeline/config.py")
}

resource "aws_s3_object" "pipeline_data_ingestion" {
  bucket = aws_s3_bucket.data_bucket.id
  key    = "scripts/pipeline/data_ingestion.py"
  source = "${path.module}/scripts/pipeline/data_ingestion.py"
  etag   = filemd5("${path.module}/scripts/pipeline/data_ingestion.py")
}

resource "aws_s3_object" "pipeline_feature_store" {
  bucket = aws_s3_bucket.data_bucket.id
  key    = "scripts/pipeline/feature_store.py"
  source = "${path.module}/scripts/pipeline/feature_store.py"
  etag   = filemd5("${path.module}/scripts/pipeline/feature_store.py")
}

resource "aws_s3_object" "pipeline_autopilot" {
  bucket = aws_s3_bucket.data_bucket.id
  key    = "scripts/pipeline/autopilot.py"
  source = "${path.module}/scripts/pipeline/autopilot.py"
  etag   = filemd5("${path.module}/scripts/pipeline/autopilot.py")
}

resource "aws_s3_object" "pipeline_training" {
  bucket = aws_s3_bucket.data_bucket.id
  key    = "scripts/pipeline/training.py"
  source = "${path.module}/scripts/pipeline/training.py"
  etag   = filemd5("${path.module}/scripts/pipeline/training.py")
}

resource "aws_s3_object" "pipeline_sagemaker_pipeline" {
  bucket = aws_s3_bucket.data_bucket.id
  key    = "scripts/pipeline/sagemaker_pipeline.py"
  source = "${path.module}/scripts/pipeline/sagemaker_pipeline.py"
  etag   = filemd5("${path.module}/scripts/pipeline/sagemaker_pipeline.py")
}

resource "aws_s3_object" "pipeline_deployment" {
  bucket = aws_s3_bucket.data_bucket.id
  key    = "scripts/pipeline/deployment.py"
  source = "${path.module}/scripts/pipeline/deployment.py"
  etag   = filemd5("${path.module}/scripts/pipeline/deployment.py")
}

resource "aws_s3_object" "pipeline_metrics" {
  bucket = aws_s3_bucket.data_bucket.id
  key    = "scripts/pipeline/metrics.py"
  source = "${path.module}/scripts/pipeline/metrics.py"
  etag   = filemd5("${path.module}/scripts/pipeline/metrics.py")
}

# ==============================================================================
# Prefixos (pastas lógicas) — criados como objetos vazios
# ==============================================================================

resource "aws_s3_object" "dataset_prefix" {
  bucket  = aws_s3_bucket.data_bucket.id
  key     = "data/"
  content = ""
}

resource "aws_s3_object" "models_prefix" {
  bucket  = aws_s3_bucket.data_bucket.id
  key     = "models/"
  content = ""
}

resource "aws_s3_object" "output_prefix" {
  bucket  = aws_s3_bucket.data_bucket.id
  key     = "output/"
  content = ""
}

resource "aws_s3_object" "feature_store_prefix" {
  bucket  = aws_s3_bucket.data_bucket.id
  key     = "feature-store/"
  content = ""
}
