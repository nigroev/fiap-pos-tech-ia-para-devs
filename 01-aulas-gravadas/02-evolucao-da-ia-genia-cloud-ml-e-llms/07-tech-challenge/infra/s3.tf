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
