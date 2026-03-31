data "aws_caller_identity" "current" {}

locals {
  account_id  = data.aws_caller_identity.current.account_id
  bucket_name = "${var.project_name}-${var.environment}-${local.account_id}"
  name_prefix = "${var.project_name}-${var.environment}"
}
