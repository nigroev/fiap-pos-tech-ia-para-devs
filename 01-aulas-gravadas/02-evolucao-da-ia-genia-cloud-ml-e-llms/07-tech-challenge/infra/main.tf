data "aws_caller_identity" "current" {}

locals {
  account_id  = data.aws_caller_identity.current.account_id
  bucket_name = "${var.project_name}-${var.environment}-${local.account_id}-data"
  name_prefix = "${var.project_name}-${var.environment}"

  # Pipeline parameters — dev_mode aplica overrides para validação rápida
  hpo_max_jobs             = var.dev_mode ? 3 : var.hpo_max_jobs
  hpo_max_parallel_jobs    = var.dev_mode ? 3 : var.hpo_max_parallel_jobs
  ga_pop                   = var.dev_mode ? 4 : var.ga_population
  ga_gen                   = var.dev_mode ? 3 : var.ga_generations
  autopilot_max_candidates = var.dev_mode ? 3 : var.autopilot_max_candidates
  autopilot_timeout        = var.dev_mode ? 20 : var.autopilot_timeout_minutes
  max_run                  = var.dev_mode ? 600 : var.training_max_run_seconds
  max_spot_wait            = var.dev_mode ? 900 : var.training_max_spot_wait_seconds
}
