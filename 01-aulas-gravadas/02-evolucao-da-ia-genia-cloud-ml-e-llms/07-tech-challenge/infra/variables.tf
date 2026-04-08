# ==============================================================================
# Variáveis Gerais
# ==============================================================================

variable "aws_region" {
  description = "Região AWS onde os recursos serão criados"
  type        = string
  default     = "sa-east-1"
}

variable "project_name" {
  description = "Nome do projeto (usado como prefixo para nomenclatura dos recursos)"
  type        = string
  default     = "avc-stroke-prediction"
}

variable "environment" {
  description = "Ambiente de deploy (dev, staging, prod)"
  type        = string
  default     = "dev"
}

# ==============================================================================
# SageMaker
# ==============================================================================

variable "notebook_instance_type" {
  description = "Tipo de instância EC2 para o SageMaker Notebook"
  type        = string
  default     = "ml.m5.xlarge"
}

variable "notebook_volume_size" {
  description = "Tamanho do volume EBS (GB) do notebook"
  type        = number
  default     = 20
}

variable "training_instance_type" {
  description = "Tipo de instância EC2 para o treinamento no SageMaker"
  type        = string
  default     = "ml.m5.xlarge"
}

variable "training_max_run_seconds" {
  description = "Tempo máximo de execução do Training Job em segundos (Managed Spot)"
  type        = number
  default     = 1800
}

variable "training_max_spot_wait_seconds" {
  description = "Tempo máximo de espera por instância spot em segundos (deve ser >= training_max_run_seconds)"
  type        = number
  default     = 3600
}

variable "endpoint_instance_type" {
  description = "Tipo de instância EC2 para o endpoint de inferência"
  type        = string
  default     = "ml.t2.large"
}

variable "endpoint_initial_instance_count" {
  description = "Número inicial de instâncias do endpoint"
  type        = number
  default     = 1
}

# ==============================================================================
# HPO Tuning Job
# ==============================================================================

variable "hpo_max_jobs" {
  description = "Número máximo de jobs no Hyperparameter Tuning"
  type        = number
  default     = 20
}

variable "hpo_max_parallel_jobs" {
  description = "Número máximo de jobs paralelos no Hyperparameter Tuning"
  type        = number
  default     = 4
}

# ==============================================================================
# Algoritmo Genético
# ==============================================================================

variable "ga_population" {
  description = "Tamanho da população do Algoritmo Genético"
  type        = number
  default     = 10
}

variable "ga_generations" {
  description = "Número de gerações do Algoritmo Genético"
  type        = number
  default     = 5
}

# ==============================================================================
# Autopilot
# ==============================================================================

variable "autopilot_max_candidates" {
  description = "Número máximo de modelos candidatos no Autopilot AutoML"
  type        = number
  default     = 3
}

variable "autopilot_timeout_minutes" {
  description = "Timeout em minutos para aguardar Autopilot (0 = sem limite)"
  type        = number
  default     = 45
}

# ==============================================================================
# S3
# ==============================================================================

variable "s3_force_destroy" {
  description = "Permitir destruir o bucket S3 mesmo com objetos dentro"
  type        = bool
  default     = true
}

# ==============================================================================
# Modo desenvolvimento
# ==============================================================================

variable "dev_mode" {
  description = "Ativa modo desenvolvimento: reduz HPO jobs, GA pop/gen e desativa Autopilot para execução rápida"
  type        = bool
  default     = true
}

variable "skip_deploy" {
  description = "Pular deploy dos endpoints após o treinamento"
  type        = bool
  default     = false
}
