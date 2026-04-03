aws_region             = "sa-east-1"
project_name           = "avc-stroke-prediction"
environment            = "dev"
notebook_instance_type = "ml.m5.xlarge"
notebook_volume_size   = 20
training_instance_type = "ml.m5.large"
endpoint_instance_type = "ml.t3.medium"
s3_force_destroy       = true

# Modo desenvolvimento: true = execução rápida (HPO 1 job, GA pop=2 gen=2, sem Autopilot)
dev_mode = true

# Pular deploy dos endpoints após o treinamento
skip_deploy = false
