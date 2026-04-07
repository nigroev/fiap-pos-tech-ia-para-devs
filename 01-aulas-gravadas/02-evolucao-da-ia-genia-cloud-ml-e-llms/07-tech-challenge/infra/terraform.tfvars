aws_region             = "sa-east-1"
project_name           = "avc-stroke-prediction"
environment            = "dev"
notebook_instance_type = "ml.m5.xlarge"
notebook_volume_size   = 20
training_instance_type = "ml.m5.large"
endpoint_instance_type = "ml.t2.large"
s3_force_destroy       = true

# HPO Tuning Job
hpo_max_jobs          = 20
hpo_max_parallel_jobs = 4

# Algoritmo Genético
ga_population  = 10
ga_generations = 5

# Autopilot
autopilot_max_candidates  = 3
autopilot_timeout_minutes = 45

# Training Job timeouts
training_max_run_seconds       = 1800 # 30min por job
training_max_spot_wait_seconds = 3600 # 60min de espera spot

# Modo desenvolvimento: true = Terraform calcula valores reduzidos automaticamente
# (HPO 3 jobs, GA pop=4 gen=3, Autopilot timeout=20min, max-run=600s)
dev_mode = true

# Pular deploy dos endpoints após o treinamento
skip_deploy = false
