# ==============================================================================
# Remote State — Bootstrap externo
# ==============================================================================
#
# O bucket S3 de remote state é criado UMA ÚNICA VEZ pelo script:
#   scripts/bootstrap_state.sh
#
# Ele é gerenciado FORA do Terraform propositalmente para evitar o problema
# de ovo-e-galinha (bucket precisa existir antes do terraform init).
#
# Após executar o bootstrap, o backend fica configurado em backend.hcl e
# nunca precisa ser alterado para apply/destroy normais.
# ==============================================================================
