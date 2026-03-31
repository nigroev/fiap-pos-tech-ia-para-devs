# ==============================================================================
# SageMaker Model, Endpoint Configuration e Endpoint
#
# NOTA: O modelo é registrado APÓS o treinamento executado pelo notebook/script.
# Este arquivo cria a infraestrutura de endpoint que será atualizada pelo
# script de treinamento após gerar o model artifact no S3.
#
# O fluxo completo é:
#   1. Terraform cria o notebook + S3 + scripts
#   2. O lifecycle on_start executa o treinamento (train_and_deploy.py)
#   3. O script registra o modelo no SageMaker e cria/atualiza o endpoint
#
# Os recursos abaixo são criados pelo script Python (via SageMaker SDK),
# não diretamente pelo Terraform, pois dependem do artefato de treinamento.
# ==============================================================================

# O endpoint é criado programaticamente pelo script train_and_deploy.py
# usando o SageMaker Python SDK. Veja scripts/train_and_deploy.py para detalhes.
#
# Para importar um endpoint existente no Terraform após a criação:
#   terraform import aws_sagemaker_endpoint.inference <endpoint-name>
