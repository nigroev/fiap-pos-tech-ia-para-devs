# Infraestrutura Terraform — Amazon SageMaker (AVC Stroke Prediction)

Infraestrutura como código (IaC) para deploy completo do pipeline de ML de predição de AVC no Amazon SageMaker.

## Arquitetura

```
┌──────────────────────────────────────────────────────────────────┐
│                          AWS Account                             │
│                                                                  │
│  ┌─────────────┐     ┌──────────────────┐     ┌──────────────┐  │
│  │   S3 Bucket  │────▶│ SageMaker Notebook│────▶│   SageMaker  │  │
│  │              │     │   Instance        │     │   Endpoint   │  │
│  │  - data/     │     │                  │     │  (Inferência) │  │
│  │  - models/   │◀────│  train_and_deploy │     │              │  │
│  │  - scripts/  │     │     .py          │     └──────────────┘  │
│  │  - output/   │     └──────────────────┘                       │
│  └─────────────┘                                                 │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    IAM Role                                  │ │
│  │  - S3 Access  - SageMaker  - CloudWatch  - ECR              │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## Recursos Criados

| Recurso | Descrição |
|---------|-----------|
| **S3 Bucket** | Armazena datasets NHANES processados, modelos treinados e scripts |
| **IAM Role** | Role com permissões para SageMaker, S3, CloudWatch e ECR |
| **SageMaker Notebook Instance** | Instância EC2 gerenciada para desenvolvimento e treinamento |
| **Lifecycle Configuration** | Scripts que configuram e executam o pipeline automaticamente |
| **SageMaker Endpoint** | Endpoint de inferência em tempo real (criado pelo script de deploy) |
| **CloudWatch Log Group** | Logs em tempo real do orquestrador (`/aws/sagemaker/<prefix>/train-and-deploy`) |

## Estrutura de Pastas

```
infra/
├── main.tf                          # Configuração principal (locals, data sources)
├── variables.tf                     # Variáveis de entrada
├── outputs.tf                       # Outputs (URLs, ARNs, nomes)
├── providers.tf                     # Provider AWS
├── versions.tf                      # Versão do Terraform e providers
├── iam.tf                           # IAM Role e policies do SageMaker
├── s3.tf                            # Bucket S3 + uploads
├── sagemaker_notebook.tf            # Notebook instance + lifecycle config
├── sagemaker_endpoint.tf            # Documentação do endpoint (criado via SDK)
├── cloudwatch.tf                    # CloudWatch Log Group para logs do orquestrador
├── terraform.tfvars.example         # Exemplo de variáveis
├── bootstrap.sh                     # Setup: limpa jobs + terraform apply
├── destroy.sh                       # Teardown completo da infraestrutura
├── stop_start_notebook.sh           # Para/inicia notebook + dependências
├── scripts/
│   ├── on_create.sh                 # Lifecycle: instalação inicial
│   ├── on_start.sh                  # Lifecycle: dispara treinamento
│   ├── train_and_deploy.py          # Orquestrador (CLI + 8 fases)
│   ├── train.py                     # Script de treinamento (SageMaker container)
│   ├── requirements.txt             # Dependências Python
│   ├── pipeline/                    # Módulos do orquestrador
│   │   ├── __init__.py              # Exports do pacote
│   │   ├── config.py                # Logging, BRT timezone, diretório de treino
│   │   ├── data_ingestion.py        # Coleta NHANES, pré-processamento, upload S3
│   │   ├── feature_store.py         # Feature Group offline (criação, ingestão)
│   │   ├── autopilot.py             # AutoML (lançamento assíncrono, polling)
│   │   ├── training.py              # HPO Tuning Job, GA Training Job
│   │   ├── sagemaker_pipeline.py    # SageMaker Pipeline (TuningStep + TrainingStep)
│   │   ├── deployment.py            # Deploy e gestão de endpoints
│   │   └── metrics.py               # Experiments e métricas consolidadas
│   └── inference_src/
│       └── inference.py             # Script de inferência do endpoint
└── README.md                        # Este arquivo
```

## Pré-requisitos

- [Terraform](https://www.terraform.io/downloads) >= 1.5.0
- [AWS CLI](https://aws.amazon.com/cli/) configurado com credenciais válidas
- Conta AWS com permissões para criar os recursos acima

## Como Usar

### 1. Configurar variáveis

```bash
cp terraform.tfvars.example terraform.tfvars
# Edite terraform.tfvars com seus valores
```

### 2. Inicializar Terraform

```bash
terraform init
```

### 3. Verificar o plano de execução

```bash
terraform plan
```

### 4. Aplicar a infraestrutura

```bash
terraform apply
```

### 5. Acessar o notebook

Após o `apply`, o output `notebook_instance_url` exibirá a URL do SageMaker Notebook. O pipeline de treinamento será disparado automaticamente pelo lifecycle script `on_start`.

### 6. Monitorar o treinamento

No SageMaker Notebook, abra um terminal e execute:

```bash
tail -f /home/ec2-user/SageMaker/train_and_deploy.log
```

Ou via **CloudWatch Logs** (tempo real, sem acesso ao notebook):

```bash
# Ver log streams disponíveis
aws logs describe-log-streams \
  --log-group-name /aws/sagemaker/avc-stroke-prediction-dev/train-and-deploy \
  --region sa-east-1 --query 'logStreams[*].logStreamName'

# Acompanhar logs em tempo real
aws logs tail /aws/sagemaker/avc-stroke-prediction-dev/train-and-deploy \
  --region sa-east-1 --follow
```

Os logs também podem ser acessados pelo **Console AWS** em CloudWatch → Log Groups → `/aws/sagemaker/avc-stroke-prediction-dev/train-and-deploy`.

### 7. Testar o endpoint

```python
import boto3, json

runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")

payload = json.dumps({
    "age": 65,
    "sbp": 140.0,
    "hba1c": 6.5,
    "bmi": 28.0,
    "gender": 1.0,
    "married": 1.0,
    "high_bp": 1.0,
    "chf": 0.0,
    "occupation": 2.0,
    "smoking": 1.0
})

response = runtime.invoke_endpoint(
    EndpointName="avc-stroke-prediction-endpoint",
    ContentType="application/json",
    Body=payload
)

result = json.loads(response["Body"].read().decode())
print(result)
```

### 8. Parar/Iniciar notebook (economia de custo)

```bash
# Ver estado atual de todos os recursos
bash stop_start_notebook.sh status

# Parar tudo (jobs, endpoints, notebook)
bash stop_start_notebook.sh stop

# Iniciar notebook (on_start.sh roda automaticamente)
bash stop_start_notebook.sh start
```

## Destruir Infraestrutura

```bash
bash destroy.sh   # ou: terraform destroy
```

> **Atenção:** O bucket S3 será destruído mesmo com objetos dentro (configurado com `force_destroy = true`). Altere `s3_force_destroy = false` em produção.

## Pipeline de ML (Fluxo do Script)

1. **Coleta** — Download dos dados NHANES (CDC) de 7 ciclos (2005-2018)
2. **Pré-processamento** — Seleção de features, binarização, merge de módulos
3. **Treinamento baseline** — LogisticRegression + RandomForest com class_weight balanced
4. **Otimização GA** — Algoritmo Genético para tuning de hiperparâmetros do RF
5. **Avaliação** — Métricas F-beta (β=1.5), ROC AUC, Classification Report
6. **Persistência** — Upload do modelo (model.tar.gz) para S3
7. **Deploy** — Criação de endpoint no SageMaker para inferência em tempo real

## Variáveis Configuráveis

| Variável | Default | Descrição |
|----------|---------|-----------|
| `aws_region` | `us-east-1` | Região AWS |
| `project_name` | `avc-stroke-prediction` | Prefixo de nomenclatura |
| `environment` | `dev` | Ambiente (dev/staging/prod) |
| `notebook_instance_type` | `ml.t3.medium` | Tipo de instância do notebook |
| `notebook_volume_size` | `20` | Volume EBS em GB |
| `training_instance_type` | `ml.m5.large` | Instância para training jobs |
| `endpoint_instance_type` | `ml.t2.medium` | Instância do endpoint |
| `s3_force_destroy` | `true` | Destruir bucket com objetos |
