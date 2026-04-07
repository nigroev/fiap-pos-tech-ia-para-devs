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
│  │  - S3 Access  - SageMaker  - CloudWatch Metrics  - ECR       │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## Recursos Criados

| Recurso | Descrição |
|---------|-----------|
| **S3 Bucket** | Armazena datasets NHANES processados, modelos treinados e scripts |
| **IAM Role** | Role com permissões para SageMaker, S3, CloudWatch Metrics e ECR |
| **SageMaker Notebook Instance** | Instância EC2 gerenciada para desenvolvimento e treinamento |
| **Lifecycle Configuration** | Scripts que configuram e executam o pipeline automaticamente |
| **SageMaker Endpoint** | Endpoint de inferência em tempo real (criado pelo script de deploy) |

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
├── terraform.tfvars                 # Variáveis do ambiente
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

### 7. Testar o endpoint

```python
import boto3, json

runtime = boto3.client("sagemaker-runtime", region_name="sa-east-1")

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

## Última Execução Bem-Sucedida (dev_mode)

Log de execução do pipeline completo em `dev_mode = true` (07/04/2026):

| Fase | Descrição | Tempo | Status |
|------|-----------|-------|--------|
| 1/8 | Setup (Role, Session, Experiment) | 1s | OK |
| 2/8 | Ingestão NHANES + Pré-processamento | 2m25s | OK — 39.473 linhas, 11 colunas |
| 3/8 | Feature Store (ingestão offline) | 2m02s | OK — 39.473 registros |
| 4/8 | Autopilot AutoML (lançamento) | 1s | OK — assíncrono |
| 5/8 | SageMaker Pipelines (HPO + GA) | 7m09s | OK — HPO 3/3 Succeeded, GA Succeeded |
| 6/8 | Autopilot (espera + comparação) | 14m02s | OK — Autopilot F1=0.2469 |
| 7/8 | Métricas consolidadas | <1s | OK |
| 8/8 | Deploy endpoint | 3m32s | OK — `avc-stroke-prediction-endpoint` InService |
| **Total** | | **~27min** | **Pipeline concluído com sucesso** |

## Troubleshooting

Problemas encontrados e corrigidos durante o desenvolvimento:

### 1. Training Jobs falhando com "ExitCode 1 Erro" (sem detalhes)

**Causa:** Em `train.py`, o bloco `if __name__ == "__main__"` estava posicionado **antes** da definição de `_main()`. Python executa top-to-bottom, resultando em `NameError` silencioso.

**Solução:** Mover `if __name__ == "__main__"` para o **final** do arquivo, após todas as definições de função. Além disso, o error handling foi melhorado para salvar o traceback completo em `/opt/ml/output/data/failure_traceback.txt` e emitir um resumo curto no stderr.

### 2. GA Training falhando com população pequena (`dev_mode`)

**Causa:** A função `tournament_selection(k=5)` usava `k=5` hardcoded, mas em `dev_mode` o GA roda com `n_pop=4`. `np.random.choice(4, 5, replace=False)` lança `ValueError`.

**Solução:** Adicionado `k = min(k, len(population))` antes do sampling.

### 3. Deploy falhando com "Could not find model data" no S3

**Causa:** Um `from sagemaker.sklearn.estimator import SKLearn` duplicado dentro de um bloco `if` na função `main()` de `train_and_deploy.py` fazia o Python tratar `SKLearn` como variável **local** em toda a função — sombreando o import global. O `SKLearn.attach()` falhava com `UnboundLocalError`, e o estimator fallback apontava para um path S3 fictício.

**Solução:** Removido o import duplicado (o `SKLearn` já é importado no top-level). Adicionado fallback via `describe_training_job` do boto3 para recuperar o `model_data` real caso `SKLearn.attach()` falhe por qualquer motivo.

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
| `aws_region` | `sa-east-1` | Região AWS |
| `project_name` | `avc-stroke-prediction` | Prefixo de nomenclatura |
| `environment` | `dev` | Ambiente (dev/staging/prod) |
| `notebook_instance_type` | `ml.m5.xlarge` | Tipo de instância do notebook |
| `notebook_volume_size` | `20` | Volume EBS em GB |
| `training_instance_type` | `ml.m5.xlarge` | Instância para training jobs |
| `training_max_run_seconds` | `1800` | Tempo máximo por Training Job (segundos) |
| `training_max_spot_wait_seconds` | `3600` | Tempo máximo de espera por instância spot |
| `endpoint_instance_type` | `ml.t2.large` | Instância do endpoint |
| `hpo_max_jobs` | `20` | Número máximo de jobs no HPO Tuning |
| `hpo_max_parallel_jobs` | `4` | Jobs paralelos no HPO Tuning |
| `ga_population` | `10` | Tamanho da população do Algoritmo Genético |
| `ga_generations` | `5` | Número de gerações do GA |
| `autopilot_max_candidates` | `3` | Candidatos máximos do Autopilot AutoML |
| `autopilot_timeout_minutes` | `45` | Timeout do Autopilot em minutos (0 = sem limite) |
| `s3_force_destroy` | `true` | Destruir bucket com objetos |
| `dev_mode` | `true` | Ativa overrides reduzidos via `locals` no Terraform |
| `skip_deploy` | `false` | Pular deploy dos endpoints |
