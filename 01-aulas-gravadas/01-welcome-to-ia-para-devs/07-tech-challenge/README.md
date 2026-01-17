# Tech Challenge - Fase 1: Diagnóstico de Acidente Vascular Cerebral (AVC) com Machine Learning

## Visão Geral do Projeto

Este projeto implementa um sistema inteligente de suporte ao diagnóstico para auxiliar na identificação de pacientes com risco de **Acidente Vascular Cerebral (AVC)** utilizando dados estruturados do NHANES (National Health and Nutrition Examination Survey). O foco é construir uma solução inicial baseada em **Machine Learning** que classifique pacientes como tendo ou não AVC, apoiando (mas não substituindo) decisões clínicas.

### Objetivo
Construir uma solução com foco em IA para processamento de dados médicos, aplicando fundamentos essenciais de Machine Learning (ML) e análise exploratória de dados (EDA), demonstrando:
- Exploração e tratamento de dados médicos reais
- Pipeline robusto de pré-processamento
- Modelagem com múltiplas técnicas de classificação
- Interpretação e comunicação de resultados

---

## Instruções de Execução

### Pré-requisitos
- Python 3.10+
- pip (gerenciador de pacotes Python)
- Acesso à internet (para download dos dados NHANES)

### Instalação e Configuração

1. **Clonar ou acessar o repositório:**
   ```bash
   cd /home/user/workspaces/fiap-pos-tech-ia-para-devs/01-aulas-gravadas/01-welcome-to-ia-para-devs/07-tech-challenge
   ```

2. **Criar e ativar um ambiente virtual (recomendado):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```

3. **Instalar dependências:**
   ```bash
   pip install -r requirements.txt
   ```


4. **Executar o notebook:**
   ```bash
   jupyter notebook tech-challenge-fase-1.ipynb
   ```

   Ou, se usar JupyterLab:
   ```bash
   jupyter lab tech-challenge-fase-1.ipynb
   ```

5. **Executar todas as células:**
   - Navegue até `Cell > Run All` ou pressione `Ctrl+Shift+Enter`
   - A execução pode levar ~10-15 minutos na primeira vez (carregamento dos dados NHANES)

---

## API REST para Predições

### Executando a API FastAPI

Após treinar o modelo no notebook e gerar o arquivo `pipe_lr_model.pkl`, você pode iniciar a API para realizar predições em tempo real:

```bash
# Certifique-se de estar no diretório do projeto com o ambiente virtual ativado
python -m fastapi dev main.py
```

A API estará disponível em: **http://localhost:8000**

### Documentação Interativa (Swagger UI)

Acesse a interface interativa para testar os endpoints:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### Endpoints Disponíveis

#### 1. **GET /** — Health Check
Verifica se a API está funcionando.

**Request:**
```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "Hello": "World"
}
```

#### 2. **POST /predict** — Predição de Risco de AVC

Recebe dados de um paciente e retorna a predição de risco de AVC.

**Request Body (JSON):**
```json
{
  "age": 67,
  "sbp": 128.0,
  "hba1c": 9.2,
  "bmi": 32.0,
  "gender": 1,
  "married": 1.0,
  "high_bp": 1,
  "chf": 0,
  "occupation": 5.0,
  "smoking": 1
}
```

**Descrição dos Campos:**
| Campo | Tipo | Descrição | Valores |
|---|---|---|---|
| `age` | int | Idade em anos | 0-150 |
| `sbp` | float | Pressão arterial sistólica (mmHg) | 72-228 |
| `hba1c` | float | Hemoglobina glicada (%) | - |
| `bmi` | float | Índice de Massa Corporal | - |
| `gender` | float | Gênero | 1 = Masculino, 0 = Feminino |
| `married` | float | Estado civil | 1 = Já foi casado, 0 = Nunca casou |
| `high_bp` | float | Histórico de hipertensão | 1 = Sim, 0 = Não |
| `chf` | float | Insuficiência cardíaca congestiva | 1 = Sim, 0 = Não |
| `occupation` | float | Situação profissional | 1-5 (categorias) |
| `smoking` | float | Histórico de tabagismo | 1 = Sim, 0 = Não |

**Response (JSON):**
```json
{
  "prediction_stroke": 0,
  "probability_no_stroke": 0.8234,
  "probability_stroke": 0.1766,
  "input": {
    "age": 67,
    "sbp": 128.0,
    "hba1c": 9.2,
    "bmi": 32.0,
    "gender": 1,
    "married": 1.0,
    "high_bp": 1,
    "chf": 0,
    "occupation": 5.0,
    "smoking": 1
  }
}
```

**Interpretação da Response:**
- `prediction_stroke`: **0** = Sem risco de AVC, **1** = Com risco de AVC
- `probability_no_stroke`: Probabilidade de NÃO ter AVC (0-1)
- `probability_stroke`: Probabilidade de TER AVC (0-1)
- `input`: Eco dos dados enviados para validação


### Integração com Sistemas Clínicos

A API foi projetada para ser integrada em sistemas hospitalares/clínicos:

1. **Sistema de Prontuário Eletrônico (EMR):** Envie dados do paciente após consulta/exame
2. **Sistema de Triagem:** Use para priorizar pacientes de alto risco
3. **Dashboard de Acompanhamento:** Monitore tendências de risco em populações

**⚠️ IMPORTANTE - USO CLÍNICO:**
- Esta API é uma **ferramenta de apoio à decisão**, NÃO um substituto do julgamento médico
- Sempre confirme casos de alto risco com avaliação clínica especializada e exames de imagem (CT/MRI)
- Considere falsos positivos/negativos nas decisões clínicas

### Estrutura de Arquivos
```
tech-challenge-fase-1/
├── tech-challenge-fase-1.ipynb    # Notebook principal com EDA, modelagem e avaliação
├── main.py                         # API FastAPI para predições em tempo real
├── pipe_lr_model.pkl              # Modelo Logistic Regression treinado (serializado)
├── requirements.txt                # Dependências do projeto
└── README.md                       # Este arquivo
```

---

## Dataset

### Fonte
**NHANES (National Health and Nutrition Examination Survey)**  
- Repositório oficial: https://www.cdc.gov/nchs/nhanes
- Acesso aos dados: https://wwwn.cdc.gov/Nchs/Nhanes/

### Características do Dataset
- **Origem:** CDC (Centers for Disease Control and Prevention) - Estados Unidos
- **Tipo:** Estudo observacional, transversal com amostragem probabilística
- **Período:** 4 ciclos bienais (2011–2012, 2013–2014, 2015–2016, 2017–2018)
- **Módulos utilizados:**
  - `DEMO`: Dados demográficos (idade, gênero, estado civil, ocupação)
  - `BPX`: Medidas de pressão arterial
  - `BPQ`: Questionário de pressão arterial e histórico de hipertensão
  - `GHB`: Hemoglobina glicada (marcador de diabetes)
  - `BMX`: Medidas corporais (IMC, peso, altura)
  - `SMQ`: Questionário de tabagismo
  - `MCQ`: Questionário médico (histórico de doenças, incluindo AVC)

### Variáveis Selecionadas (n=12 features + target)
| Código Original | Nome Renomeado | Tipo | Descrição |
|---|---|---|---|
| SEQN | SEQN_id | Numérico | ID único do participante |
| RIAGENDR | RIAGENDR_gender | Categórico | Gênero (1=Masculino, 2=Feminino) |
| RIDAGEYR | RIDAGEYR_age | Numérico | Idade em anos (0–150) |
| BPQ020 | BPQ020_high_bp | Categórico | Histórico de hipertensão (1=Sim, 2=Não) |
| MCQ160B | MCQ160B_chf | Categórico | Insuficiência cardíaca congestiva (1=Sim, 2=Não) |
| DMDMARTL | DMDMARTL_marital | Categórico | Estado civil (1–6) |
| OCQ260 | OCQ260_occupation | Categórico | Situação profissional (1–6) |
| BPXSY1 | BPXSY1_sbp | Numérico | Pressão arterial sistólica (72–228 mmHg) |
| LBXGH | LBXGH_hba1c | Numérico | Hemoglobina glicada (%) |
| BMXBMI | BMXBMI_bmi | Numérico | Índice de Massa Corporal |
| SMQ020 | SMQ020_smoking | Categórico | Histórico de tabagismo (1=Sim, 2=Não) |
| **MCQ160F** | **MCQ160F_stroke** | **Categórico** | **ALVO: Histórico de AVC (1=Sim, 2=Não)** |

### Download Automático
O notebook carrega os dados automaticamente via URLs do CDC. **Sem necessidade de download manual.**

### Tamanho e Prevalência
- **Amostra inicial:** ~20,000+ participantes (múltiplos ciclos)
- **Amostra final (após limpeza):** ~14,000+ registros válidos
- **Prevalência de AVC:** ~4–5% (classe minoritária — desbalanceada)

---

## Resultados Obtidos

### Resumo Executivo
- **Dataset**: Carregado, explorado e limpo com sucesso (~14,000+ registros válidos)
- **EDA**: Visualizações de correlação, distribuições, taxas por grupo
- **Pré-processamento**: Pipeline robusto implementado (imputação + scaling + encoding)
- **Modelos**: Regressão Logística e Random Forest treinados e avaliados
- **Métricas**: ROC AUC, PR AUC, F1-score, Recall — todas calculadas
- **Interpretação**: Importância por permutação implementada
- **Produtização**: Modelo serializado (pickle) e API REST (FastAPI) funcionando
- **Balanceamento**: Dataset balanceado via undersampling para melhorar Recall

### Principais Achados

#### 1. **Exploração de Dados (EDA)**
- **Idade:** Distribuição normal; pacientes com AVC tendem a ser ~10 anos mais velhos
- **Pressão arterial sistólica (sbp):** Forte preditor visual — valores mais altos associados a AVC
- **Fatores de risco:** Hipertensão, insuficiência cardíaca e tabagismo mostram correlação positiva com AVC
- **Balanceamento:** Dataset desbalanceado (~95% sem AVC, ~5% com AVC) → métricas como Recall e PR AUC são críticas

#### 2. **Resultados de Modelagem**

**Importante:** Os modelos foram treinados com **dataset balanceado via undersampling** para melhorar a detecção de casos positivos (AVC).

**Regressão Logística (Baseline - Base Balanceada):**
- ROC AUC: ~0.78
- Recall: ~0.68 (captura ~68% dos verdadeiros positivos)
- Precisão: ~0.15 (muitos falsos positivos)
- F1-score: ~0.25
- **Modelo escolhido para produção** (API FastAPI)

**Random Forest (Melhor desempenho - Base Balanceada):**
- ROC AUC: ~0.82 ⭐
- Recall: ~0.72 (captura ~72% dos AVC verdadeiros)
- Precisão: ~0.18 (melhorado vs. LR)
- F1-score: ~0.30

**Estratégia de Balanceamento:**
- **Método:** Undersampling da classe majoritária (sem AVC)
- **Justificativa:** Dataset original tinha ~95% sem AVC, ~5% com AVC (desbalanceamento extremo)
- **Impacto:** Redução de ~14,000 para ~600 registros, mas melhoria significativa em Recall
- **Alternativa aplicada:** `class_weight='balanced'` nos modelos para ajuste automático

**⚠️ Nota sobre Undersampling:**
- Vantagem: Melhora Recall (crítico para diagnóstico médico)
- Desvantagem: Perda de informação da classe majoritária
- Produção: Modelo treinado em base balanceada foi serializado em `pipe_lr_model.pkl`

#### 3. **Importância das Features**
**Top 5 features mais importantes (por permutação):**
1. Idade (RIDAGEYR_age) — 0.032
2. Pressão arterial sistólica (BPXSY1_sbp) — 0.025
3. Hemoglobina glicada (LBXGH_hba1c) — 0.018
4. Histórico de insuficiência cardíaca (MCQ160B_chf) — 0.016
5. Histórico de hipertensão (BPQ020_high_bp) — 0.014

#### 4. **Implicações Clínicas**
- **Modelo é viável para triagem inicial**: Recall ~72% significa capturar ~7 em 10 pacientes com AVC
- **Precisão baixa**: Muitos falsos positivos (necessário confirmar com especialista)
- **Uso recomendado**: **Ferramenta de apoio à decisão clínica, não substituição** do julgamento médico

---

## Relatório Técnico


#### 3.2 Interpretação de Resultados

**RF vs. LR:**
```
            Logistic Regression    Random Forest
ROC AUC              0.78              0.82  ⭐
Recall               0.68              0.72  ⭐
Precisão             0.14              0.18  ⭐
F1-score             0.25              0.30  ⭐
```

**RF é superior** em todas as métricas. Ganho de ROC AUC de +0.04 é relevante em diagnóstico médico.

#### 3.3 Importância de Features (Top 10)

**Método:** Permutação importance com 30 repetiçõesScoring: ROC AUC (alinhado com métrica principal)

1. **RIDAGEYR_age** (0.032) — Idade é o fator mais importante
   - Interpretação: AVC aumenta exponencialmente com idade (fisiologia cardiocerebral)
   
2. **BPXSY1_sbp** (0.025) — Pressão arterial sistólica
   - Interpretação: Hipertensão = fator de risco major para AVC
   
3. **LBXGH_hba1c** (0.018) — Hemoglobina glicada (diabetes)
   - Interpretação: Diabetes eleva risco de AVC
   
4. **MCQ160B_chf** (0.016) — Insuficiência cardíaca
   - Interpretação: Doença cardíaca comórbida aumenta risco
   
5. **BPQ020_high_bp** (0.014) — Histórico de hipertensão
   - Interpretação: Auto-relato confirma a importância de BP

**Insights:**
- Features biomédicas (idade, pressão, glicemia) dominam
- Fatores sociodemográficos (gênero, estado civil) têm impacto menor
- Combinar idade + pressão + glicemia captura ~70% da importância total

---

## Referências

1. CDC NHANES — https://www.cdc.gov/nchs/nhanes
2. Documentação scikit-learn — https://scikit-learn.org
3. Pandascouple. Projeto ML Previsão de AVC — https://pandascouple.medium.com/projeto-machine-learning-previs%C3%A3o-de-avc-f4b7dce11929
4. USP Rádio. Uso de IA e análise de dados na prevenção de AVC — https://jornal.usp.br/radio-usp/uso-de-ia-e-analise-de-dados-na-prevencao-de-avc-e-ataque-isquemico-transitorio/
5. Nature Research. Machine learning para risco cardiovascular — https://www.nature.com/articles/s41598-024-61665-4
6. Journal of Health Informatics. AVC diagnosis — https://jhi.sbis.org.br/index.php/jhi-sbis/article/view/980
7. Nature Research 2025. AVC modeling — https://www.nature.com/articles/s41598-025-01855-w

---
