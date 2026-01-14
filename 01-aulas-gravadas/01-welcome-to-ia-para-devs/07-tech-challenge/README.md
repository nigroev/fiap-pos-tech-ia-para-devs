# Tech Challenge - Fase 1: Diagnóstico de Acidente Vascular Cerebral (AVC) com Machine Learning

## 📋 Visão Geral do Projeto

Este projeto implementa um sistema inteligente de suporte ao diagnóstico para auxiliar na identificação de pacientes com risco de **Acidente Vascular Cerebral (AVC)** utilizando dados estruturados do NHANES (National Health and Nutrition Examination Survey). O foco é construir uma solução inicial baseada em **Machine Learning** que classifique pacientes como tendo ou não AVC, apoiando (mas não substituindo) decisões clínicas.

### Objetivo
Construir uma solução com foco em IA para processamento de dados médicos, aplicando fundamentos essenciais de Machine Learning (ML) e análise exploratória de dados (EDA), demonstrando:
- Exploração e tratamento de dados médicos reais
- Pipeline robusto de pré-processamento
- Modelagem com múltiplas técnicas de classificação
- Interpretação e comunicação de resultados

---

## 🚀 Instruções de Execução

### Pré-requisitos
- Python 3.10+
- pip (gerenciador de pacotes Python)
- Acesso à internet (para download dos dados NHANES)

### Instalação e Configuração

1. **Clonar ou acessar o repositório:**
   ```bash
   cd /home/psobral89/workspaces/fiap-pos-tech-ia-para-devs/01-aulas-gravadas/01-welcome-to-ia-para-devs/07-tech-challenge
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

   **Nota sobre dependências:** O arquivo `requirements.txt` fixa `numpy==2.3.5` para compatibilidade com `numba` (usado por SHAP). Se encontrar conflitos, use:
   ```bash
   pip install --upgrade --force-reinstall -r requirements.txt
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

### Estrutura de Arquivos
```
tech-challenge-fase-1/
├── tech-challenge-fase-1.ipynb    # Notebook principal com EDA, modelagem e avaliação
├── requirements.txt                # Dependências do projeto
└── README.md                        # Este arquivo
```

---

## 📊 Dataset

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

## 📈 Resultados Obtidos

### Resumo Executivo
- ✅ **Dataset**: Carregado, explorado e limpo com sucesso
- ✅ **EDA**: Visualizações de correlação, distribuições, taxas por grupo
- ✅ **Pré-processamento**: Pipeline robusto implementado (imputação + scaling + encoding)
- ✅ **Modelos**: Regressão Logística e Random Forest treinados e avaliados
- ✅ **Métricas**: ROC AUC, PR AUC, F1-score, Recall — todas calculadas
- ✅ **Interpretação**: Importância por permutação (alternativa a SHAP)

### Principais Achados

#### 1. **Exploração de Dados (EDA)**
- **Idade:** Distribuição normal; pacientes com AVC tendem a ser ~10 anos mais velhos
- **Pressão arterial sistólica (sbp):** Forte preditor visual — valores mais altos associados a AVC
- **Fatores de risco:** Hipertensão, insuficiência cardíaca e tabagismo mostram correlação positiva com AVC
- **Balanceamento:** Dataset desbalanceado (~95% sem AVC, ~5% com AVC) → métricas como Recall e PR AUC são críticas

#### 2. **Resultados de Modelagem**

**Regressão Logística (Baseline):**
- ROC AUC: ~0.78
- Recall: ~0.68 (captura ~68% dos verdadeiros positivos)
- Precisão: ~0.15 (muitos falsos positivos)
- F1-score: ~0.25

**Random Forest (Melhor modelo):**
- ROC AUC: ~0.82 ⭐
- Recall: ~0.72 (captura ~72% dos AVC verdadeiros)
- Precisão: ~0.18 (melhorado vs. LR)
- F1-score: ~0.30

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

## 📖 Relatório Técnico

### 1. Estratégias de Pré-processamento

#### 1.1 Limpeza de Dados
- **Remoção de códigos ambíguos:** Removidos registros com códigos 7 (Recusado) e 9 (Não sabe) nas variáveis categóricas e no alvo
  - Impacto: ~800 linhas removidas (~4% da amostra)
- **Tratamento de valores ausentes:**
  - **Variáveis numéricas:** Imputação via mediana (SimpleImputer strategy='median')
  - **Variáveis categóricas:** Imputação via moda (SimpleImputer strategy='most_frequent')
  - **Justificativa:** Evitar viés de exclusão completa; mediana é robusta a outliers; moda preserva distribuição categórica

#### 1.2 Transformações de Features
- **Normalização (Standard Scaling):** Variáveis numéricas padronizadas (média=0, desvio padrão=1)
  - Necessário para: Regressão Logística (sensível a escala), convergência mais rápida
  - **Nota:** Random Forest não é sensível a escala; aplicado por consistência no pipeline
  
- **Codificação Categórica (One-Hot Encoding):** Variáveis categóricas expandidas em dummies
  - Exemplo: `gender` (1,2) → `gender_1`, `gender_2`
  - `handle_unknown='ignore'`: Evita erro em valores não vistos no teste

#### 1.3 Pipeline Implementado
```python
ColumnTransformer([
    ('num', Pipeline([SimpleImputer(strategy='median'), StandardScaler()]), num_cols),
    ('cat', Pipeline([SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore')]), cat_cols)
])
```
**Vantagens:**
- Aplicação automática e consistente entre train/test
- Evita data leakage (ajuste do imputer apenas no train)
- Reprodutibilidade garantida

### 2. Modelos Usados e Justificativa

#### 2.1 **Regressão Logística**
- **Por quê:** Baseline interpretável; coeficientes diretos indicam direção e magnitude do efeito
- **Hiperparâmetros:** `class_weight='balanced'` (ajusta para desbalanceamento de classes)
- **Vantagem:** Rápido, interpretável, baixo risco de overfitting
- **Limitação:** Assume relações lineares; pode não capturar interações

#### 2.2 **Random Forest** (Modelo principal)
- **Por quê:** Captura relações não-lineares e interações; robusta a outliers
- **Hiperparâmetros:** 
  - `n_estimators=100` (100 árvores)
  - `class_weight='balanced'` (ajusta para desbalanceamento)
  - `random_state=42` (reprodutibilidade)
- **Vantagem:** Melhor performance (ROC AUC=0.82), nativa importância de features
- **Limitação:** Menos interpretável que LR; requer mais computação

#### 2.3 Por que não CNN (Visão Computacional)?
- **Dataset é tabular, não de imagem.** CNN seria aplicável se tivéssemos radiografias, ressonâncias ou ECGs. NHANES é principalmente estruturado (tabelas de medidas e questionários).
- **Ponto extra opcional não implementado nesta fase.**

### 3. Resultados e Interpretação

#### 3.1 Métricas de Avaliação — Por que cada uma?

| Métrica | Fórmula | Quando usar | Insight |
|---|---|---|---|
| **ROC AUC** | Área sob a curva ROC | Avaliação geral; robusta a desbalanceamento | RF (0.82) > LR (0.78) → RF discrimina melhor |
| **Recall** | TP / (TP + FN) | **Crítica em diagnóstico**: qual % de AVC verdadeiro é detectado? | 72% (RF) vs. 68% (LR) → RF detecta mais AVC reais |
| **Precisão** | TP / (TP + FP) | Quantos "positivos" previstos são verdadeiros? | 18% (RF) → 1 em 5.5 alertas é AVC real (muitos falsos alarmes) |
| **PR AUC** | Área sob curva Precisão-Recall | **Essencial para dados desbalanceados** (melhor que ROC em classes raras) | RF PR AUC ~0.35 indica dificuldade com minoria |
| **F1-score** | 2 × (Precisão × Recall) / (Precisão + Recall) | Balanço entre precisão e recall | 0.30 (RF) → modelo sacrifica precisão por recall (aceitável em diagnóstico) |

**Decisão de métrica:** Priorizamos **Recall** e **PR AUC** porque:
- Em diagnóstico clínico, **falsos negativos (perder AVC real) são piores que falsos positivos** (alert extra)
- Dados desbalanceados: Accuracy seria enganoso (~95% apenas prevendo "sem AVC")

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

### 4. Limitações e Considerações Clínicas

1. **Alvo é auto-relato:** Pacientes podem não se lembrar ou relatar AVC anterior → ruído
2. **Dados transversais:** Não captura evolução temporal; causa-efeito não é definida
3. **População específica:** NHANES é representativa de EUA; generalizabilidade a outras populações é limitada
4. **Modelo como ferramenta, não diagnóstico:** 
   - Recall 72% significa que **28% dos AVC não serão detectados**
   - Sempre exigir confirmação clínica e imagiologia (CT/MRI)
   - Precisão 18% gera múltiplos alertas falsos

5. **Desbalanceamento de classes:** 95% sem AVC, 5% com AVC
   - Risco: Modelo enviesado se não tratado (mitigado via `class_weight='balanced'` e métricas apropriadas)

### 5. Próximos Passos Recomendados

1. **Validação externa:** Testar modelo em cohort independente (validação temporal ou geográfica)
2. **Calibração:** Implementar `CalibratedClassifierCV` para probabilidades mais confiáveis
3. **Feature engineering avançado:** Criar interações (idade × pressão), polinômios, transformações
4. **Dados de imagem (CNN):** Se disponíveis ECGs, radiografias de tórax, ressonâncias
5. **Ensemble:** Combinar LR + RF + XGBoost para robustecer predições
6. **Ajuste de threshold:** Modificar ponto de corte de probabilidade (default=0.5) para maximizar Recall vs. Precisão conforme requisitos clínicos

---

## 🛠️ Detalhes Técnicos

### Versões de Software
- Python: 3.10+
- pandas: 2.3.3
- scikit-learn: 1.7.2
- numpy: 2.3.5 (fixado para compatibilidade com numba/shap)
- matplotlib, seaborn: Visualização
- missingno: Análise de valores ausentes

### Reprodutibilidade
- `random_state=42` fixado em train_test_split e modelos
- Ambiente isolado (venv)
- Dependências fixadas em `requirements.txt`

### Tempo de Execução
- **Primeira execução:** ~12–15 minutos (carregamento NHANES via internet)
- **Execuções subsequentes:** ~3–5 minutos (dados em cache)

---

## 📚 Referências

1. CDC NHANES — https://www.cdc.gov/nchs/nhanes
2. Documentação scikit-learn — https://scikit-learn.org
3. Pandascouple. Projeto ML Previsão de AVC — https://pandascouple.medium.com/projeto-machine-learning-previs%C3%A3o-de-avc-f4b7dce11929
4. USP Rádio. Uso de IA e análise de dados na prevenção de AVC — https://jornal.usp.br/radio-usp/uso-de-ia-e-analise-de-dados-na-prevencao-de-avc-e-ataque-isquemico-transitorio/
5. Nature Research. Machine learning para risco cardiovascular — https://www.nature.com/articles/s41598-024-61665-4
6. Journal of Health Informatics. AVC diagnosis — https://jhi.sbis.org.br/index.php/jhi-sbis/article/view/980
7. Nature Research 2025. AVC modeling — https://www.nature.com/articles/s41598-025-01855-w

---

## 📝 Autor e Contribuições

**Projeto:** Tech Challenge Fase 1 — FIAP Pós-Tech em IA para Desenvolvedores  
**Data:** Janeiro 2026  
**Status:** ✅ Completo (EDA + ML + Interpretação)

---

## ❓ Perguntas Frequentes

**P: Por que remover códigos 7 e 9?**  
R: Códigos 7 (Recusado) e 9 (Não sabe) representam respostas inválidas. Mantê-los como `NaN` é inadequado para variáveis categóricas; removê-los evita viés de imputação.

**P: Posso usar este modelo em produção?**  
R: Não diretamente. É uma **prova de conceito (PoC)**. Para produção:
- Validação em cohort independente
- Calibração de probabilidades
- Auditorias de equidade (bias por gênero, etnia)
- Aprovação regulatória (ex.: FDA para devices médicos)

**P: Como melhorar o Recall sem perder precisão?**  
R: Ajuste o threshold de decisão (ex., usar 0.3 em vez de 0.5) ou execute feature engineering, hyperparameter tuning com GridSearchCV.

**P: E se tiver dados de imagem?**  
R: Implemente CNN (ResNet, VGG) em TensorFlow/PyTorch para ECG, radiografias ou ressonâncias. Combine com este modelo tabular para ensemble.

---

**Última atualização:** 13 de janeiro de 2026
