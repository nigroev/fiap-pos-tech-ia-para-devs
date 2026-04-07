"""
train.py — Script de treinamento executado dentro do SageMaker Training Job
============================================================================
Projeto: AVC Stroke Prediction (NHANES)

Este script é o entry_point do SKLearn Estimator do SageMaker.
O SageMaker baixa os dados do S3 para /opt/ml/input/data/training/
e espera que o modelo treinado seja salvo em /opt/ml/model/.

Modos de execução (--mode):
  - hpo:  Treina um único RandomForest com hiperparâmetros fornecidos e emite
          métricas para o SageMaker Hyperparameter Tuning Job.
  - full: Pipeline completo — baseline models + GA com warm start opcional
          a partir dos melhores hiperparâmetros do HPO Tuning Job.

Integração SageMaker Experiments:
  - Se --experiment-name for fornecido, todas as métricas são logadas via
    SageMaker Experiments Run (log_metric, log_parameter).
"""

import argparse
import json
import logging
import os
import sys
import traceback
from copy import deepcopy
from datetime import datetime, timezone, timedelta

# Fuso horário de São Paulo (UTC-3)
BRT = timezone(timedelta(hours=-3))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    fbeta_score,
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# SageMaker Experiments (disponível no container SKLearn >= 1.2-1)
try:
    from sagemaker.experiments.run import load_run
    HAS_EXPERIMENTS = True
except ImportError:
    HAS_EXPERIMENTS = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.Formatter.converter = lambda *args: datetime.now(BRT).timetuple()
logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTES
# ==============================================================================

NUM_COLS = [
    "RIDAGEYR_age",
    "BPXSY1_sbp",
    "LBXGH_hba1c",
    "BMXBMI_bmi",
]
CAT_COLS = [
    "RIAGENDR_gender_bin",
    "DMDMARTL_married_bin",
    "BPQ020_high_bp_bin",
    "MCQ160B_chf_bin",
    "OCQ260_occupation",
    "SMQ020_smoking_bin",
]

# ==============================================================================
# PIPELINE DE PRÉ-PROCESSAMENTO
# ==============================================================================


def build_preprocessing_pipeline():
    """Cria o pipeline de pré-processamento com imputação e scaling."""
    num_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("num", num_pipe, NUM_COLS), ("cat", cat_pipe, CAT_COLS)]
    )


# ==============================================================================
# TREINAMENTO DOS MODELOS BASELINE
# ==============================================================================


def train_baseline_models(X_train, X_test, y_train, y_test, run=None):
    """Treina modelos baseline e retorna o melhor."""
    logger.info("Treinando modelos baseline...")

    preproc = build_preprocessing_pipeline()

    models = {
        "LogisticRegression": Pipeline(
            [
                ("pre", preproc),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("pre", preproc),
                (
                    "clf",
                    RandomForestClassifier(
                        class_weight="balanced", n_estimators=100, random_state=42, n_jobs=-1
                    ),
                ),
            ]
        ),
    }

    results = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        roc = roc_auc_score(y_test, y_prob)
        fbeta = fbeta_score(y_test, y_pred, beta=1.5)

        logger.info(f"\n===== {name} =====")
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        logger.info(f"ROC AUC: {roc:.4f}")
        logger.info(f"F-beta (β=1.5): {fbeta:.4f}")

        # SageMaker Experiments — logar métricas dos baselines
        if run:
            run.log_metric(f"baseline_{name}_roc_auc", roc)
            run.log_metric(f"baseline_{name}_fbeta_1_5", fbeta)

        results[name] = {"model": pipe, "roc_auc": roc, "fbeta": fbeta}

    best_name = max(results, key=lambda k: results[k]["fbeta"])
    logger.info(f"\nMelhor modelo baseline: {best_name} (F-beta={results[best_name]['fbeta']:.4f})")
    return best_name, results[best_name]["model"], results


# ==============================================================================
# OTIMIZAÇÃO COM ALGORITMO GENÉTICO
# ==============================================================================


def genetic_algorithm_optimization(X_train, y_train, n_pop=20, n_gen=10, warm_start_params=None, run=None):
    """Otimiza hiperparâmetros do RandomForest via Algoritmo Genético.

    Se warm_start_params for fornecido (lista de dicts vindos do HPO Tuning Job),
    os primeiros indivíduos da população são inicializados com esses valores,
    dando ao GA um ponto de partida mais informado (hot start).

    Se run for fornecido (SageMaker Experiments Run), loga fitness por geração.
    """
    logger.info("Iniciando otimização com Algoritmo Genético...")

    preproc = build_preprocessing_pipeline()

    hp_ranges = {
        "n_estimators": (50, 300),
        "max_depth": (3, 20),
        "min_samples_split": (2, 20),
        "min_samples_leaf": (1, 10),
    }

    def random_individual():
        return {
            "n_estimators": np.random.randint(*hp_ranges["n_estimators"]),
            "max_depth": np.random.randint(*hp_ranges["max_depth"]),
            "min_samples_split": np.random.randint(*hp_ranges["min_samples_split"]),
            "min_samples_leaf": np.random.randint(*hp_ranges["min_samples_leaf"]),
        }

    def fitness(individual):
        pipe = Pipeline(
            [
                ("pre", deepcopy(preproc)),
                (
                    "clf",
                    RandomForestClassifier(
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                        **individual,
                    ),
                ),
            ]
        )
        scorer = make_scorer(fbeta_score, beta=1.5)
        scores = cross_val_score(pipe, X_train, y_train, cv=3, scoring=scorer, n_jobs=-1)
        return scores.mean()

    def tournament_selection(population, fitnesses, k=5):
        k = min(k, len(population))
        idxs = np.random.choice(len(population), k, replace=False)
        best_idx = idxs[np.argmax([fitnesses[i] for i in idxs])]
        return deepcopy(population[best_idx])

    def crossover(p1, p2):
        child = {}
        for key in p1:
            child[key] = p1[key] if np.random.random() < 0.5 else p2[key]
        return child

    def mutate(individual, mutation_rate=0.1):
        ind = deepcopy(individual)
        for key in ind:
            if np.random.random() < mutation_rate:
                low, high = hp_ranges[key]
                ind[key] = np.random.randint(low, high)
        return ind

    population = [random_individual() for _ in range(n_pop)]

    # Warm start: substituir primeiros indivíduos pelos melhores do HPO
    if warm_start_params:
        for i, params in enumerate(warm_start_params[:n_pop]):
            population[i] = {
                "n_estimators": int(params["n_estimators"]),
                "max_depth": int(params["max_depth"]),
                "min_samples_split": int(params["min_samples_split"]),
                "min_samples_leaf": int(params["min_samples_leaf"]),
            }
        logger.info(f"GA warm start: {min(len(warm_start_params), n_pop)} indivíduos do HPO Tuning Job")

    best_overall = None
    best_fitness = -1

    for gen in range(n_gen):
        fitnesses = [fitness(ind) for ind in population]

        gen_best_idx = np.argmax(fitnesses)
        gen_best_fit = fitnesses[gen_best_idx]
        logger.info(f"  Geração {gen+1}/{n_gen} — Melhor fitness: {gen_best_fit:.4f}")

        # SageMaker Experiments — logar fitness por geração
        if run:
            run.log_metric("ga_best_fitness", gen_best_fit, step=gen + 1)

        if gen_best_fit > best_fitness:
            best_fitness = gen_best_fit
            best_overall = deepcopy(population[gen_best_idx])

        sorted_idxs = np.argsort(fitnesses)[::-1]
        new_pop = [deepcopy(population[sorted_idxs[0]]), deepcopy(population[sorted_idxs[1]])]

        while len(new_pop) < n_pop:
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)
            child = crossover(p1, p2)
            child = mutate(child, mutation_rate=0.1)
            new_pop.append(child)

        population = new_pop

    logger.info(f"GA concluído. Melhor fitness: {best_fitness:.4f}")
    logger.info(f"Melhores hiperparâmetros: {best_overall}")

    best_pipe = Pipeline(
        [
            ("pre", build_preprocessing_pipeline()),
            (
                "clf",
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                    **best_overall,
                ),
            ),
        ]
    )
    best_pipe.fit(X_train, y_train)
    return best_pipe, best_overall, best_fitness


# ==============================================================================
# MAIN — Executado pelo SageMaker Training Job
# ==============================================================================


def _main():
    parser = argparse.ArgumentParser()

    # Modo de execução
    parser.add_argument("--mode", type=str, default="full", choices=["full", "hpo"])

    # Hiperparâmetros do GA (modo full)
    parser.add_argument("--ga-pop", type=int, default=10)
    parser.add_argument("--ga-gen", type=int, default=5)

    # Hiperparâmetros do RF (modo hpo — variados pelo SageMaker HPO Tuner)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-samples-split", type=int, default=5)
    parser.add_argument("--min-samples-leaf", type=int, default=2)

    # Warm start do GA com resultados do HPO (JSON string)
    parser.add_argument("--warm-start-params", type=str, default=None)

    # Diretórios padrão do SageMaker
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/training")))
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"  SageMaker Training Job — AVC Stroke Prediction (mode={args.mode})")
    logger.info("=" * 60)

    # Carregar Run do SageMaker Experiments (se configurado pelo orquestrador)
    experiment_run = None
    if HAS_EXPERIMENTS:
        try:
            experiment_run = load_run().__enter__()
            logger.info(f"SageMaker Experiments Run carregado: {experiment_run.run_name}")
        except Exception as e:
            logger.info(f"SageMaker Experiments não configurado para este job: {e}")
            experiment_run = None

    # 1. Ler dados pré-processados do canal de treinamento
    train_dir = args.train
    logger.info(f"Procurando arquivos CSV em: {train_dir}")
    
    # Debug: Listar variáveis de ambiente SM_CHANNEL
    logger.info("Canais de entrada disponíveis:")
    for env_key in os.environ:
        if env_key.startswith("SM_CHANNEL_"):
            logger.info(f"  {env_key}: {os.environ[env_key]}")

    # Fallback para qualquer canal que comece com SM_CHANNEL_ caso o padrão não exista ou esteja vazio
    if not os.path.exists(train_dir) or not [f for f in os.listdir(train_dir) if f.endswith(".csv")]:
        logger.info(f"Diretório '{train_dir}' não existe ou não contém CSVs. Buscando em outros canais...")
        channels = [v for k, v in os.environ.items() if k.startswith("SM_CHANNEL_")]
        found_valid_channel = False
        for channel in channels:
            if os.path.exists(channel) and [f for f in os.listdir(channel) if f.endswith(".csv")]:
                train_dir = channel
                logger.info(f"Canal válido encontrado em: {train_dir}")
                found_valid_channel = True
                break
        
        if not found_valid_channel:
            logger.warning("Nenhum canal SM_CHANNEL_ contém arquivos CSV.")

    if os.path.exists(train_dir):
        csv_files = [f for f in os.listdir(train_dir) if f.endswith(".csv")]
    else:
        csv_files = []

    if not csv_files:
        # Debug radical: listar tudo recursivamente se possível ou apenas a raiz do input
        logger.info("Listagem geral de /opt/ml/input/data/ :")
        try:
            for root, dirs, files in os.walk("/opt/ml/input/data/"):
                logger.info(f"  {root} -> {files}")
        except Exception:
            pass
        raise FileNotFoundError(f"Nenhum arquivo CSV encontrado em {train_dir}")

    df = pd.read_csv(os.path.join(train_dir, csv_files[0]))
    logger.info(f"Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")

    # 2. Preparar features e target
    df_model = df.dropna(subset=["MCQ160F_stroke_bin"]).copy()
    X = df_model[NUM_COLS + CAT_COLS]
    y = df_model["MCQ160F_stroke_bin"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    logger.info(f"Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")

    # ======================================================================
    # MODO HPO — Treinamento único para SageMaker Hyperparameter Tuning Job
    # ======================================================================
    if args.mode == "hpo":
        rf_params = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_split": args.min_samples_split,
            "min_samples_leaf": args.min_samples_leaf,
        }
        logger.info(f"Modo HPO — Treinando RF com: {rf_params}")

        preproc = build_preprocessing_pipeline()
        pipe = Pipeline([
            ("pre", preproc),
            ("clf", RandomForestClassifier(
                class_weight="balanced", random_state=42, n_jobs=-1, **rf_params
            )),
        ])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_prob)
        fbeta = fbeta_score(y_test, y_pred, beta=1.5)

        # Emitir métricas em formato parseável pelo SageMaker HPO Tuner
        logger.info(f"fbeta_1_5={fbeta:.6f};")
        logger.info(f"roc_auc={roc:.6f};")

        # SageMaker Experiments — logar métricas do HPO
        if experiment_run:
            for k, v in rf_params.items():
                experiment_run.log_parameter(k, v)
            experiment_run.log_metric("fbeta_1_5", fbeta)
            experiment_run.log_metric("roc_auc", roc)

        # Salvar modelo
        model_path = os.path.join(args.model_dir, "model.joblib")
        joblib.dump(pipe, model_path)

        metrics = {
            "model_name": "RandomForest_HPO",
            "roc_auc": float(roc),
            "fbeta_1_5": float(fbeta),
            "hyperparameters": rf_params,
        }
        with open(os.path.join(args.model_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Modo HPO concluído — F-beta: {fbeta:.4f}, ROC AUC: {roc:.4f}")

    # ======================================================================
    # MODO FULL — Pipeline completo (baseline + GA com warm start do HPO)
    # ======================================================================
    elif args.mode == "full":
        # 3. Treinar modelos baseline
        best_name, best_baseline, all_results = train_baseline_models(
            X_train, X_test, y_train, y_test, run=experiment_run
        )

        # 4. Parsear warm start do HPO (se fornecido)
        warm_start = None
        if args.warm_start_params:
            warm_start = json.loads(args.warm_start_params)
            logger.info(f"Warm start do HPO recebido: {len(warm_start)} configurações")

        # 5. Otimizar com Algoritmo Genético (com warm start)
        ga_model, ga_params, ga_fitness = genetic_algorithm_optimization(
            X_train, y_train,
            n_pop=args.ga_pop,
            n_gen=args.ga_gen,
            warm_start_params=warm_start,
            run=experiment_run,
        )

        # Avaliar modelo GA no teste
        y_pred_ga = ga_model.predict(X_test)
        y_prob_ga = ga_model.predict_proba(X_test)[:, 1]
        ga_roc = roc_auc_score(y_test, y_prob_ga)
        ga_fbeta = fbeta_score(y_test, y_pred_ga, beta=1.5)
        logger.info(f"\nModelo GA — ROC AUC: {ga_roc:.4f}, F-beta: {ga_fbeta:.4f}")
        logger.info(f"\n{classification_report(y_test, y_pred_ga)}")

        # 6. Selecionar o melhor modelo (baseline vs GA)
        if ga_fbeta > all_results[best_name]["fbeta"]:
            final_model = ga_model
            final_name = "RandomForest_GA_Optimized"
            final_roc = ga_roc
            final_fbeta = ga_fbeta
            logger.info(f"Modelo selecionado: GA Otimizado (F-beta={ga_fbeta:.4f})")
        else:
            final_model = best_baseline
            final_name = best_name
            final_roc = all_results[best_name]["roc_auc"]
            final_fbeta = all_results[best_name]["fbeta"]
            logger.info(f"Modelo selecionado: Baseline {best_name} (F-beta={final_fbeta:.4f})")

        # Emitir métricas em formato parseável
        logger.info(f"fbeta_1_5={final_fbeta:.6f};")
        logger.info(f"roc_auc={final_roc:.6f};")

        # SageMaker Experiments — logar resultado final
        if experiment_run:
            experiment_run.log_parameter("final_model_name", final_name)
            experiment_run.log_parameter("warm_start_from_hpo", str(warm_start is not None))
            experiment_run.log_parameter("ga_pop", args.ga_pop)
            experiment_run.log_parameter("ga_gen", args.ga_gen)
            experiment_run.log_metric("final_fbeta_1_5", final_fbeta)
            experiment_run.log_metric("final_roc_auc", final_roc)
            experiment_run.log_metric("ga_fbeta_1_5", ga_fbeta)
            experiment_run.log_metric("ga_roc_auc", ga_roc)
            if ga_params:
                for k, v in ga_params.items():
                    experiment_run.log_parameter(f"ga_{k}", v)

        # 7. Salvar modelo no diretório de modelo do SageMaker
        model_path = os.path.join(args.model_dir, "model.joblib")
        joblib.dump(final_model, model_path)
        logger.info(f"Modelo salvo em: {model_path}")

        # 8. Salvar métricas
        metrics = {
            "model_name": final_name,
            "roc_auc": float(final_roc),
            "fbeta_1_5": float(final_fbeta),
            "ga_params": ga_params if final_name == "RandomForest_GA_Optimized" else None,
            "warm_start_from_hpo": warm_start is not None,
        }
        metrics_path = os.path.join(args.model_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"Métricas salvas em: {metrics_path}")

    # Finalizar SageMaker Experiments Run
    if experiment_run:
        try:
            experiment_run.__exit__(None, None, None)
        except Exception:
            pass

    logger.info("=" * 60)
    logger.info("  Training Job concluído com sucesso!")
    logger.info("=" * 60)


# ==============================================================================
# Entry point — precisa estar APÓS a definição de _main()
# ==============================================================================

if __name__ == "__main__":
    try:
        _main()
    except Exception:
        # Imprime traceback completo para stdout (capturado na FailureReason do SageMaker)
        tb = traceback.format_exc()
        print("=" * 60, flush=True)
        print("ERRO NO TRAINING JOB — traceback completo:", flush=True)
        print(tb, flush=True)
        print("=" * 60, flush=True)
        # Escrever resumo curto primeiro no stderr para que apareça no FailureReason
        # (SageMaker trunca FailureReason; o framework traceback consome quase tudo)
        short_err = tb.strip().split("\n")[-1]
        sys.stderr.write(f"TRAIN_ERROR: {short_err}\n")
        # Salvar traceback completo em /opt/ml/output/ para recuperação via S3
        try:
            out_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "failure_traceback.txt"), "w") as f:
                f.write(tb)
        except Exception:
            pass
        sys.exit(1)
