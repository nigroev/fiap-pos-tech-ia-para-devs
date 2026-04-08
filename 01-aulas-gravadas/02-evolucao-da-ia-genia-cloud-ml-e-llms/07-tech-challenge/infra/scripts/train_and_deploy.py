"""
train_and_deploy.py — Orquestrador de treinamento e deploy via SageMaker
=========================================================================
Projeto: AVC Stroke Prediction (NHANES)

Este script executa:
  1. Download e merge dos dados NHANES (mesma lógica do notebook)
  2. Pré-processamento e feature engineering
  3. SageMaker Feature Store — persistência das features offline
  4. SageMaker Autopilot — AutoML (feature engineering, tuning, training, ranking)
  5. SageMaker Hyperparameter Tuning Job — otimiza hiperparâmetros do RF
  6. Training Job com GA usando warm start dos melhores hiperparâmetros do HPO
  7. Comparação dos resultados e deploy do melhor modelo

Integrações SageMaker:
  - Experiments: Rastreamento de todos os experimentos com Run/Experiment
  - Feature Store: Feature Group offline para features pré-processadas
  - Pipelines: Orquestração via SageMaker Pipelines (opcional com --use-pipeline)

Módulos (pacote pipeline/):
  - config.py          — Logging, BRT timezone, diretório de treino
  - data_ingestion.py  — Coleta NHANES, pré-processamento, upload S3
  - feature_store.py   — Feature Group offline (criação, ingestão, leitura)
  - autopilot.py       — AutoML (lançamento assíncrono, polling)
  - training.py        — HPO Tuning Job, GA Training Job
  - sagemaker_pipeline.py — SageMaker Pipeline (TuningStep + TrainingStep)
  - deployment.py      — Deploy e gestão de endpoints
  - metrics.py         — Experiments e métricas consolidadas
"""

import argparse
import math
import time
from datetime import datetime

import boto3
import sagemaker
from sagemaker.experiments import Run
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.pipeline_context import PipelineSession

from pipeline import (
    BRT,
    logger,
    load_nhanes_data,
    preprocess_data,
    upload_dataset_to_s3,
    create_or_get_feature_group,
    ingest_features,
    create_experiment,
    run_autopilot_job,
    wait_for_autopilot,
    run_hpo_tuning_job,
    run_training_job,
    create_sagemaker_pipeline,
    deploy_sagemaker_endpoint,
    deploy_autopilot_endpoint,
)
from pipeline.metrics import save_metrics


# ==============================================================================
# MAIN — Orquestra o pipeline completo
# ==============================================================================

_TOTAL_PHASES = 8
_phase_start_time: dict = {}


def _phase(n: int, label: str, end: bool = False) -> None:
    """Loga um banner de início ou fim de fase com timestamp e contador."""
    bar = "─" * 56
    ts = datetime.now(BRT).strftime("%H:%M:%S")
    if end:
        elapsed = ""
        if n in _phase_start_time:
            secs = time.time() - _phase_start_time[n]
            elapsed = f"  ({math.floor(secs // 60)}m{int(secs % 60)}s)"
        logger.info(f"╰{bar}╯")
        logger.info(f"  ✓ FASE {n}/{_TOTAL_PHASES} concluída{elapsed}  [{ts}]")
        logger.info("")
    else:
        _phase_start_time[n] = time.time()
        logger.info("")
        logger.info(f"╭{bar}╮")
        logger.info(f"  ▶ FASE {n}/{_TOTAL_PHASES}: {label}  [{ts}]")
        logger.info(f"╰{bar}╯")


def main():
    parser = argparse.ArgumentParser(
        description="Orquestrador de treinamento e deploy AVC/Stroke via SageMaker"
    )
    parser.add_argument("--bucket", required=True, help="Nome do bucket S3")
    parser.add_argument("--region", required=True, help="Região AWS")
    parser.add_argument("--project", required=True, help="Nome do projeto")
    parser.add_argument("--skip-deploy", action="store_true", help="Pular deploy dos endpoints")
    parser.add_argument("--skip-autopilot", action="store_true", help="Pular Autopilot AutoML")
    parser.add_argument("--skip-feature-store", action="store_true", help="Pular Feature Store")
    parser.add_argument(
        "--model-data", type=str, default=None,
        help="URI S3 do artefato model.tar.gz. Quando fornecido, pula fases 2-7 e executa apenas o deploy.",
    )
    parser.add_argument(
        "--use-pipeline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Usar SageMaker Pipelines (padrão: ativado). Use --no-use-pipeline para modo manual.",
    )
    parser.add_argument("--ga-pop", type=int, default=10, help="Tamanho da população do GA")
    parser.add_argument("--ga-gen", type=int, default=5, help="Número de gerações do GA")
    parser.add_argument(
        "--training-instance-type", type=str, default="ml.m5.large",
        help="Tipo de instância para Training Jobs"
    )
    parser.add_argument(
        "--endpoint-instance-type", type=str, default="ml.t2.medium",
        help="Tipo de instância para Endpoints de inferência"
    )
    parser.add_argument(
        "--max-run", type=int, default=1800,
        help="Tempo máximo de execução por Training Job (segundos)"
    )
    parser.add_argument(
        "--max-spot-wait", type=int, default=3600,
        help="Tempo máximo de espera por instância spot (segundos, >= max-run)"
    )
    parser.add_argument(
        "--hpo-max-jobs", type=int, default=6,
        help="Número máximo de jobs no HPO Tuning"
    )
    parser.add_argument(
        "--hpo-max-parallel-jobs", type=int, default=3,
        help="Número máximo de jobs paralelos no HPO Tuning"
    )
    parser.add_argument(
        "--autopilot-max-candidates", type=int, default=5,
        help="Número máximo de modelos candidatos do Autopilot"
    )
    parser.add_argument(
        "--autopilot-timeout", type=int, default=45,
        help="Timeout em minutos para aguardar Autopilot (0=sem limite)"
    )
    args = parser.parse_args()

    bar = "═" * 58
    logger.info(f"╔{bar}╗")
    logger.info("║  Orquestrador — AVC Stroke Prediction (SageMaker)       ║")
    logger.info("║  Experiments · Feature Store · Pipelines                ║")
    logger.info("║  Autopilot · HPO · GA                                   ║")
    logger.info(f"╚{bar}╝")
    logger.info(f"  Início: {datetime.now(BRT).strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Bucket: {args.bucket} | Região: {args.region} | Projeto: {args.project}")
    logger.info(f"  Modo: {'Pipeline' if args.use_pipeline else 'Manual'} | "
                f"Autopilot: {'sim' if not args.skip_autopilot else 'não'} | "
                f"FeatureStore: {'sim' if not args.skip_feature_store else 'não'}")
    logger.info(
        f"  HPO jobs={args.hpo_max_jobs} parallel={args.hpo_max_parallel_jobs} | "
        f"GA pop={args.ga_pop} gen={args.ga_gen} | "
        f"max-run={args.max_run}s | "
        f"autopilot-timeout={args.autopilot_timeout}min"
    )

    # ================================================================
    # 1. Setup — Role, Session, Experiment
    # ================================================================
    _phase(1, "Setup — Role, Session, SageMaker Experiment")
    iam = boto3.client("iam", region_name=args.region)
    role_name = f"{args.project}-dev-sagemaker-role"
    role_arn = iam.get_role(RoleName=role_name)["Role"]["Arn"]
    logger.info(f"Usando role: {role_arn}")

    session = sagemaker.Session(
        boto_session=boto3.Session(region_name=args.region),
        default_bucket=args.bucket,
    )
    pipeline_session = PipelineSession(
        boto_session=boto3.Session(region_name=args.region),
        default_bucket=args.bucket,
    )

    # Criar Experiment para rastrear todos os jobs
    experiment_name = create_experiment(session, args.project)
    _phase(1, "", end=True)

    # ================================================================
    # Deploy-only: pula fases 2-7 quando --model-data é fornecido
    # ================================================================
    if args.model_data:
        logger.info(f"Modo deploy-only ativado. Artefato: {args.model_data}")
        if not args.skip_deploy:
            # Criar estimator mínimo apenas para carregar a sessão e role
            stub_estimator = SKLearn(
                entry_point="train.py",
                role=role_arn,
                instance_type=args.training_instance_type,
                framework_version="1.2-1",
                py_version="py3",
                sagemaker_session=session,
            )
            # Sobrescrever model_data com o URI fornecido
            stub_estimator.model_data = args.model_data
            _phase(8, "Deploy dos endpoints de inferência")
            endpoint_name = deploy_sagemaker_endpoint(
                stub_estimator, args.project, args.endpoint_instance_type, role_arn=role_arn
            )
            logger.info(f"Endpoint GA ativo: {endpoint_name}")
            _phase(8, "", end=True)
        else:
            logger.info("Deploy ignorado (--skip-deploy).")
        return

    # ================================================================
    # 2. Carregar e pré-processar dados
    # ================================================================
    _phase(2, "Ingestão e pré-processamento de dados (NHANES)")
    df_raw = load_nhanes_data(bucket=args.bucket, region=args.region)
    df = preprocess_data(df_raw)

    # ================================================================
    # 3. Feature Store — persistir features pré-processadas
    # ================================================================
    _phase(3, "Feature Store — ingestão de features pré-processadas")
    if not args.skip_feature_store:
        fg, fg_name = create_or_get_feature_group(
            session=session,
            role_arn=role_arn,
            bucket=args.bucket,
            project=args.project,
        )
        ingest_features(df, fg, session)
        logger.info(f"Features ingeridas no Feature Store: {fg_name}")
    else:
        logger.info("Feature Store ignorado (--skip-feature-store).")
    _phase(3, "", end=True)

    # Upload CSV para S3 (usado como canal de treinamento)
    data_s3_uri = upload_dataset_to_s3(df, args.bucket, args.region)

    # Log de dados no Experiment
    _phase(2, "", end=True)
    try:
        with Run(
            experiment_name=experiment_name,
            run_name=f"data-prep-{datetime.now(BRT).strftime('%Y%m%d%H%M%S')}",
            sagemaker_session=session,
        ) as run:
            run.log_parameter("dataset_rows", df.shape[0])
            run.log_parameter("dataset_cols", df.shape[1])
            run.log_parameter("positive_rate", float(df["MCQ160F_stroke_bin"].mean()))
            run.log_parameter("feature_store_enabled", str(not args.skip_feature_store))
    except Exception as e:
        logger.warning(f"Experiment logging falhou (não-fatal): {e}")

    # ================================================================
    # 4. Lançar Autopilot AutoML (assíncrono — roda em paralelo)
    # ================================================================
    _phase(4, "Autopilot AutoML — lançamento assíncrono")
    auto_ml = None
    autopilot_job = None
    if not args.skip_autopilot:
        ap_timeout = args.autopilot_timeout
        result = run_autopilot_job(
            data_s3_uri=data_s3_uri,
            bucket=args.bucket,
            region=args.region,
            project=args.project,
            role_arn=role_arn,
            session=session,
            max_candidates=args.autopilot_max_candidates,
            max_total_runtime=ap_timeout * 60 if ap_timeout > 0 else 2700,
        )
        if result is not None and result != (None, None):
            auto_ml, autopilot_job = result
        else:
            logger.warning("Autopilot não pôde ser lançado; continuando sem ele.")
    else:
        logger.info("Autopilot ignorado (--skip-autopilot).")
    _phase(4, "", end=True)

    # ================================================================
    # 5. Execução — Pipeline ou manual
    # ================================================================
    _phase(5, f"Treinamento — {'SageMaker Pipelines (HPO + GA)' if args.use_pipeline else 'Modo manual (HPO + GA)'}")
    if args.use_pipeline:
        # ----- Modo Pipeline -----
        logger.info("Executando via SageMaker Pipelines...")

        pipeline, ga_estimator = create_sagemaker_pipeline(
            bucket=args.bucket,
            region=args.region,
            project=args.project,
            role_arn=role_arn,
            session=pipeline_session,
            training_instance_type=args.training_instance_type,
            endpoint_instance_type=args.endpoint_instance_type,
            ga_pop=args.ga_pop,
            ga_gen=args.ga_gen,
            max_run=args.max_run,
            max_spot_wait=args.max_spot_wait,
            hpo_max_jobs=args.hpo_max_jobs,
            hpo_max_parallel_jobs=args.hpo_max_parallel_jobs,
            data_s3_uri=data_s3_uri,
            experiment_name=experiment_name,
        )

        # Criar/atualizar e executar o Pipeline
        pipeline.upsert(role_arn=role_arn)
        execution = pipeline.start(
            execution_display_name=f"exec-{datetime.now(BRT).strftime('%Y%m%d%H%M%S')}",
            # Passando parâmetros explicitamente garante que os valores
            # independente da definição em cache no SageMaker
            parameters={
                "TrainingInstanceType": args.training_instance_type,
                "GAPop": args.ga_pop,
                "GAGen": args.ga_gen,
                "HPOMaxJobs": args.hpo_max_jobs,
                "HPOMaxParallelJobs": args.hpo_max_parallel_jobs,
            },
        )
        logger.info(f"Pipeline execution ARN: {execution.arn}")
        logger.info(
            f"Parâmetros: HPO max_jobs={args.hpo_max_jobs} parallel={args.hpo_max_parallel_jobs} | "
            f"GA pop={args.ga_pop} gen={args.ga_gen} | "
            f"instance={args.training_instance_type}"
        )
        logger.info("Acompanhando execução do Pipeline... (poll a cada 60s)")

        sm_client = session.sagemaker_client
        poll_interval = 60
        last_step_statuses = {}

        while True:
            exec_desc = execution.describe()
            exec_status = exec_desc["PipelineExecutionStatus"]

            # Listar steps e logar mudanças de status
            steps = execution.list_steps()
            total_steps = len(steps)
            done_steps = sum(1 for s in steps if s["StepStatus"] in ("Succeeded", "Failed", "Stopped"))

            _STEP_DESC = {
                "TuningStep": "HPO Bayesian (otimização de hiperparâmetros)",
                "TrainingStep": "GA Training (treinamento com warm start)",
            }

            for idx, step in enumerate(steps, 1):
                step_name = step["StepName"]
                step_status = step["StepStatus"]
                step_type = step.get("StepType", "")
                desc = _STEP_DESC.get(step_type, step_type)
                prev = last_step_statuses.get(step_name)
                if step_status != prev:
                    logger.info(
                        f"  [Step {idx}/{total_steps}] {step_name} ({desc}): "
                        f"{prev or 'Pendente'} → {step_status}"
                    )
                    last_step_statuses[step_name] = step_status

                # Detalhar progresso do HPO Tuning
                if step_status == "Executing" and step.get("Metadata", {}).get("TuningJob"):
                    tuning_job_name = step["Metadata"]["TuningJob"]["Arn"].split("/")[-1]
                    try:
                        tj = sm_client.describe_hyper_parameter_tuning_job(
                            HyperParameterTuningJobName=tuning_job_name
                        )
                        counts = tj.get("TrainingJobStatusCounters", {})
                        best = tj.get("BestTrainingJob", {})
                        best_metric = best.get("FinalHyperParameterTuningJobObjectiveMetric", {})
                        logger.info(
                            f"    HPO '{tuning_job_name}': "
                            f"Completed={counts.get('Completed', 0)}, "
                            f"InProgress={counts.get('InProgress', 0)}, "
                            f"Failed={counts.get('Failed', 0)} | "
                            f"BestMetric={best_metric.get('MetricName', '-')}="
                            f"{best_metric.get('Value', '-')}"
                        )
                    except Exception:
                        pass

                # Detalhar progresso do Training Job
                if step_status == "Executing" and step.get("Metadata", {}).get("TrainingJob"):
                    training_job_arn = step["Metadata"]["TrainingJob"]["Arn"]
                    training_job_name = training_job_arn.split("/")[-1]
                    try:
                        tj = sm_client.describe_training_job(TrainingJobName=training_job_name)
                        tj_status = tj["TrainingJobStatus"]
                        secondary = tj.get("SecondaryStatus", "")
                        elapsed = ""
                        if "TrainingStartTime" in tj:
                            secs = (datetime.now(tj["TrainingStartTime"].tzinfo) - tj["TrainingStartTime"]).total_seconds()
                            elapsed = f" | Elapsed={math.floor(secs//60)}m{int(secs%60)}s"
                        logger.info(
                            f"    Training '{training_job_name}': {tj_status} ({secondary}){elapsed}"
                        )
                    except Exception:
                        pass

            if exec_status in ("Succeeded", "Failed", "Stopped"):
                logger.info(f"Pipeline finalizado com status: {exec_status}")
                break

            logger.info(f"  Pipeline: {exec_status} | steps {done_steps}/{total_steps} concluídos — aguardando {poll_interval}s...")
            time.sleep(poll_interval)

        # Para deploy, recuperar o training job do Pipeline e recriar estimator com session normal
        estimator = ga_estimator
        if exec_status == "Succeeded":
            try:
                # Buscar o TrainingStep do Pipeline para obter o job name real
                ga_step = next(
                    (s for s in execution.list_steps()
                     if s.get("Metadata", {}).get("TrainingJob")),
                    None
                )
                if ga_step:
                    training_job_name = ga_step["Metadata"]["TrainingJob"]["Arn"].split("/")[-1]
                    estimator = SKLearn.attach(training_job_name, sagemaker_session=session)
                    logger.info(f"Estimator recuperado do Pipeline: {training_job_name}")
            except Exception as e:
                logger.warning(f"SKLearn.attach falhou: {e}. Tentando fallback via describe_training_job...")
                try:
                    ga_step = next(
                        (s for s in execution.list_steps()
                         if s.get("Metadata", {}).get("TrainingJob")),
                        None
                    )
                    if ga_step:
                        training_job_name = ga_step["Metadata"]["TrainingJob"]["Arn"].split("/")[-1]
                        tj_desc = sm_client.describe_training_job(TrainingJobName=training_job_name)
                        real_model_data = tj_desc["ModelArtifacts"]["S3ModelArtifacts"]
                        ga_estimator._current_job_name = training_job_name
                        ga_estimator.model_data = real_model_data
                        estimator = ga_estimator
                        logger.info(f"Fallback: model_data={real_model_data} (job={training_job_name})")
                except Exception as e2:
                    logger.warning(f"Fallback também falhou: {e2}")
        elif exec_status in ("Failed", "Stopped"):
            # Tentar recuperar o training job mesmo em caso de falha, para diagnóstico
            try:
                ga_step = next(
                    (s for s in execution.list_steps()
                     if s.get("Metadata", {}).get("TrainingJob")),
                    None
                )
                if ga_step:
                    training_job_name = ga_step["Metadata"]["TrainingJob"]["Arn"].split("/")[-1]
                    tj = session.sagemaker_client.describe_training_job(TrainingJobName=training_job_name)
                    failure_reason = tj.get("FailureReason", "N/A")
                    logger.error(f"Training job falhou: {training_job_name}")
                    logger.error(f"  Motivo: {failure_reason}")
            except Exception:
                pass
            # Diagnosticar falha no HPO step
            try:
                hpo_step = next(
                    (s for s in execution.list_steps()
                     if s.get("Metadata", {}).get("TuningJob")),
                    None
                )
                if hpo_step:
                    tuning_job_name = hpo_step["Metadata"]["TuningJob"]["Arn"].split("/")[-1]
                    tj = session.sagemaker_client.describe_hyper_parameter_tuning_job(
                        HyperParameterTuningJobName=tuning_job_name
                    )
                    failure_reason = tj.get("FailureReason", "N/A")
                    counts = tj.get("TrainingJobStatusCounters", {})
                    logger.error(f"HPO Tuning job: {tuning_job_name}")
                    logger.error(f"  Status: {tj.get('HyperParameterTuningJobStatus')}")
                    logger.error(f"  Jobs: Completed={counts.get('Completed',0)}, "
                                 f"Failed={counts.get('Failed',0)}, "
                                 f"Stopped={counts.get('Stopped',0)}")
                    logger.error(f"  Motivo: {failure_reason}")
                    # Mostrar motivo do primeiro job com falha
                    failed_jobs = session.sagemaker_client.list_training_jobs_for_hyper_parameter_tuning_job(
                        HyperParameterTuningJobName=tuning_job_name,
                        StatusEquals="Failed",
                        MaxResults=1,
                    ).get("TrainingJobSummaries", [])
                    if failed_jobs:
                        failed_name = failed_jobs[0]["TrainingJobName"]
                        failed_tj = session.sagemaker_client.describe_training_job(TrainingJobName=failed_name)
                        logger.error(f"  Primeiro job falho: {failed_name}")
                        logger.error(f"    Motivo: {failed_tj.get('FailureReason', 'N/A')}")
            except Exception as diag_exc:
                logger.warning(f"Não foi possível diagnosticar falha do HPO: {diag_exc}")
            logger.error(
                f"Pipeline terminou com status '{exec_status}'. "
                "O modelo GA não foi gerado — deploy será ignorado."
            )
            estimator = None
        tuner = None
        warm_start_params = None
        _phase(5, "", end=True)

    else:
        # ----- Modo manual (sem Pipeline) -----
        logger.info("Executando em modo manual (sem SageMaker Pipelines)...")

        # HPO Tuning Job
        tuner, warm_start_params = run_hpo_tuning_job(
            data_s3_uri=data_s3_uri,
            bucket=args.bucket,
            region=args.region,
            project=args.project,
            role_arn=role_arn,
            session=session,
            training_instance_type=args.training_instance_type,
            max_run=args.max_run,
            max_spot_wait=args.max_spot_wait,
            hpo_max_jobs=args.hpo_max_jobs,
            hpo_max_parallel_jobs=args.hpo_max_parallel_jobs,
            experiment_name=experiment_name,
        )

        # Logar HPO results no Experiment
        with Run(
            experiment_name=experiment_name,
            run_name=f"hpo-results-{datetime.now(BRT).strftime('%Y%m%d%H%M%S')}",
            sagemaker_session=session,
        ) as run:
            run.log_parameter("hpo_tuning_job", tuner.latest_tuning_job.name)
            run.log_parameter("hpo_max_jobs", args.hpo_max_jobs)
            for i, params in enumerate(warm_start_params):
                for k, v in params.items():
                    run.log_parameter(f"hpo_top{i+1}_{k}", v)

        # GA Training Job com warm start
        estimator = run_training_job(
            data_s3_uri=data_s3_uri,
            bucket=args.bucket,
            region=args.region,
            project=args.project,
            role_arn=role_arn,
            session=session,
            training_instance_type=args.training_instance_type,
            ga_pop=args.ga_pop,
            ga_gen=args.ga_gen,
            max_run=args.max_run,
            max_spot_wait=args.max_spot_wait,
            warm_start_params=warm_start_params,
            experiment_name=experiment_name,
        )
        _phase(5, "", end=True)

    # ================================================================
    # 6. Aguardar Autopilot e comparar resultados
    # ================================================================
    _phase(6, "Aguardar Autopilot e comparar resultados")
    autopilot_results = None
    if autopilot_job:
        autopilot_results = wait_for_autopilot(
            None, autopilot_job, session,
            timeout_minutes=getattr(args, 'autopilot_timeout', 0)
        )

    # Logar comparação no Experiment
    with Run(
        experiment_name=experiment_name,
        run_name=f"comparison-{datetime.now(BRT).strftime('%Y%m%d%H%M%S')}",
        sagemaker_session=session,
    ) as run:
        if estimator is not None:
            run.log_parameter("ga_training_job", estimator.latest_training_job.name)
            run.log_parameter("ga_model_s3_uri", str(estimator.model_data))
        run.log_parameter("pipeline_mode", str(args.use_pipeline))
        if autopilot_results:
            run.log_parameter("autopilot_candidate", autopilot_results["candidate_name"])
            run.log_metric("autopilot_f1", autopilot_results["metric_value"])
        if tuner:
            run.log_parameter("hpo_tuning_job", tuner.latest_tuning_job.name)

    _phase(6, "", end=True)
    logger.info("  ┌─ COMPARAÇÃO DE RESULTADOS ─────────────────────────────┐")
    if estimator is not None:
        logger.info(f"  │  GA Job  : {estimator.latest_training_job.name}")
        logger.info(f"  │  GA Model: {estimator.model_data}")
    else:
        logger.info("  │  GA Job  : FALHOU — sem artefato de modelo")
    if autopilot_results:
        logger.info(f"  │  Autopilot candidato : {autopilot_results['candidate_name']}")
        logger.info(f"  │  Autopilot {autopilot_results['metric_name']}: "
                    f"{autopilot_results['metric_value']:.4f}")
    logger.info("  └──────────────────────────────────────────────────────────┘")

    # ================================================================
    # 7. Salvar métricas consolidadas no S3
    # ================================================================
    _phase(7, "Salvar métricas consolidadas no S3")
    save_metrics(
        bucket=args.bucket,
        region=args.region,
        experiment_name=experiment_name,
        estimator=estimator,
        tuner=tuner,
        warm_start_params=warm_start_params,
        autopilot_job=autopilot_job,
        autopilot_results=autopilot_results,
        use_pipeline=args.use_pipeline,
        skip_feature_store=args.skip_feature_store,
    )
    _phase(7, "", end=True)

    # ================================================================
    # 8. Deploy dos endpoints
    # ================================================================
    _phase(8, "Deploy dos endpoints de inferência")
    if not args.skip_deploy:
        if estimator is None:
            logger.warning("Deploy do modelo GA ignorado: training job não foi concluído com sucesso.")
        else:
            # Deploy do modelo GA (endpoint principal com inference.py customizado)
            endpoint_name = deploy_sagemaker_endpoint(
                estimator, args.project, args.endpoint_instance_type, role_arn=role_arn
            )
            logger.info(f"Endpoint GA ativo: {endpoint_name}")

        # Deploy do Autopilot (endpoint separado para comparação)
        if autopilot_results and auto_ml:
            autopilot_endpoint = deploy_autopilot_endpoint(
                auto_ml, autopilot_job, session,
                args.project, args.endpoint_instance_type
            )
            logger.info(f"Endpoint Autopilot ativo: {autopilot_endpoint}")
    else:
        logger.info("Deploy dos endpoints ignorado (--skip-deploy).")
    _phase(8, "", end=True)

    bar = "═" * 58
    logger.info(f"╔{bar}╗")
    logger.info("║  ✓ Pipeline concluído com sucesso!                      ║")
    logger.info(f"║  Fim: {datetime.now(BRT).strftime('%Y-%m-%d %H:%M:%S')}                               ║")
    logger.info(f"╚{bar}╝")


if __name__ == "__main__":
    main()
