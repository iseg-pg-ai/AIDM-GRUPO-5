# C4 — System Context

```mermaid
flowchart TB
  user[ML Engineer / AIDM-Grupo-5]

  subgraph boundary["AIDM Loan Default MLOps System"]
    studio[SageMaker Studio / Notebook Python Orchestrator]
    train[SageMaker Training Job\n(scikit-learn container)]
    hpo[SageMaker Hyperparameter Tuning Job]
    reg[SageMaker Model Registry\n(Model Package Group)]
    card[SageMaker Model Card + Dashboard]
    endpoint[SageMaker Endpoint\n(BYOC Inference Container)]
    datacapture[Endpoint Data Capture\n(Inference inputs)]
    monitor[SageMaker Model Monitor\n(Data Quality Schedule)]
  end

  s3[(Amazon S3\n i32419/ai-deployment-monitoring-grupo-5\n datasets, outputs, baselines)]
  ecr[(Amazon ECR\nBYOC Image)]
  cw[CloudWatch Logs / Metrics]
  mlflow[MLflow Tracking Server (ARN)]

  user --> studio
  studio --> s3

  studio --> train
  studio --> hpo
  train --> s3
  hpo --> s3
  train --> mlflow
  hpo --> mlflow

  studio --> reg
  reg --> s3
  studio --> card

  studio --> ecr
  ecr --> endpoint

  endpoint --> datacapture
  datacapture --> s3
  endpoint --> cw

  studio --> monitor
  monitor --> s3
  monitor --> cw