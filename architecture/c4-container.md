flowchart TB
  user[ML Engineer / AIDM-Grupo-5]

  subgraph boundary["AIDM Loan Default MLOps System"]
    studio[SageMaker Studio / Notebook<br/>Python Orchestrator]
    s3[(Amazon S3<br/>i32419/ai-deployment-monitoring-grupo-5)]
    ecr[(Amazon ECR<br/>BYOC Image)]
    train[SageMaker Training Job<br/>scikit-learn container]
    hpo[SageMaker Hyperparameter Tuning Job]
    reg[SageMaker Model Registry<br/>Model Package Group]
    card[SageMaker Model Card + Dashboard]
    endpoint[SageMaker Endpoint<br/>BYOC Inference Container]
    datacapture[Endpoint Data Capture<br/>Inference inputs]
    monitor[SageMaker Model Monitor<br/>Data Quality Schedule]
    cw[CloudWatch Logs / Metrics]
    mlflow[MLflow Tracking Server (ARN)]
  end

  user --> studio

  %% Data preparation
  studio --> s3

  %% Training & HPO
  studio --> train
  studio --> hpo
  train --> s3
  hpo --> s3
  train --> mlflow
  hpo --> mlflow

  %% Registry & governance
  studio --> reg
  reg --> s3
  studio --> card

  %% BYOC build & deploy
  studio --> ecr
  ecr --> endpoint

  %% Inference & data capture
  endpoint --> datacapture
  datacapture --> s3

  %% Data quality monitoring
  studio --> monitor
  monitor --> s3
  monitor --> cw
  endpoint --> cw
