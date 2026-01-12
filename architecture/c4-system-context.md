flowchart LR
  user[ML Engineer / AIDM-Grupo-5] --> studio[SageMaker Studio / Notebook]

  subgraph sys["AIDM Loan Default MLOps System"]
    studio --> sm[SageMaker APIs (Boto3 / SageMaker SDK)]
  end

  sm --> s3[(Amazon S3: i32419/ai-deployment-monitoring-grupo-5<br/>datasets, outputs, baselines)]
  sm --> train[SageMaker Training Job]
  sm --> hpo[SageMaker Hyperparameter Tuning Job]
  sm --> reg[SageMaker Model Registry<br/>(Model Package Group)]
  sm --> card[SageMaker Model Card + Model Dashboard]
  sm --> monitor[SageMaker Model Monitor<br/>Monitoring Schedules]
  sm --> endpoint[SageMaker Real-time Endpoint (BYOC)]

  train --> s3
  hpo --> s3
  train --> mlflow[MLflow Tracking Server (ARN)]
  hpo --> mlflow

  endpoint --> cw[CloudWatch Logs/Metrics]
  monitor --> cw
  endpoint --> s3
  monitor --> s3

  github[GitHub Repo / Pull Request] --> user
