flowchart TB
  user["ML Engineer / AIDM-Grupo-5"]

  subgraph boundary["AIDM Loan Default MLOps System"]
    studio["SageMaker Studio / Notebook<br/>Python Orchestrator"]
    train["SageMaker Training Job<br/>(scikit-learn container)"]
    hpo["SageMaker Hyperparameter Tuning Job"]
    reg["SageMaker Model Registry<br/>(Model Package Group)"]
    card["SageMaker Model Card + Dashboard"]
    endpoint["SageMaker Endpoint<br/>(BYOC Inference Container)"]
    datacapture["Endpoint Data Capture<br/>(Inference inputs)"]
    monitor["SageMaker Model Monitor<br/>(Data Quality Schedule)"]
  end

  s3[("Amazon S3<br/>i32419/ai-deployment-monitoring-grupo-5<br/>datasets, outputs, baselines")]
  ecr[("Amazon ECR<br/>BYOC Image")]
  cw["CloudWatch Logs / Metrics"]
  mlflow["MLflow Tracking Server (ARN)"]

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
