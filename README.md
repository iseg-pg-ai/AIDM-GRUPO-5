AIDM – Loan Default Prediction (Grupo 5)

Este projeto implementa uma solução completa de MLOps em AWS SageMaker para previsão de loan default, cobrindo treino, HPO, tracking, governação, deployment, monitorização e simulação de data drift, conforme o enunciado da unidade curricular AI Deployment & Monitoring (AIDM).

Objetivo

Construir um pipeline end-to-end de Machine Learning com:

 - Treino e HPO em SageMaker
 - Tracking e governação com MLflow e Model Registry
 - Deployment em endpoint real-time via BYOC
 - Monitorização de qualidade de dados e drift
 - Model Card e dashboards de governação

Arquitetura

 - A solução segue o modelo C4, com:
 - System Context Diagram
 - Container Diagram

Os diagramas encontram-se em architecture/ e são descritos em Mermaid (renderizados nativamente pelo GitHub).

Modelo

 - Algoritmo: XGBoost (classificação binária)
 - Target: Status (loan default)
 - Output: probabilidade + classe (threshold = 0.5 mas configurável)
 - Pré-processamento: imputação + one-hot encoding via sklearn Pipeline

Pipeline MLOps

 - Preparação de dados (Notebook)
 - Treino + HPO com SageMaker SKLearn Estimator
 - MLflow tracking (parâmetros, métricas, artefactos)
 - Model Registry (Model Package Group + aprovação)
 - Model Card & Dashboard
 - BYOC inference endpoint
 - Data Capture + Model Monitor
 - Simulação de Data Drift

Monitorização & Drift

 - Baseline gerado a partir do dataset de treino
 - Data Quality Monitor com schedules horários
 - Drift simulado alterando distribuições numéricas
 - Métricas persistidas em S3 e CloudWatch

Estrutura do Projeto
.
├── 01_training_hpo_mlflow.ipynb
├── 02_byoc_build_push_deploy.ipynb
├── 03_model_governance_monitoring.ipynb
├── training_code/
│   ├── train.py
│   └── requirements.txt
├── BYOC-Single-Model/
│   ├── container/
│        ├── Classificador-binario/
│        │    ├── nginx.conf
│        │    ├── wsgi.py
│        │    ├── serve
│        │    ├── predictor.py
│        │    ├── main.py
│        │    ├── pyproject.toml
│        │    └── uv.lock
│        ├── Dockerfile 
│        └── .dockerignore 
├── architecture/
│   ├── c4-system-context.md
│   └── c4-container.md
├── Dataset/
│    └── Loan_Default.csv
├── data/
│    ├── tain.csv
│    └── validation.csv
├── baseline_features.csv
├── main.py
└── README.md

Tecnologias

 - AWS SageMaker (Training, HPO, Endpoint, Monitor)
 - MLflow + SageMaker MLflow Tracking Server
 - Docker (BYOC)
 - XGBoost, scikit-learn
 - Mermaid (arquitetura)

Equipa

AIDM - Grupo 5:
Pedro Oliveira
Marco Moucho
Tiago Gonçalves
Marta Galvão
Rebeca Menezes