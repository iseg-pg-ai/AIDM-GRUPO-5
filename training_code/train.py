import argparse
import json
import os
import pathlib

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# Pipeline de pré-processamento: imputação de valores em falta + one-hot para categóricas
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
# Classificador XGBoost (não-built-in), embebido num sklearn Pipeline para incluir o pré-processamento
from xgboost import XGBClassifier

import mlflow

def _load_csv(channel_name: str) -> pd.DataFrame:
    base = pathlib.Path("/opt/ml/input/data") / channel_name
    csv_files = list(base.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in channel: {base}")
    return pd.read_csv(csv_files[0])


def main():
    parser = argparse.ArgumentParser()

    # Hiperparametros XGBoost
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)

    args = parser.parse_args()

    train_df = _load_csv("train")
    val_df = _load_csv("validation")

    if "Status" not in train_df.columns:
        raise ValueError("Target column 'Status' not found")

    y_train = train_df["Status"].astype(int)
    X_train = train_df.drop(columns=["Status"])

    y_val = val_df["Status"].astype(int)
    X_val = val_df.drop(columns=["Status"])

    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]
    
# Pipeline de pré-processamento: imputação de valores em falta + one-hot para categóricas.
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )
    
# Classificador XGBoost (não-built-in), embebido num sklearn Pipeline para incluir o pré-processamento
    clf = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_jobs=1,
        random_state=42,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

# Ir buscar às variáveis de ambiente o tracking arn
    tracking_arn = os.environ.get("MLFLOW_TRACKING_ARN")
    if not tracking_arn:
        raise ValueError("Missing env var MLFLOW_TRACKING_ARN")

# Configuração do MLflow para enviar métricas/params para o tracking server gerido no SageMaker
    mlflow.set_tracking_uri(tracking_arn)
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "grupo-5-aidm-loan-default"))

    sm_env = json.loads(os.environ.get("SM_TRAINING_ENV", "{}")) # "SM_TRAINING_ENV" é uma variável ambiente injetada automaticamente pelo SageMaker dentro do container de treino. json.loads é para converter o json num dicionário python
    training_job_name = sm_env.get("job_name", "unknown") # Vamos buscar o job name através da variável ambiente SM_TRAINING_ENV
    
# Iniciar run no MLflow; usamos o nome do Training Job para facilitar auditoria/reprodutibilidade
    with mlflow.start_run(run_name=training_job_name):
        # Treino do pipeline completo (pré-processamento + XGBoost)
        model.fit(X_train, y_train)

        val_proba = model.predict_proba(X_val)[:, 1]
        val_pred = (val_proba >= 0.5).astype(int)

        auc = roc_auc_score(y_val, val_proba)
        acc = accuracy_score(y_val, val_pred)

        mlflow.log_params(
            {
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "learning_rate": args.learning_rate,
                "subsample": args.subsample,
                "colsample_bytree": args.colsample_bytree,
                "model_type": "sklearn_pipeline_xgbclassifier",
                "num_features": len(num_cols),
                "cat_features": len(cat_cols),
            }
        )
        mlflow.log_metrics({"validation_auc": float(auc), "validation_accuracy": float(acc)})

        mlflow.set_tags(
            {
                "training_job_name": training_job_name,
                "dataset": "Loan_Default.csv",
                "task": "binary_classification",
                "target": "Status",
                "git_sha": os.environ.get("GIT_SHA", "unknown"),
            }
        )

        metrics_path = "/opt/ml/output/metrics.json"
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump({"validation_auc": float(auc), "validation_accuracy": float(acc)}, f)
        mlflow.log_artifact(metrics_path)

# Verificar que o diretório /opt/ml/model existe no container e criar caso não exista
        model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model") # SM_MODEL_DIR é uma variável de ambiente que o SageMaker injeta automaticamente dentro do container de treino que indica o diretório onde o SageMaker espera encontrar o modelo final depois do treino terminar
        os.makedirs(model_dir, exist_ok=True)
        
# Guardar o modelo treinado em /opt/ml/model depois o SageMaker empacota tudo dentro de /opt/ml/model em model.tar.gz e envia para o S3 e depois podemos utilizar o model.tar.gz para fazer deployment
        joblib.dump(model, os.path.join(model_dir, "model.joblib"))

        print(f"validation_auc: {auc}")


if __name__ == "__main__":
    main()
