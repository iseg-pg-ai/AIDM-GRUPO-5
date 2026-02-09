from flask import Flask
import flask
import os
import json
import joblib
import pandas as pd
import io

# Load in model (SageMaker mounts model artifacts under /opt/ml/model/)
MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/ml/model/model.joblib")
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))

model = joblib.load(MODEL_PATH)


# The flask app for serving predictions
app = Flask(__name__)
@app.route('/ping', methods=['GET'])
def ping():
    # Check if the classifier was loaded correctly
    health = model is not None
    status = 200 if health else 404
    return flask.Response(response= '\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
# Lê o Content-Type do request HTTP e evita erro se header não existir
    ct = (flask.request.content_type or "").lower()
    
    if "text/csv" in ct:
        body = flask.request.data.decode("utf-8")
        # Espera CSV com header
        X = pd.read_csv(io.StringIO(body))
    else:
        # Processa inputa
        input_json = flask.request.get_json()
        # Formatos Aceites:
        # 1) {"input": {...}}  (apenas 1 linha)
        # 2) {"input": [{...}, {...}]} (múltiplas linhas)
        # 3) {"instances": [{...}, {...}]} (formato comum)
        payload = None # Iniciar a variável payload
        if isinstance(input_json, dict):
            if "instances" in input_json:
                payload = input_json["instances"] # instances é um padrão muito comum em serving de ML (SDK's de ML, SageMaker etc...)
            else:
                payload = input_json.get("input")
        # Se o JSON não for um dict
        else:
            payload = input_json

        if payload is None:
            return flask.Response(
                response=json.dumps({"error": "Missing 'input' or 'instances' in JSON body."}),
                status=400,
                mimetype='application/json'
            )
        # Trata casos de apenas uma linha e também de múltiplas linhas. Dicionários e listas de dicioários
        X = pd.DataFrame([payload]) if isinstance(payload, dict) else pd.DataFrame(payload)


    # Predict
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)

    # Retorna probabilidade + classe
    if "text/csv" in ct:
        # CSV output: probabilidade,previsão
        out_df = pd.DataFrame({
            "probability": [float(p) for p in proba],
            "prediction": [int(y) for y in pred],
        })
        return flask.Response(response=out_df.to_csv(index=False), status=200, mimetype="text/csv")
    else:
        # JSON output
        result = {
            "predictions": [
                {"probability": float(p), "prediction": int(y)}
                for p, y in zip(proba, pred)
            ],
            "threshold": THRESHOLD
        }
        return flask.Response(response=json.dumps(result), status=200, mimetype="application/json")