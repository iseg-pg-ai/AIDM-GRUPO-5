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
    ct = (flask.request.content_type or "").lower()
    
    if "text/csv" in ct:
        body = flask.request.data.decode("utf-8")
        # Espera CSV com header
        X = pd.read_csv(io.StringIO(body))
    else:
        #Process input
        input_json = flask.request.get_json()
        # Formatos Aceites:
        # 1) {"input": {...}}  (single row)
        # 2) {"input": [{...}, {...}]} (multiple rows)
        # 3) {"instances": [{...}, {...}]} (common format)
        payload = None
        if isinstance(input_json, dict):
            if "instances" in input_json:
                payload = input_json["instances"]
            else:
                payload = input_json.get("input")
        else:
            payload = input_json

        if payload is None:
            return flask.Response(
                response=json.dumps({"error": "Missing 'input' or 'instances' in JSON body."}),
                status=400,
                mimetype='application/json'
            )

        X = pd.DataFrame([payload]) if isinstance(payload, dict) else pd.DataFrame(payload)


    # Predict
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)

    # Return probability + class
    if "text/csv" in ct:
        # CSV output: probability,prediction
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