from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
import pandas as pd
import os


app = FastAPI(title="Iris ML Prediction API with MLflow Registry")


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5000")
MODEL_URI = os.getenv("MODEL_URI", "models:/IrisClassifier/1")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.pyfunc.load_model(MODEL_URI)


class_names = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}


@app.get("/")
def home():
    return {
        "message": "Iris ML Prediction API using MLflow Registry",
        "status": "running",
        "model_uri": MODEL_URI,
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(data: IrisInput):
    input_df = pd.DataFrame(
        [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]],
        columns=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width"
        ]
    )

    prediction = model.predict(input_df)[0]

    return {
        "prediction": int(prediction),
        "class_name": class_names[int(prediction)]
    }