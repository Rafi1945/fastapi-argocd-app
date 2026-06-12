from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os


app = FastAPI(title="Iris ML Prediction API")


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


MODEL_PATH = os.path.join(os.path.dirname(__file__), "iris_model.pkl")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


class_names = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}


@app.get("/")
def home():
    return {
        "message": "Iris ML Prediction API deployed using Argo CD",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(data: IrisInput):
    input_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(input_data)[0]

    return {
        "prediction": int(prediction),
        "class_name": class_names[int(prediction)]
    }