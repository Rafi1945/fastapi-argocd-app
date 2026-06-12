from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {
        "message": "FastAPI app deployed using Argo CD",
        "status": "running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }