from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import onnxruntime as ort
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sess = ort.InferenceSession("models/addition_model.onnx")
input_name = sess.get_inputs()[0].name

def build_oddeven_model():
    model = Sequential([
        Dense(16, activation='relu', input_shape=(10,)),
        Dense(1, activation='sigmoid')
    ])
    return model

odd_even_model = build_oddeven_model()
weights = np.load("models/odd_even_model.npz")
odd_even_model.set_weights([weights['W1'], weights['b1'], weights['W2'], weights['b2']])

class Project(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    type: str

projects_db = [
    {"id": 1, "name": "ML Sum Predictor", "description": "Neural network that predicts sum of two numbers", "type": "sum"},
    {"id": 2, "name": "Odd/Even Predictor", "description": "Neural network that predicts if a number is odd or even", "type": "oddeven"}
]

@app.get("/")
def root():
    return {"status": "running"}

@app.get("/api/projects")
def get_projects():
    return projects_db

@app.get("/api/projects/{project_id}")
def get_project(project_id: int):
    for project in projects_db:
        if project["id"] == project_id:
            return project
    raise HTTPException(status_code=404, detail="Project not found")

@app.post("/predict/sum")
def predict_sum(a: float, b: float):
    x = np.array([[a,b]], dtype=np.float32) / 100
    y = sess.run(None, {input_name: x})[0][0][0] * 200
    return {"result": float(y)}

@app.post("/predict/oddeven")
def predict_oddeven(n: int):
    x = ((np.array([[n]]) & (1 << np.arange(10))) > 0).astype(float)
    y = odd_even_model.predict(x)[0][0]
    is_even = y < 0.5
    return {"result": bool(is_even)}
