from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import onnxruntime as ort
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sess = ort.InferenceSession("models/addition_model.onnx")
input_name = sess.get_inputs()[0].name

churn_sess = ort.InferenceSession("models/churn_model.onnx")
churn_input_name = churn_sess.get_inputs()[0].name

churn_scaler = joblib.load("models/scaler_churn.pkl")

cnn_sess = ort.InferenceSession("models/cnn_model.onnx")
cnn_input_name = cnn_sess.get_inputs()[0].name

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

class ChurnRequest(BaseModel):
    CreditScore: float
    Geography: str
    Gender: str
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: float
    HasCrCard: float
    IsActiveMember: float
    EstimatedSalary: float

projects_db = [
    {"id": 1, "name": "ML Sum Predictor", "description": "Neural network that predicts sum of two numbers", "type": "sum"},
    {"id": 2, "name": "Odd/Even Predictor", "description": "Neural network that predicts if a number is odd or even", "type": "oddeven"},
    {"id": 3, "name": "Churn Predictor", "description": "Neural network that predicts customer churn", "type": "churn"},
    {"id": 4, "name": "Image Classifier", "description": "CNN model that classifies images", "type": "cnn"}
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

@app.post("/predict/churn")
def predict_churn(request: ChurnRequest):
    CreditScore = request.CreditScore
    Geography = request.Geography
    Gender = request.Gender
    Age = request.Age
    Tenure = request.Tenure
    Balance = request.Balance
    NumOfProducts = request.NumOfProducts
    HasCrCard = request.HasCrCard
    IsActiveMember = request.IsActiveMember
    EstimatedSalary = request.EstimatedSalary
    
    geography_france = 1 if Geography == "France" else 0
    geography_germany = 1 if Geography == "Germany" else 0
    geography_spain = 1 if Geography == "Spain" else 0
    
    gender = 1 if Gender == "Male" else 0
    
    x = np.array([[
        CreditScore,
        gender,
        Age,
        Tenure,
        Balance,
        NumOfProducts,
        HasCrCard,
        IsActiveMember,
        EstimatedSalary,
        geography_france,
        geography_germany,
        geography_spain
    ]], dtype=np.float32)
    
    x_scaled = churn_scaler.transform(x)
    y = churn_sess.run(None, {churn_input_name: x_scaled})[0][0][0]
    will_churn = y > 0.34
    return {"result": bool(will_churn), "probability": float(y)}

@app.post("/predict/cnn")
async def predict_cnn(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((32, 32))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    y = cnn_sess.run(None, {cnn_input_name: img_array})[0][0]
    predicted_class = int(np.argmax(y))
    confidence = float(np.max(y))
    
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    return {"class": class_names[predicted_class], "confidence": confidence}
