from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import traceback


app = FastAPI()

# ✅ Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load Model
try:
    model = joblib.load("aqi_model.pkl")
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Failed to load model:", e)
    traceback.print_exc()

# Input Schema
from pydantic import BaseModel

class AQIInput(BaseModel):
    so2: float
    co: float
    o3: float
    o3_8hr: float
    pm10: float
    pm2_5: float
    no2: float
    nox: float
    no: float
    windspeed: float
    winddirec: float
    co_8hr: float
    pm2_5_avg: float
    pm10_avg: float
    so2_avg: float

# Prediction Endpoint
@app.post("/predict")
def predict_aqi(data: AQIInput):
    try:
        print(f"Received input: {data}")
        input_data = [[
    data.so2, data.co, data.o3, data.o3_8hr, data.pm10, data.pm2_5,
    data.no2, data.nox, data.no, data.windspeed, data.winddirec,
    data.co_8hr, data.pm2_5_avg, data.pm10_avg, data.so2_avg
]]
        print(f"Prepared input for model: {input_data}")
        prediction = model.predict(input_data)
        print(f"Prediction result: {prediction}")
        return {"predicted_aqi": prediction[0]}
    except Exception as e:
        print("❌ Prediction failed:", e)
        traceback.print_exc()
        return {"error": str(e)}


