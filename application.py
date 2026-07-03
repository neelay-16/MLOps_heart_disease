from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.online_feature_store import OnlineFeatureStore
from pipeline.online_prediction_pipeline import OnlineHeartDiseasePrediction

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Real-time Heart Disease Prediction using Online Feature Store",
    version="1.0.0"
)

# ====================== CORS Middleware (Important Fix) ======================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # Allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],           # Allow all methods including OPTIONS
    allow_headers=["*"],
)

# ====================== Mount Frontend ======================
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

# Global objects
predictor = None
online_store = None


@app.on_event("startup")
async def startup_event():
    global predictor, online_store
    try:
        print("🚀 Starting Heart Disease Prediction API...")
        predictor = OnlineHeartDiseasePrediction()
        online_store = OnlineFeatureStore()
        print("✅ Models and Online Feature Store loaded successfully!")
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise


# ====================== Request Model ======================

class PredictionRequest(BaseModel):
    patient_id: Optional[str] = None
    features: Optional[Dict[str, Any]] = None


# ====================== API Endpoints ======================

@app.get("/")
def serve_frontend():
    """Serve the HTML frontend"""
    return FileResponse("templates/index.html")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": len(predictor.models) if predictor else 0
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        if not predictor:
            raise HTTPException(status_code=503, detail="Models not loaded yet")

        if request.patient_id:
            features = online_store.get_patient_features(request.patient_id)
            if not features:
                raise HTTPException(status_code=404, detail=f"No features found for patient_id: {request.patient_id}")
            result = predictor.predict_online(feature_dict=features)
        
        elif request.features:
            result = predictor.predict_online(feature_dict=request.features)
        
        else:
            raise HTTPException(status_code=400, detail="Please provide either 'patient_id' or 'features'")

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ====================== Run Server ===========================

if __name__ == "__main__":
    uvicorn.run("application:app", host="0.0.0.0", port=8000)