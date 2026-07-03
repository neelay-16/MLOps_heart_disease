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

# ====================== CORS Middleware ======================
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


# ====================== Mock Database Logging Function ======================
def log_prediction_to_db(patient_id: Optional[str], features: dict, prediction: int, probability: float, model_version: str = "v1"):
    """
    Logs prediction results and metrics into the analytical tracking database.
    Replace this print statement with your actual PostgreSQL connection / insert logic.
    """
    try:
        print("\n📝 [DB LOGGING] Saving record to PostgreSQL...")
        print(f"   Patient ID:    {patient_id}")
        print(f"   Prediction:    {prediction}")
        print(f"   Probability:   {probability}")
        print(f"   Model Version: {model_version}")
        print(f"   Features Count: {len(features) if features else 0}")
    except Exception as db_err:
        print(f"⚠️ Failed to log prediction metrics to tracking database: {db_err}")


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

        # Case 1: Prediction using an existing Patient ID
        if request.patient_id:
            features = online_store.get_patient_features(request.patient_id)
            if not features:
                raise HTTPException(status_code=404, detail=f"No features found for patient_id: {request.patient_id}")
            
            # Pass patient_id directly to use the class routing logic safely
            result = predictor.predict_online(patient_id=request.patient_id)
        
        # Case 2: Prediction using raw manual features provided via the API body
        elif request.features:
            features = request.features
            result = predictor.predict_online(feature_dict=features)
        
        else:
            raise HTTPException(status_code=400, detail="Please provide either 'patient_id' or 'features'")

        # Parse final prediction statistics from the pipeline results payload
        # Falls back to parsing individual model values if keys are structured inside nested dicts
        final_prediction = result.get("final_prediction")
        avg_probability = result.get("average_probability")

        # If your pipeline class outputs fields under alternate key names, parse them safely here:
        if final_prediction is None:
            # Look inside individual model results or calculate the aggregate if needed
            pass

        # Log metrics to database
        log_prediction_to_db(
            patient_id=request.patient_id,  # Passes None gracefully if manually input
            features=features,
            prediction=final_prediction if final_prediction is None else int(final_prediction),
            probability=avg_probability if avg_probability is None else float(avg_probability),
            model_version="v1"
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ====================== Run Server ===========================

if __name__ == "__main__":
    uvicorn.run("application:app", host="0.0.0.0", port=8000)
