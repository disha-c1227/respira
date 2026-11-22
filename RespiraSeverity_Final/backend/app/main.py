from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.services.ml_service import MLService
from backend.app.routes import validation, prediction, explain, metrics, report

app = FastAPI(
    title="RespiraSeverity API",
    version="2.0.0",
    description="Deep Learning-Based Framework for Pulmonary Disease Classification and Severity Assessment Using Chest X-Rays"
)

# Enable CORS for frontend Vite server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML Service at startup
ml_service = MLService()

@app.get("/api/health", tags=["System"])
def health_check():
    return {
        "status": "HEALTHY",
        "disease_model_loaded": ml_service.disease_model is not None,
        "covid_severity_model_loaded": ml_service.covid_severity_model is not None,
        "pneumonia_clustering_ready": ml_service.pneumonia_kmeans is not None,
        "tb_clustering_ready": ml_service.tb_kmeans is not None,
        "covid_reference_centroids_ready": ml_service.ref_centroids is not None,
        "device": str(ml_service.device),
        "api_version": "2.0.0"
    }

# Include all modular routers
app.include_router(validation.router)
app.include_router(prediction.router)
app.include_router(explain.router)
app.include_router(metrics.router)
app.include_router(report.router)
