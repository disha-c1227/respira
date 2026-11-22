from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class ImageValidationResponse(BaseModel):
    valid: bool
    quality_score: float = Field(..., ge=0, le=100)
    status: str
    warnings: List[str] = []
    error: Optional[str] = None
    metrics: Dict[str, Any] = {}

class DiseasePredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    raw_probabilities: Dict[str, float]
    probabilities: Dict[str, float]
    routing_destination: str
    explanation: str
    is_uncertain: bool = False
    rejection_status: str = "ACCEPTED"  # "ACCEPTED" or "REJECTED_FOR_MANUAL_REVIEW"
    confidence_threshold: float = 70.0
    entropy: float = 0.0
    margin: float = 0.0
    disclaimer: str

class CovidSeverityResponse(BaseModel):
    severity: str
    confidence: float
    probabilities: Dict[str, float]
    dataset_context: str
    disclaimer: str

class ClusterPredictionResponse(BaseModel):
    disease: str
    assigned_cluster: int
    cluster_name: str
    cluster_interpretation: str
    mean_opacity_percentile: Optional[float] = None
    nearest_covid_reference_centroid: str
    distance_to_covid_centroids: Dict[str, float]
    cluster_metrics: Dict[str, Any]
    is_exploratory: bool = True
    disclaimer: str

class GradCamResponse(BaseModel):
    predicted_class: str
    confidence: float
    original_base64: str
    heatmap_base64: str
    overlay_base64: str
    attention_summary: str
    target_layer: str
    disclaimer: str

class ModelInfoResponse(BaseModel):
    project_title: str
    architecture: str
    device: str
    disease_model_status: str
    disease_model_metrics: Dict[str, Any]
    covid_severity_status: str
    covid_severity_metrics: Dict[str, Any]
    clustering_status: str
    clustering_metrics: Dict[str, Any]
    dataset_summary: Dict[str, Any]
