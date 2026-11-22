import io
from PIL import Image
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from backend.app.services.image_validator import validate_xray_image
from backend.app.services.ml_service import MLService

router = APIRouter(prefix="/api", tags=["Prediction"])
ml_service = MLService()

async def read_and_validate(file: UploadFile) -> Image.Image:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a chest X-ray image (JPEG or PNG).")
    
    content = await file.read()
    val = validate_xray_image(content, filename=file.filename or "upload.jpg")
    if not val["valid"]:
        raise HTTPException(status_code=422, detail={
            "message": val["error"] or "Image failed quality validation checks.",
            "status": val["status"],
            "quality_score": val["quality_score"],
            "warnings": val["warnings"]
        })
    try:
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
        return pil_img, val
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {str(e)}")

@router.post("/predict/disease")
async def predict_disease_endpoint(
    file: UploadFile = File(...),
    threshold: float = Query(70.0, ge=0.0, le=100.0, description="Selective classification confidence threshold (%)")
):
    """
    Performs 4-class pulmonary disease classification with uncertainty & selective rejection.
    """
    pil_img, val = await read_and_validate(file)
    try:
        pred = ml_service.predict_disease(pil_img, threshold=threshold)
        pred["quality_validation"] = val
        return pred
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Disease prediction failed: {str(e)}")

@router.post("/predict/covid-severity")
async def predict_covid_severity_endpoint(file: UploadFile = File(...)):
    """
    Performs supervised 3-class COVID-19 severity assessment (Mild, Moderate, Severe).
    """
    pil_img, val = await read_and_validate(file)
    try:
        sev = ml_service.predict_covid_severity(pil_img)
        sev["quality_validation"] = val
        return sev
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"COVID severity prediction failed: {str(e)}")

@router.post("/predict/pneumonia-cluster")
async def predict_pneumonia_cluster_endpoint(file: UploadFile = File(...)):
    """
    Assigns Pneumonia radiograph to exploratory K-means feature cluster (k=3)
    and computes proximity to COVID-19 reference centroids.
    """
    pil_img, val = await read_and_validate(file)
    try:
        cluster_res = ml_service.predict_clustering(pil_img, disease_type="Pneumonia")
        cluster_res["quality_validation"] = val
        return cluster_res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pneumonia clustering failed: {str(e)}")

@router.post("/predict/tb-cluster")
async def predict_tb_cluster_endpoint(file: UploadFile = File(...)):
    """
    Assigns Tuberculosis radiograph to exploratory K-means feature cluster (k=3)
    and computes proximity to COVID-19 reference centroids.
    """
    pil_img, val = await read_and_validate(file)
    try:
        cluster_res = ml_service.predict_clustering(pil_img, disease_type="TB")
        cluster_res["quality_validation"] = val
        return cluster_res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TB clustering failed: {str(e)}")

@router.post("/predict")
async def unified_predict_endpoint(
    file: UploadFile = File(...),
    threshold: float = Query(70.0, ge=0.0, le=100.0, description="Selective classification confidence threshold (%)")
):
    """
    Unified end-to-end pipeline:
    1. Validates image quality
    2. Runs disease classifier with selective rejection threshold
    3. Conditionally routes to appropriate severity module if accepted
    4. Generates Grad-CAM visual explainability
    """
    pil_img, val = await read_and_validate(file)
    disease_res = ml_service.predict_disease(pil_img, threshold=threshold)
    predicted_disease = disease_res["predicted_class"]

    severity_res = None
    clustering_res = None

    # Only route if not rejected by the selective classification mechanism
    if disease_res["rejection_status"] == "ACCEPTED":
        if predicted_disease == "COVID":
            severity_res = ml_service.predict_covid_severity(pil_img)
        elif predicted_disease == "Pneumonia":
            clustering_res = ml_service.predict_clustering(pil_img, disease_type="Pneumonia")
        elif predicted_disease == "TB":
            clustering_res = ml_service.predict_clustering(pil_img, disease_type="TB")

    gradcam_res = None
    try:
        gradcam_res = ml_service.generate_gradcam(pil_img)
    except Exception as e:
        print(f"[Warning] GradCAM generation failed: {e}")

    return {
        "quality": val["quality_score"],
        "quality_validation": val,
        "disease": disease_res,
        "severity": severity_res,
        "clustering": clustering_res,
        "gradcam": gradcam_res,
        "disclaimer": "Academic/research prototype; not a cleared clinical medical diagnosis."
    }
