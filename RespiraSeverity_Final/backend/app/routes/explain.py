import io
from PIL import Image
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from backend.app.services.image_validator import validate_xray_image
from backend.app.services.ml_service import MLService

router = APIRouter(prefix="/api", tags=["Explainability"])
ml_service = MLService()

@router.post("/explain/gradcam")
async def explain_gradcam_endpoint(
    file: UploadFile = File(...),
    class_idx: int = Query(None, description="Optional class index override (0=Normal, 1=Pneumonia, 2=TB, 3=COVID)")
):
    """
    Computes Grad-CAM feature attribution on the last inverted residual layer of MobileNetV3-Small.
    Returns base64 encoded original, heatmap, and overlay images.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a chest X-ray image.")

    content = await file.read()
    val = validate_xray_image(content, filename=file.filename or "upload.jpg")
    if not val["valid"]:
        raise HTTPException(status_code=422, detail=val["error"])

    try:
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
        cam_result = ml_service.generate_gradcam(pil_img, class_idx=class_idx)
        return cam_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GradCAM generation failed: {str(e)}")
