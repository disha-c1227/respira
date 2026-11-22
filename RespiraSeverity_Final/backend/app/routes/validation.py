from fastapi import APIRouter, File, UploadFile, HTTPException
from backend.app.services.image_validator import validate_xray_image

router = APIRouter(prefix="/api", tags=["Validation"])

@router.post("/validate-image")
async def validate_image_endpoint(file: UploadFile = File(...)):
    """
    Validates uploaded chest X-ray image for technical quality:
    corrupted bytes, resolution, excessive blur, extreme underexposure/overexposure, and dynamic contrast.
    """
    try:
        content = await file.read()
        res = validate_xray_image(content, filename=file.filename or "upload.jpg")
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image validation error: {str(e)}")
