from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from typing import Dict, Any, Optional
from backend.app.services.report_service import generate_pdf_report

router = APIRouter(prefix="/api", tags=["Report Generation"])

class ReportPayload(BaseModel):
    image_id: Optional[str] = "CXR_SAMPLE_001"
    validation_data: Dict[str, Any]
    disease_data: Dict[str, Any]
    severity_data: Optional[Dict[str, Any]] = None
    clustering_data: Optional[Dict[str, Any]] = None
    gradcam_images: Optional[Dict[str, Any]] = None

@router.post("/report/generate")
async def generate_report_endpoint(payload: ReportPayload):
    """
    Generates and downloads a structured clinical research PDF report
    with embedded classification tables, routing conclusions, and Grad-CAM visualizations.
    """
    try:
        pdf_bytes = generate_pdf_report(
            image_id=payload.image_id or "CXR_QUERY",
            validation_data=payload.validation_data,
            disease_data=payload.disease_data,
            severity_data=payload.severity_data,
            clustering_data=payload.clustering_data,
            gradcam_images=payload.gradcam_images
        )
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=RespiraSeverity_Report_{payload.image_id}.pdf"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF report generation failed: {str(e)}")
