import io
import numpy as np
from PIL import Image

def validate_xray_image(image_bytes: bytes, filename: str = "upload.jpg") -> dict:
    """
    Validates uploaded chest X-ray image for technical quality:
    - Format verification (JPEG, PNG)
    - File corruption check
    - Resolution boundaries
    - Blur detection (Laplacian variance)
    - Brightness and contrast boundaries
    """
    if not image_bytes or len(image_bytes) == 0:
        return {
            "valid": False,
            "quality_score": 0.0,
            "status": "REJECTED",
            "warnings": ["Uploaded file is empty (0 bytes)."],
            "error": "Empty file received. Please upload a valid chest X-ray image.",
            "metrics": {}
        }

    # 1. Format and corruption check
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image.verify()  # Verify integrity
        # Re-open because verify() closes the file pointer in PIL
        pil_image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        return {
            "valid": False,
            "quality_score": 0.0,
            "status": "CORRUPT",
            "warnings": [f"Cannot decode image file: {str(e)}"],
            "error": "The image file appears corrupted or in an unsupported format. Please upload a valid JPG or PNG file.",
            "metrics": {}
        }

    img_format = (pil_image.format or "UNKNOWN").upper()
    if img_format not in ["JPEG", "JPG", "PNG"]:
        return {
            "valid": False,
            "quality_score": 0.0,
            "status": "UNSUPPORTED_FORMAT",
            "warnings": [f"Unsupported format '{img_format}'. Supported formats: JPEG, PNG."],
            "error": f"Unsupported format '{img_format}'. Please upload a standard JPEG or PNG image.",
            "metrics": {"format": img_format}
        }

    width, height = pil_image.size
    warnings = []
    deductions = 0.0

    # 2. Resolution check
    if width < 128 or height < 128:
        return {
            "valid": False,
            "quality_score": 10.0,
            "status": "LOW_RESOLUTION",
            "warnings": [f"Image resolution ({width}x{height}) is extremely low (minimum required: 128x128)."],
            "error": f"Image resolution {width}x{height} is too low for reliable neural network feature extraction.",
            "metrics": {"width": width, "height": height, "format": img_format}
        }

    if width < 224 or height < 224:
        warnings.append(f"Image resolution ({width}x{height}) is below standard 224x224. Image will be upscaled, which may soften fine edge features.")
        deductions += 15.0

    # Convert to grayscale numpy array for photometric and focus analysis
    gray_img = pil_image.convert("L")
    arr = np.asarray(gray_img, dtype=np.float32)

    # 3. Brightness check (mean intensity 0-255)
    mean_intensity = float(np.mean(arr))
    if mean_intensity < 15.0:
        return {
            "valid": False,
            "quality_score": 15.0,
            "status": "UNDEREXPOSED",
            "warnings": ["Image is extremely dark/underexposed (mean intensity < 15/255)."],
            "error": "Image is excessively dark or underexposed, obscuring thoracic anatomy.",
            "metrics": {"mean_intensity": round(mean_intensity, 2), "width": width, "height": height}
        }
    elif mean_intensity > 240.0:
        return {
            "valid": False,
            "quality_score": 15.0,
            "status": "OVEREXPOSED",
            "warnings": ["Image is washed out/overexposed (mean intensity > 240/255)."],
            "error": "Image is washed out or overexposed, causing loss of lung field details.",
            "metrics": {"mean_intensity": round(mean_intensity, 2), "width": width, "height": height}
        }
    elif mean_intensity < 40.0:
        warnings.append(f"Low overall brightness (mean: {mean_intensity:.1f}/255). Sub-optimal contrast.")
        deductions += 10.0
    elif mean_intensity > 210.0:
        warnings.append(f"High overall brightness (mean: {mean_intensity:.1f}/255). Highlights may be clipped.")
        deductions += 10.0

    # 4. Contrast check (standard deviation 0-255)
    std_contrast = float(np.std(arr))
    if std_contrast < 10.0:
        return {
            "valid": False,
            "quality_score": 10.0,
            "status": "LOW_CONTRAST",
            "warnings": ["Image has virtually zero dynamic range/contrast (std < 10). Likely blank or solid tone."],
            "error": "Image has insufficient contrast (almost solid tone) and cannot be interpreted.",
            "metrics": {"contrast_std": round(std_contrast, 2), "width": width, "height": height}
        }
    elif std_contrast < 25.0:
        warnings.append(f"Sub-optimal dynamic contrast (std: {std_contrast:.1f}). Subtle infiltrates may be hard to distinguish.")
        deductions += 12.0

    # 5. Blur metric: Variance of discrete Laplacian
    # 3x3 Laplacian kernel approximation: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    laplacian = (
        np.roll(arr, 1, axis=0) + np.roll(arr, -1, axis=0) +
        np.roll(arr, 1, axis=1) + np.roll(arr, -1, axis=1) - 4 * arr
    )[1:-1, 1:-1]
    laplacian_var = float(np.var(laplacian))

    if laplacian_var < 15.0:
        return {
            "valid": False,
            "quality_score": 20.0,
            "status": "EXCESSIVE_BLUR",
            "warnings": [f"Laplacian focus variance ({laplacian_var:.1f}) indicates severe blurring."],
            "error": "Image has severe motion or optical blur. Diagnostic lung markings are obscured.",
            "metrics": {"blur_variance": round(laplacian_var, 2), "width": width, "height": height}
        }
    elif laplacian_var < 45.0:
        warnings.append(f"Moderate blur detected (Laplacian variance: {laplacian_var:.1f}). Sharpness is degraded.")
        deductions += 10.0

    # 6. Aspect Ratio check
    aspect_ratio = float(width) / float(height)
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        warnings.append(f"Unusual image aspect ratio ({aspect_ratio:.2f}). Standard frontal radiographs typically range 0.75-1.3.")
        deductions += 8.0

    # Compute overall quality score (0 to 100)
    quality_score = max(25.0, min(100.0, 100.0 - deductions))

    return {
        "valid": True,
        "quality_score": round(quality_score, 1),
        "status": "PASSED" if quality_score >= 70.0 else "SUBOPTIMAL",
        "warnings": warnings,
        "error": None,
        "metrics": {
            "width": width,
            "height": height,
            "format": img_format,
            "mean_intensity": round(mean_intensity, 2),
            "contrast_std": round(std_contrast, 2),
            "blur_variance": round(laplacian_var, 2),
            "aspect_ratio": round(aspect_ratio, 2)
        }
    }
