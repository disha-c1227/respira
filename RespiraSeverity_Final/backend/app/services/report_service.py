import io
import base64
from datetime import datetime
from PIL import Image

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
)

def generate_pdf_report(
    image_id: str,
    validation_data: dict,
    disease_data: dict,
    severity_data: dict = None,
    clustering_data: dict = None,
    gradcam_images: dict = None
) -> bytes:
    """
    Generates a structured medical research report PDF in memory.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom palette
    navy = colors.HexColor("#0F2942")
    teal = colors.HexColor("#008080")
    light_bg = colors.HexColor("#F8FAFC")
    dark_gray = colors.HexColor("#334155")
    alert_red = colors.HexColor("#DC2626")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=navy
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=teal
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=navy,
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=dark_gray
    )
    bold_body = ParagraphStyle(
        "BoldBody",
        parent=body_style,
        fontName="Helvetica-Bold"
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748B")
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("RespiraSeverity | Clinical Research Analysis Report", title_style))
    story.append(Paragraph("A Deep Learning Framework for Pulmonary Disease Classification and Severity Assessment", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=teal, spaceBefore=2, spaceAfter=10))

    # 2. Metadata Block
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_data = [
        [
            Paragraph(f"<b>Image Identifier:</b> {image_id}", body_style),
            Paragraph(f"<b>Analysis Timestamp:</b> {now_str}", body_style)
        ],
        [
            Paragraph(f"<b>Quality Score:</b> {validation_data.get('quality_score', 'N/A')}/100 ({validation_data.get('status', 'OK')})", body_style),
            Paragraph(f"<b>Primary Backbone:</b> MobileNetV3-Small (Transfer Learning)", body_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[270, 270])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), light_bg),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # 3. Disease Classification Finding
    story.append(Paragraph("1. Primary Disease Classification Finding", section_heading))
    pred_disease = disease_data.get("predicted_class", "Unknown")
    confidence = disease_data.get("confidence", 0.0)
    probs = disease_data.get("probabilities", {})

    disease_summary = [
        [
            Paragraph("<b>Predicted Pulmonary Condition:</b>", bold_body),
            Paragraph(f"<font size=12 color='{navy}'><b>{pred_disease.upper()}</b></font>", bold_body)
        ],
        [
            Paragraph("<b>Model Confidence:</b>", bold_body),
            Paragraph(f"<b>{confidence:.2f}%</b>", bold_body)
        ]
    ]
    dis_table = Table(disease_summary, colWidths=[200, 340])
    dis_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(dis_table)
    story.append(Spacer(1, 6))

    # Probability breakdown table
    prob_rows = [["Disease Class", "Probability", "Status"]]
    for k in ["Normal", "Pneumonia", "TB", "COVID"]:
        val = probs.get(k, probs.get(k.upper(), 0.0))
        status = "Predicted" if k.lower() == pred_disease.lower() else "Non-target"
        prob_rows.append([k, f"{val:.2f}%", status])

    p_table = Table(prob_rows, colWidths=[180, 180, 180])
    p_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), navy),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(p_table)
    story.append(Spacer(1, 12))

    # 4. Severity Assessment Module
    story.append(Paragraph("2. Condition Severity Assessment & Staging", section_heading))
    if pred_disease.upper() == "NORMAL":
        norm_text = "<b>Normal Finding:</b> No pulmonary infection pathology detected. Severity grading is bypassed."
        story.append(Paragraph(norm_text, body_style))
    elif pred_disease.upper() in ["COVID", "COVID-19"] and severity_data:
        sev_grade = severity_data.get("severity", "Unknown")
        sev_conf = severity_data.get("confidence", 0.0)
        sev_probs = severity_data.get("probabilities", {})
        
        sev_rows = [
            [Paragraph("<b>Supervised Severity Grade:</b>", bold_body), Paragraph(f"<b>{sev_grade.upper()}</b>", bold_body)],
            [Paragraph("<b>Confidence:</b>", bold_body), Paragraph(f"{sev_conf:.2f}%", body_style)],
            [Paragraph("<b>Probability Breakdown:</b>", bold_body), Paragraph(", ".join([f"{k}: {v:.1f}%" for k, v in sev_probs.items()]), body_style)],
            [Paragraph("<b>Annotation Source:</b>", bold_body), Paragraph("Trained on 760 clinically annotated COVID-19 cases (RICORD / BIMCV).", disclaimer_style)]
        ]
        sev_table = Table(sev_rows, colWidths=[180, 360])
        sev_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#93C5FD")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DBEAFE")),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(sev_table)
    elif clustering_data:
        cl_id = clustering_data.get("assigned_cluster", 0)
        cl_name = clustering_data.get("cluster_name", f"Cluster {cl_id}")
        cl_interp = clustering_data.get("cluster_interpretation", "Exploratory grouping")
        nearest_covid = clustering_data.get("nearest_covid_reference_centroid", "N/A")
        
        cluster_rows = [
            [Paragraph("<b>Exploratory Cluster Group:</b>", bold_body), Paragraph(f"<b>{cl_name}</b>", bold_body)],
            [Paragraph("<b>Radiographic Profile:</b>", bold_body), Paragraph(cl_interp, body_style)],
            [Paragraph("<b>COVID Centroid Proximity:</b>", bold_body), Paragraph(f"Closest to COVID-19 {nearest_covid} reference centroid in 1024-D embedding space.", body_style)],
            [Paragraph("<b>Clinical Caveat:</b>", bold_body), Paragraph("<b>EXPLORATORY ONLY:</b> Unsupervised K-means cluster derived from radiographic feature embeddings. NOT a validated clinical severity diagnosis.", disclaimer_style)]
        ]
        cl_table = Table(cluster_rows, colWidths=[180, 360])
        cl_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEF3C7")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#FCD34D")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#FDE68A")),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(cl_table)
    else:
        story.append(Paragraph("Severity assessment not applicable or model unavailable for this finding.", body_style))

    story.append(Spacer(1, 12))

    # 5. Explainability (Grad-CAM Visualizations)
    if gradcam_images and "original_base64" in gradcam_images:
        story.append(Paragraph("3. Model Visual Explainability (Grad-CAM)", section_heading))
        story.append(Paragraph("Gradient-weighted Class Activation Mapping highlights regions contributing to the neural network's final decision:", body_style))
        story.append(Spacer(1, 6))

        def b64_to_rl_image(b64_str):
            if not b64_str: return None
            try:
                if "," in b64_str: b64_str = b64_str.split(",")[1]
                img_data = base64.b64decode(b64_str)
                img_buf = io.BytesIO(img_data)
                return RLImage(img_buf, width=160, height=160)
            except Exception:
                return None

        im_orig = b64_to_rl_image(gradcam_images.get("original_base64"))
        im_heat = b64_to_rl_image(gradcam_images.get("heatmap_base64"))
        im_over = b64_to_rl_image(gradcam_images.get("overlay_base64"))

        if im_orig and im_heat and im_over:
            cam_table_data = [
                [Paragraph("<b>Original X-Ray</b>", bold_body), Paragraph("<b>Activation Heatmap</b>", bold_body), Paragraph("<b>Diagnostic Overlay</b>", bold_body)],
                [im_orig, im_heat, im_over]
            ]
            cam_table = Table(cam_table_data, colWidths=[180, 180, 180])
            cam_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
            ]))
            story.append(cam_table)
        story.append(Spacer(1, 10))

    # 6. Research Disclaimer & Limitations
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#CBD5E1"), spaceBefore=6, spaceAfter=8))
    story.append(Paragraph("<b>RESEARCH USE DISCLAIMER:</b> RespiraSeverity is an academic research software prototype engineered for methodology demonstration and final-year dissertation studies. The predictions, probabilities, Grad-CAM heatmaps, and exploratory cluster assignments generated by this framework have NOT been cleared by regulatory health authorities and must NEVER be used as a substitute for professional clinical diagnosis or patient management decisions.", disclaimer_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
