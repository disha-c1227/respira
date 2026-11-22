import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from backend.app.services.ml_service import MLService, PROJECT_ROOT, MODELS_DIR, REPORTS_DIR

router = APIRouter(prefix="/api", tags=["Metrics"])
ml_service = MLService()

@router.get("/metrics/disease")
def get_disease_metrics():
    """
    Returns holdout evaluation performance for the 4-class disease classification model
    as reported in the research paper.
    """
    return {
        "architecture": "MobileNetV3-Small (Transfer Learning)",
        "dataset_total": 28434,
        "holdout_samples": 5687,
        "accuracy": 80.62,
        "macro_f1": 84.39,
        "weighted_f1": 80.50,
        "macro_precision": 85.25,
        "macro_recall": 86.18,
        "per_class": {
            "Normal": {"precision": 66.35, "recall": 81.24, "f1": 73.05, "support": 1818},
            "Pneumonia": {"precision": 79.05, "recall": 99.74, "f1": 88.20, "support": 1165},
            "Tuberculosis": {"precision": 97.66, "recall": 64.61, "f1": 77.77, "support": 2128},
            "COVID-19": {"precision": 97.94, "recall": 99.13, "f1": 98.53, "support": 576}
        },
        "confusion_matrix": {
            "labels": ["Normal", "Pneumonia", "Tuberculosis", "COVID-19"],
            "matrix": [
                [1477, 307, 32, 2],
                [3, 1162, 0, 0],
                [743, 0, 1375, 10],
                [3, 1, 1, 571]
            ]
        },
        "training_history": {
            "epochs": [1, 2, 3, 4, 5, 6, 7, 8],
            "train_acc": [71.2, 75.8, 78.4, 80.1, 81.5, 82.3, 83.0, 83.7],
            "val_acc": [74.5, 77.2, 79.1, 80.62, 80.62, 80.5, 80.4, 80.62],
            "train_loss": [0.68, 0.55, 0.48, 0.42, 0.38, 0.35, 0.32, 0.30],
            "val_loss": [0.62, 0.54, 0.49, 0.45, 0.46, 0.47, 0.49, 0.46]
        }
    }

@router.get("/metrics/thresholds")
def get_threshold_metrics():
    """
    Returns selective classification uncertainty evaluation across confidence thresholds (tau)
    measured on the 5,687 holdout validation set.
    """
    thresh_file = REPORTS_DIR / "selective_classification_metrics.json"
    if thresh_file.exists():
        with open(thresh_file, "r") as f:
            data = json.load(f)
            return {
                "thresholds": data,
                "recommended_threshold": 0.70,
                "notes": "Thresholding at tau = 0.70 retains 83.07% coverage while elevating selective accuracy from 80.62% to 86.11% (Macro F1 89.74%)."
            }
    return {"thresholds": [], "recommended_threshold": 0.70}

@router.get("/metrics/severity")
def get_severity_metrics():
    """
    Returns evaluation performance for the supervised COVID-19 severity model
    evaluated on the held-out test split.
    """
    return {
        "architecture": "MobileNetV3-Small (Transfer Learning)",
        "classes": ["Mild", "Moderate", "Severe"],
        "dataset_total": 760,
        "splits": {"train": 532, "val": 114, "test": 114},
        "test_accuracy": 54.39,
        "macro_precision": 56.72,
        "macro_recall": 56.91,
        "macro_f1": 50.93,
        "per_class": {
            "Mild": {"precision": 67.74, "recall": 60.00, "f1": 63.64, "support": 35},
            "Moderate": {"precision": 53.85, "recall": 16.28, "f1": 25.00, "support": 43},
            "Severe": {"precision": 48.57, "recall": 94.44, "f1": 64.15, "support": 36}
        },
        "confusion_matrix": {
            "labels": ["Mild", "Moderate", "Severe"],
            "matrix": [
                [21, 5, 9],
                [9, 7, 27],
                [1, 1, 34]
            ]
        }
    }

@router.get("/metrics/clustering")
def get_clustering_metrics():
    """
    Returns exploratory clustering metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
    and 2D PCA projection scatter data for interactive visualization.
    """
    metrics_path = REPORTS_DIR / "clustering_metrics.json"
    pca_path = MODELS_DIR / "pca_visualization_sample.json"

    metrics_data = {}
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)

    pca_data = {}
    if pca_path.exists():
        with open(pca_path, "r") as f:
            pca_data = json.load(f)

    return {
        "summary": metrics_data,
        "pca_projections": pca_data,
        "disclaimer": "Exploratory research data. Unsupervised clusters do not represent validated clinical severity."
    }

@router.get("/model-info")
def get_model_info():
    """
    Returns comprehensive system architecture status and dataset statistics.
    """
    return {
        "project_title": "RespiraSeverity: A Deep Learning-Based Framework for Pulmonary Disease Classification and Severity Assessment Using Chest X-Ray Images",
        "authors": "Prof. Harshith C, Disha C, P R Sri Nischal, Ranjith K S",
        "institution": "Vidyavardhaka College of Engineering, Mysuru, Karnataka, India",
        "device": str(ml_service.device),
        "disease_model": {
            "status": "LOADED" if ml_service.disease_model is not None else "UNAVAILABLE",
            "backbone": "MobileNetV3-Small",
            "classes": ["Normal", "Pneumonia", "Tuberculosis", "COVID-19"],
            "reported_accuracy": 80.62,
            "macro_f1": 84.39,
            "rejection_threshold": 70.0
        },
        "covid_severity_model": {
            "status": "LOADED" if ml_service.covid_severity_model is not None else "UNAVAILABLE",
            "backbone": "MobileNetV3-Small",
            "classes": ["Mild", "Moderate", "Severe"],
            "test_accuracy": 54.39,
            "macro_f1": 50.93
        },
        "clustering_module": {
            "status": "READY" if (ml_service.pneumonia_kmeans is not None and ml_service.tb_kmeans is not None) else "PARTIAL",
            "feature_dim": 1024,
            "pneumonia_clusters": 3,
            "tb_clusters": 3,
            "covid_reference_centroids": "LOADED" if ml_service.ref_centroids is not None else "UNAVAILABLE"
        },
        "dataset_statistics": {
            "total_images": 28434,
            "distribution": {
                "Tuberculosis": 10641,
                "Normal": 9088,
                "Pneumonia": 5824,
                "COVID-19": 2881
            },
            "covid_severity_labelled_cases": 760,
            "sources": [
                {"name": "Chest X-Ray Dataset (muhammadrehan00)", "count": 25553},
                {"name": "BIMCV COVID-19 (rafiko1)", "count": 1859},
                {"name": "RICORD COVID-19 (raddar)", "count": 1022}
            ]
        }
    }
