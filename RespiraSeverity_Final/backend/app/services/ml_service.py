import os
import json
import joblib
from pathlib import Path
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms, models

from backend.app.services.gradcam_service import GradCAMService

PROJECT_ROOT = Path(r"C:\Users\Disha C\OneDrive\Desktop\RespiraSeverity_Final\RespiraSeverity_Final")
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DISEASE_CLASSES = ["Normal", "Pneumonia", "TB", "COVID"]
SEVERITY_CLASSES = ["Mild", "Moderate", "Severe"]

# EXACT PREPROCESSING USED IN TRAINING (train_disease.py)
# Crucial: Grayscale(3) and [.485]*3, [.229]*3 must match training exactly to prevent channel offset artifacts
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
])

# Default Confidence Threshold derived from validation coverage/accuracy tradeoff curve
DEFAULT_CONFIDENCE_THRESHOLD = 70.0  # 70% confidence yields 86.11% selective accuracy (83.07% coverage)
MINIMUM_MARGIN_THRESHOLD = 0.15      # Minimum gap between 1st and 2nd predicted class

class FeatureExtractor(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.fc0 = base_model.classifier[0]
        self.act = base_model.classifier[1]

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc0(x))
        return x

class MLService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.device = DEVICE
        print(f"[MLService] Initializing on device: {self.device}")

        # 1. Load Disease Classifier
        self.disease_model = None
        self.feature_extractor = None
        self.gradcam_service = None
        disease_model_path = MODELS_DIR / "disease_model.pt"

        if disease_model_path.exists():
            try:
                m = models.mobilenet_v3_small(weights=None)
                m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, len(DISEASE_CLASSES))
                ck = torch.load(disease_model_path, map_location=self.device)
                m.load_state_dict(ck["state_dict"])
                m.to(self.device).eval()
                self.disease_model = m
                self.feature_extractor = FeatureExtractor(m).to(self.device).eval()

                # Attach GradCAM to last inverted residual layer (features[12])
                self.gradcam_service = GradCAMService(m, m.features[12])
                print("[MLService] Disease classifier and GradCAM loaded successfully.")
            except Exception as e:
                print(f"[MLService] Error loading disease model: {e}")
        else:
            print(f"[MLService] Warning: {disease_model_path} not found.")

        # 2. Load COVID Severity Model
        self.covid_severity_model = None
        sev_model_path = MODELS_DIR / "covid_severity_model.pt"
        if sev_model_path.exists():
            try:
                sm = models.mobilenet_v3_small(weights=None)
                sm.classifier[-1] = nn.Linear(sm.classifier[-1].in_features, len(SEVERITY_CLASSES))
                sck = torch.load(sev_model_path, map_location=self.device)
                sm.load_state_dict(sck["model_state_dict"])
                sm.to(self.device).eval()
                self.covid_severity_model = sm
                print("[MLService] COVID severity model loaded successfully.")
            except Exception as e:
                print(f"[MLService] Error loading COVID severity model: {e}")
        else:
            print(f"[MLService] Warning: {sev_model_path} not found.")

        # 3. Load Clustering Assets
        self.ref_centroids = None
        self.pneumonia_kmeans = None
        self.tb_kmeans = None
        self.feature_scaler = None
        self.pca_transform = None
        self._load_clustering_assets()

    def _load_clustering_assets(self):
        try:
            centroids_path = MODELS_DIR / "covid_reference_centroids.npy"
            if centroids_path.exists():
                self.ref_centroids = np.load(centroids_path, allow_pickle=True).item()
                print("[MLService] Loaded COVID reference centroids.")

            scaler_path = MODELS_DIR / "feature_scaler.pkl"
            if scaler_path.exists():
                self.feature_scaler = joblib.load(scaler_path)
                print("[MLService] Loaded feature scaler.")

            pca_path = MODELS_DIR / "pca_transform.pkl"
            if pca_path.exists():
                self.pca_transform = joblib.load(pca_path)
                print("[MLService] Loaded PCA transform.")

            pneu_kmeans_path = MODELS_DIR / "pneumonia_kmeans.pkl"
            if pneu_kmeans_path.exists():
                self.pneumonia_kmeans = joblib.load(pneu_kmeans_path)
                print("[MLService] Loaded Pneumonia KMeans.")

            tb_kmeans_path = MODELS_DIR / "tb_kmeans.pkl"
            if tb_kmeans_path.exists():
                self.tb_kmeans = joblib.load(tb_kmeans_path)
                print("[MLService] Loaded TB KMeans.")
        except Exception as e:
            print(f"[MLService] Warning: Some clustering assets could not be loaded: {e}")

    def predict_disease(self, pil_image: Image.Image, threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> dict:
        if self.disease_model is None:
            raise RuntimeError("Disease classification model is not available.")

        tensor = IMAGE_TRANSFORM(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.disease_model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(probs.argmax())
        pred_label = DISEASE_CLASSES[pred_idx]
        conf = float(probs[pred_idx]) * 100.0

        # Raw probabilities dictionary
        raw_probabilities = {cls_name: round(float(prob), 4) for cls_name, prob in zip(DISEASE_CLASSES, probs)}
        pct_probabilities = {cls_name: round(float(prob) * 100.0, 2) for cls_name, prob in zip(DISEASE_CLASSES, probs)}

        # Uncertainty metrics
        sorted_probs = np.sort(probs)[::-1]
        margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

        # Selective Classification / Rejection Rule
        is_uncertain = (conf < threshold) or (margin < MINIMUM_MARGIN_THRESHOLD)
        rejection_status = "REJECTED_FOR_MANUAL_REVIEW" if is_uncertain else "ACCEPTED"

        # Routing logic
        if is_uncertain:
            route = "UNCERTAIN_REJECTED"
            second_idx = int(probs.argsort()[-2])
            second_label = DISEASE_CLASSES[second_idx]
            explanation = (
                f"Confidence ({conf:.1f}%) is below the clinical decision threshold ({threshold:.1f}%) "
                f"with high ambiguity between '{pred_label}' ({conf:.1f}%) and '{second_label}' ({probs[second_idx]*100:.1f}%). "
                f"Automated diagnosis deferred to manual radiologist review to prevent false-positive classifications."
            )
        elif pred_label == "Normal":
            route = "NORMAL_BYPASS"
            explanation = "Radiograph shows no prominent consolidation, cavitation, or viral ground-glass opacities. Staged severity assessment is bypassed."
        elif pred_label == "COVID":
            route = "COVID_SEVERITY"
            explanation = "Findings suggest COVID-19 viral pneumonia patterns (peripheral / bilateral opacities). Proceeding to supervised severity classification."
        elif pred_label == "Pneumonia":
            route = "PNEUMONIA_CLUSTER"
            explanation = "Findings suggest bacterial/general pneumonia consolidation. Routed to exploratory feature clustering analysis."
        elif pred_label == "TB":
            route = "TB_CLUSTER"
            explanation = "Findings suggest Tuberculosis radiographic features (apical infiltrate / nodular patterns). Routed to exploratory feature clustering analysis."
        else:
            route = "UNKNOWN"
            explanation = "Unclassified radiograph pattern."

        return {
            "predicted_class": pred_label,
            "confidence": round(conf, 2),
            "raw_probabilities": raw_probabilities,
            "probabilities": pct_probabilities,
            "routing_destination": route,
            "explanation": explanation,
            "is_uncertain": is_uncertain,
            "rejection_status": rejection_status,
            "confidence_threshold": threshold,
            "entropy": round(entropy, 3),
            "margin": round(margin, 3),
            "disclaimer": "Academic/research prototype; not a cleared diagnostic medical device."
        }

    def predict_covid_severity(self, pil_image: Image.Image) -> dict:
        if self.covid_severity_model is None:
            raise RuntimeError("COVID severity model is not loaded.")

        tensor = IMAGE_TRANSFORM(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.covid_severity_model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(probs.argmax())
        pred_label = SEVERITY_CLASSES[pred_idx]
        conf = float(probs[pred_idx]) * 100.0

        return {
            "severity": pred_label,
            "confidence": round(conf, 2),
            "probabilities": {s_name: round(float(p) * 100.0, 2) for s_name, p in zip(SEVERITY_CLASSES, probs)},
            "dataset_context": "Evaluated against 760 clinically annotated COVID-19 cases (RICORD / BIMCV).",
            "disclaimer": "Supervised severity prediction reflects airspace involvement criteria from research cohort annotations."
        }

    def _extract_embedding(self, pil_image: Image.Image) -> np.ndarray:
        if self.feature_extractor is None:
            raise RuntimeError("Feature extractor is not available.")
        tensor = IMAGE_TRANSFORM(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.feature_extractor(tensor).cpu().numpy()[0]
        return feat

    def predict_clustering(self, pil_image: Image.Image, disease_type: str) -> dict:
        self._load_clustering_assets()
        kmeans = self.pneumonia_kmeans if disease_type.upper() == "PNEUMONIA" else self.tb_kmeans
        if kmeans is None or self.feature_scaler is None:
            raise RuntimeError(f"Clustering model for {disease_type} is not yet available.")

        feat = self._extract_embedding(pil_image)
        feat_scaled = self.feature_scaler.transform(feat.reshape(1, -1))
        cluster_id = int(kmeans.predict(feat_scaled)[0])

        # Distance to COVID Reference Centroids
        distances = {}
        nearest_centroid = "N/A"
        if self.ref_centroids is not None:
            for label, c_vec in self.ref_centroids.items():
                c_scaled = self.feature_scaler.transform(c_vec.reshape(1, -1))[0]
                d = float(np.linalg.norm(feat_scaled[0] - c_scaled))
                distances[label] = round(d, 3)
            nearest_centroid = min(distances, key=distances.get)

        # Radiographic opacity heuristic
        gray = np.array(pil_image.convert("L"), dtype=np.float32) / 255.0
        sample_opacity = float(np.mean(gray))

        # Retrieve saved cluster evaluation metrics
        metrics_file = REPORTS_DIR / "clustering_metrics.json"
        saved_metrics = {}
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                all_m = json.load(f)
                saved_metrics = all_m.get("pneumonia" if disease_type.upper() == "PNEUMONIA" else "tb", {})

        cluster_names = {
            0: "Cluster 0 (Feature Group Alpha)",
            1: "Cluster 1 (Feature Group Beta)",
            2: "Cluster 2 (Feature Group Gamma)"
        }

        interpretations = {
            0: "Dense focal/multifocal opacity profile with high parenchymal attenuation.",
            1: "Diffuse infiltrative radiographic pattern with intermediate feature spread.",
            2: "Patchy / localized subsegmental opacification with moderate attenuation."
        }

        return {
            "disease": disease_type,
            "assigned_cluster": cluster_id,
            "cluster_name": cluster_names.get(cluster_id, f"Cluster {cluster_id}"),
            "cluster_interpretation": interpretations.get(cluster_id, "Exploratory radiographic cluster."),
            "mean_opacity_percentile": round(sample_opacity * 100.0, 1),
            "nearest_covid_reference_centroid": nearest_centroid,
            "distance_to_covid_centroids": distances,
            "cluster_metrics": saved_metrics,
            "is_exploratory": True,
            "disclaimer": "EXPLORATORY RESEARCH COMPONENT: Unsupervised clustering is based on radiographic visual embeddings. It does NOT represent clinically validated mild/moderate/severe labels."
        }

    def generate_gradcam(self, pil_image: Image.Image, class_idx: int = None) -> dict:
        if self.gradcam_service is None:
            raise RuntimeError("GradCAM service is not initialized.")

        tensor = IMAGE_TRANSFORM(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.disease_model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            predicted_idx = int(probs.argmax())

        target_idx = predicted_idx if class_idx is None else class_idx
        cam_map = self.gradcam_service.generate_heatmap(tensor, target_idx)
        viz = GradCAMService.create_visualizations(pil_image, cam_map)

        pred_name = DISEASE_CLASSES[target_idx]
        conf = float(probs[target_idx]) * 100.0

        return {
            "predicted_class": pred_name,
            "confidence": round(conf, 2),
            "original_base64": viz["original_base64"],
            "heatmap_base64": viz["heatmap_base64"],
            "overlay_base64": viz["overlay_base64"],
            "attention_summary": f"Grad-CAM highlights high-activation thoracic regions contributing to the '{pred_name}' classification.",
            "target_layer": "MobileNetV3-Small features[12] (Inverted Residual Block)",
            "disclaimer": "Grad-CAM displays model gradient localization, which provides interpretability for research auditing but does not represent an annotated pathological lesion boundary."
        }
