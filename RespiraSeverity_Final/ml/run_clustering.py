import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

PROJECT_ROOT = Path(r"C:\Users\Disha C\OneDrive\Desktop\RespiraSeverity_Final\RespiraSeverity_Final")
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "disease_model.pt"
OUTPUT_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")

print("=" * 70)
print("RespiraSeverity: Feature Extraction & Clustering Module")
print("=" * 70)

# 1. Load Trained Disease Model as Feature Extractor
print("Loading trained MobileNetV3-Small backbone...")
model = models.mobilenet_v3_small(weights=None)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 4)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# Feature extractor: backbone features + avgpool + classifier[0] (Linear -> 1024)
class MobileNetFeatureExtractor(nn.Module):
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

feature_extractor = MobileNetFeatureExtractor(model)
feature_extractor.eval()

img_transform = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
])

def extract_features_from_paths(paths, batch_size=32, desc=""):
    embeddings = []
    opacities = []
    print(f"Extracting features for {len(paths)} {desc} images...")
    
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i+batch_size]
        batch_tensors = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                # Calculate mean grayscale pixel intensity as simple radiographic opacity metric
                gray = np.array(img.convert("L"), dtype=np.float32) / 255.0
                opacities.append(float(np.mean(gray)))
                batch_tensors.append(img_transform(img))
            except Exception as e:
                print(f"Error reading {p}: {e}")
        
        if batch_tensors:
            batch_x = torch.stack(batch_tensors)
            with torch.no_grad():
                feat = feature_extractor(batch_x).numpy()
                embeddings.append(feat)
        
        if (i // batch_size) % 10 == 0 or i + batch_size >= len(paths):
            sys.stdout.write(f"\rProcessed {min(i + batch_size, len(paths))}/{len(paths)} images")
            sys.stdout.flush()
    print()
    return np.concatenate(embeddings, axis=0), np.array(opacities)

# Load Manifest
df = pd.read_csv(MANIFEST_PATH)

# A. Extract COVID-19 Severity embeddings & Reference Centroids
print("\n--- STEP A & B: COVID-19 Reference Centroids ---")
covid_df = df[df["severity"].isin(["Mild", "Moderate", "Severe"])].copy()
covid_mild = covid_df[covid_df["severity"] == "Mild"]["path"].tolist()
covid_mod = covid_df[covid_df["severity"] == "Moderate"]["path"].tolist()
covid_sev = covid_df[covid_df["severity"] == "Severe"]["path"].tolist()

mild_feats, _ = extract_features_from_paths(covid_mild, desc="COVID Mild")
mod_feats, _ = extract_features_from_paths(covid_mod, desc="COVID Moderate")
sev_feats, _ = extract_features_from_paths(covid_sev, desc="COVID Severe")

centroid_mild = np.mean(mild_feats, axis=0)
centroid_mod = np.mean(mod_feats, axis=0)
centroid_sev = np.mean(sev_feats, axis=0)

reference_centroids = {
    "Mild": centroid_mild,
    "Moderate": centroid_mod,
    "Severe": centroid_sev
}
np.save(OUTPUT_DIR / "covid_reference_centroids.npy", reference_centroids)
print(f"Saved COVID reference centroids to {OUTPUT_DIR / 'covid_reference_centroids.npy'}")

# Combine all COVID features for scaling and PCA
all_covid_feats = np.concatenate([mild_feats, mod_feats, sev_feats], axis=0)
all_covid_labels = ["Mild"] * len(mild_feats) + ["Moderate"] * len(mod_feats) + ["Severe"] * len(sev_feats)

# B. Extract Pneumonia embeddings (sample 1,500 images for robust clustering)
print("\n--- STEP C: Pneumonia Clustering ---")
pneu_df = df[df["disease"] == "Pneumonia"].sample(n=min(1500, len(df[df["disease"] == "Pneumonia"])), random_state=42)
pneu_feats, pneu_opacities = extract_features_from_paths(pneu_df["path"].tolist(), desc="Pneumonia")

# C. Extract TB embeddings (sample 1,500 images for robust clustering)
print("\n--- STEP D: Tuberculosis Clustering ---")
tb_df = df[df["disease"] == "TB"].sample(n=min(1500, len(df[df["disease"] == "TB"])), random_state=42)
tb_feats, tb_opacities = extract_features_from_paths(tb_df["path"].tolist(), desc="Tuberculosis")

# D. Feature Scaler & PCA
print("\n--- Fitting Scaler and PCA ---")
scaler = StandardScaler()
combined_feats = np.concatenate([all_covid_feats, pneu_feats, tb_feats], axis=0)
scaler.fit(combined_feats)

joblib.dump(scaler, OUTPUT_DIR / "feature_scaler.pkl")
print(f"Saved feature scaler to {OUTPUT_DIR / 'feature_scaler.pkl'}")

pneu_scaled = scaler.transform(pneu_feats)
tb_scaled = scaler.transform(tb_feats)
covid_scaled = scaler.transform(all_covid_feats)

ref_centroids_scaled = {
    "Mild": scaler.transform(centroid_mild.reshape(1, -1))[0],
    "Moderate": scaler.transform(centroid_mod.reshape(1, -1))[0],
    "Severe": scaler.transform(centroid_sev.reshape(1, -1))[0]
}

pca = PCA(n_components=2, random_state=42)
pca.fit(scaler.transform(combined_feats))
joblib.dump(pca, OUTPUT_DIR / "pca_transform.pkl")
print(f"Saved PCA transform to {OUTPUT_DIR / 'pca_transform.pkl'}")

# E. Fit K-Means on Pneumonia
kmeans_pneu = KMeans(n_clusters=3, random_state=42, n_init=10)
pneu_clusters = kmeans_pneu.fit_predict(pneu_scaled)
joblib.dump(kmeans_pneu, OUTPUT_DIR / "pneumonia_kmeans.pkl")
print(f"Saved Pneumonia K-means to {OUTPUT_DIR / 'pneumonia_kmeans.pkl'}")

# Pneumonia metrics
pneu_sil = float(silhouette_score(pneu_scaled, pneu_clusters))
pneu_db = float(davies_bouldin_score(pneu_scaled, pneu_clusters))
pneu_ch = float(calinski_harabasz_score(pneu_scaled, pneu_clusters))

# Pneumonia cluster counts & mean opacity
pneu_counts = [int(np.sum(pneu_clusters == k)) for k in range(3)]
pneu_mean_opacities = [float(np.mean(pneu_opacities[pneu_clusters == k])) for k in range(3)]

# Assign Pneumonia to nearest COVID reference centroid
ref_matrix = np.array([ref_centroids_scaled["Mild"], ref_centroids_scaled["Moderate"], ref_centroids_scaled["Severe"]])
ref_labels = ["Mild", "Moderate", "Severe"]
dists_pneu = np.linalg.norm(pneu_scaled[:, None, :] - ref_matrix[None, :, :], axis=2)
pneu_ref_assigned = np.argmin(dists_pneu, axis=1)
pneu_ref_counts = {ref_labels[i]: int(np.sum(pneu_ref_assigned == i)) for i in range(3)}

# F. Fit K-Means on TB
kmeans_tb = KMeans(n_clusters=3, random_state=42, n_init=10)
tb_clusters = kmeans_tb.fit_predict(tb_scaled)
joblib.dump(kmeans_tb, OUTPUT_DIR / "tb_kmeans.pkl")
print(f"Saved TB K-means to {OUTPUT_DIR / 'tb_kmeans.pkl'}")

# TB metrics
tb_sil = float(silhouette_score(tb_scaled, tb_clusters))
tb_db = float(davies_bouldin_score(tb_scaled, tb_clusters))
tb_ch = float(calinski_harabasz_score(tb_scaled, tb_clusters))

tb_counts = [int(np.sum(tb_clusters == k)) for k in range(3)]
tb_mean_opacities = [float(np.mean(tb_opacities[tb_clusters == k])) for k in range(3)]

# Assign TB to nearest COVID reference centroid
dists_tb = np.linalg.norm(tb_scaled[:, None, :] - ref_matrix[None, :, :], axis=2)
tb_ref_assigned = np.argmin(dists_tb, axis=1)
tb_ref_counts = {ref_labels[i]: int(np.sum(tb_ref_assigned == i)) for i in range(3)}

# Transform PCA coordinates for dashboard visualization (subsampled for quick web chart loading)
pneu_pca = pca.transform(pneu_scaled)
tb_pca = pca.transform(tb_scaled)
covid_pca = pca.transform(covid_scaled)

pca_viz_data = {
    "covid": [
        {"x": round(float(pneu_pca[i, 0]), 3), "y": round(float(pneu_pca[i, 1]), 3), "label": all_covid_labels[i]}
        for i in range(min(300, len(all_covid_labels)))
    ],
    "pneumonia": [
        {"x": round(float(pneu_pca[i, 0]), 3), "y": round(float(pneu_pca[i, 1]), 3), "cluster": int(pneu_clusters[i]), "ref_assigned": ref_labels[pneu_ref_assigned[i]]}
        for i in range(min(300, len(pneu_clusters)))
    ],
    "tb": [
        {"x": round(float(tb_pca[i, 0]), 3), "y": round(float(tb_pca[i, 1]), 3), "cluster": int(tb_clusters[i]), "ref_assigned": ref_labels[tb_ref_assigned[i]]}
        for i in range(min(300, len(tb_clusters)))
    ],
    "ref_centroids_pca": {
        label: [round(float(v), 3) for v in pca.transform(ref_centroids_scaled[label].reshape(1, -1))[0]]
        for label in ref_labels
    }
}

with open(OUTPUT_DIR / "pca_visualization_sample.json", "w") as f:
    json.dump(pca_viz_data, f, indent=2)

# Save comprehensive clustering metrics summary
metrics_summary = {
    "pneumonia": {
        "sample_size": len(pneu_df),
        "silhouette_score": round(pneu_sil, 4),
        "davies_bouldin_index": round(pneu_db, 4),
        "calinski_harabasz_score": round(pneu_ch, 2),
        "cluster_distribution": {
            "Cluster 0": pneu_counts[0],
            "Cluster 1": pneu_counts[1],
            "Cluster 2": pneu_counts[2]
        },
        "mean_radiographic_opacities": {
            "Cluster 0": round(pneu_mean_opacities[0], 4),
            "Cluster 1": round(pneu_mean_opacities[1], 4),
            "Cluster 2": round(pneu_mean_opacities[2], 4)
        },
        "covid_ref_assignment": pneu_ref_counts
    },
    "tb": {
        "sample_size": len(tb_df),
        "silhouette_score": round(tb_sil, 4),
        "davies_bouldin_index": round(tb_db, 4),
        "calinski_harabasz_score": round(tb_ch, 2),
        "cluster_distribution": {
            "Cluster 0": tb_counts[0],
            "Cluster 1": tb_counts[1],
            "Cluster 2": tb_counts[2]
        },
        "mean_radiographic_opacities": {
            "Cluster 0": round(tb_mean_opacities[0], 4),
            "Cluster 1": round(tb_mean_opacities[1], 4),
            "Cluster 2": round(tb_mean_opacities[2], 4)
        },
        "covid_ref_assignment": tb_ref_counts
    },
    "covid_reference": {
        "total_cases": len(all_covid_labels),
        "mild_count": len(mild_feats),
        "moderate_count": len(mod_feats),
        "severe_count": len(sev_feats)
    }
}

with open(REPORTS_DIR / "clustering_metrics.json", "w") as f:
    json.dump(metrics_summary, f, indent=2)

print("\n" + "=" * 70)
print("CLUSTERING SUMMARY")
print("=" * 70)
print(json.dumps(metrics_summary, indent=2))
print(f"\nSaved metrics to {REPORTS_DIR / 'clustering_metrics.json'}")
print("Feature extraction and exploratory clustering completed successfully!")
