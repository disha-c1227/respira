# RespiraSeverity: A Deep Learning-Based Framework for Pulmonary Disease Classification and Severity Assessment Using Chest X-Ray Images

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch%202.14-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![React](https://img.shields.io/badge/Frontend-React%2018%20%7C%20Vite-61DAFB.svg?logo=react&logoColor=black)](https://react.dev)
[![Tailwind CSS](https://img.shields.io/badge/UI-Tailwind%20CSS-38B2AC.svg?logo=tailwind-css&logoColor=white)](https://tailwindcss.com)
[![Status](https://img.shields.io/badge/Research%20Prototype-Active%20v2.0-0F2942.svg)]()

> **Final-Year Academic Research Project & Dissertation**  
> **Institution:** Department of Computer Science & Engineering, Vidyavardhaka College of Engineering, Mysuru, Karnataka, India  
> **Faculty Guide:** Prof. Harshith C (Assistant Professor)  
> **Project Investigators:** Disha C (`dishakashyap1927@gmail.com`), P R Sri Nischal (`srinischalpr@gmail.com`), Ranjith K S (`ranjithks825@gmail.com`)

---

## 1. Project Overview & Problem Statement

Chest radiography is the primary, most widely accessible diagnostic modality for evaluating thoracic infections. While recent deep learning literature has produced numerous binary or multi-label classifiers, the vast majority stop at naming the disease: **they answer whether an infection is present, but fail to quantify how far the disease has progressed.** 

Patients sharing the same diagnosis often present with drastically different degrees of parenchymal consolidation, cavitation, or viral ground-glass infiltration. 

**RespiraSeverity** bridges this gap through a staged, hierarchical deep learning framework:
1. **Primary Four-Class Disease Classification:** Differentiates Normal radiographs from Pneumonia, Tuberculosis, and COVID-19 using a transfer-learned MobileNetV3-Small backbone.
2. **Intelligent Staged Severity Routing:** Avoids enforcing an arbitrary common severity scale across distinct pathologies. Instead:
   - **Normal cases:** Severity grading is bypassed.
   - **COVID-19 cases:** Routed to a **supervised 3-tier severity classifier** (Mild, Moderate, Severe) trained on 760 clinically annotated cases.
   - **Pneumonia & Tuberculosis cases:** Routed to an **exploratory 1024-D feature clustering module** (K-Means, $k=3$) and evaluated against COVID-19 reference centroids to assess radiographic burden without claiming unverified clinical labels.
3. **Transparent Explainability:** Integrated **Grad-CAM** visual attribution mapping reveals the high-activation thoracic regions that conditioned the network's decisions.
4. **Technical Quality Gate:** Automated photometric, focus (Laplacian variance), and resolution validation before inference.
5. **Downloadable Clinical Research PDF Reports:** Compiles formal medical imaging summaries with embedded activation maps.

---

## 2. Research Paper Performance vs. Baseline Results

The baseline disease classifier was evaluated on a stratified 80:20 holdout partition of **28,434** radiographs (5,687 holdout test images):

### Primary Disease Classification Metrics
- **Overall Accuracy:** **80.62%**
- **Macro Average F1-Score:** **84.39%**
- **Weighted F1-Score:** **80.50%**
- **Architecture Backbone:** MobileNetV3-Small (Transfer Learning from ImageNet)

| Disease Class | Precision | Recall | F1-Score | Support (Holdout Images) |
| :--- | :---: | :---: | :---: | :---: |
| **Normal** | 66.35% | 81.24% | 73.05% | 1,818 |
| **Pneumonia** | 79.05% | 99.74% | 88.20% | 1,165 |
| **Tuberculosis (TB)** | 97.66% | 64.61% | 77.77% | 2,128 |
| **COVID-19** | 97.94% | 99.13% | 98.53% | 576 |

### Disease Holdout Confusion Matrix
| Actual \ Predicted | Normal | Pneumonia | Tuberculosis | COVID-19 |
| :--- | :---: | :---: | :---: | :---: |
| **Normal** | 1,477 | 307 | 32 | 2 |
| **Pneumonia** | 3 | 1,162 | 0 | 0 |
| **Tuberculosis** | 743 | 0 | 1,375 | 10 |
| **COVID-19** | 3 | 1 | 1 | 571 |

### Supervised COVID-19 Severity Staging (Held-Out Test Set)
- **Cohort Size:** 760 cases (Mild: 234, Moderate: 284, Severe: 242)
- **Held-Out Test Accuracy:** **54.39%** (Macro Precision: 56.72%, Macro F1: 50.93%)
- **Test Confusion Matrix (Mild/Mod/Sev):**
  - Mild (35): 21 Mild, 5 Moderate, 9 Severe
  - Moderate (43): 9 Mild, 7 Moderate, 27 Severe
  - Severe (36): 1 Mild, 1 Moderate, 34 Severe

### Exploratory Feature Clustering (Pneumonia & TB)
- **Pneumonia ($n=1,500$):** Silhouette: **0.2123** | Davies-Bouldin: **1.7853** | Calinski-Harabasz: **382.84**
  - Cluster 0: 509 cases (48.6% mean opacity)
  - Cluster 1: 621 cases (48.4% mean opacity)
  - Cluster 2: 370 cases (48.1% mean opacity)
  - COVID Centroid Proximity: Mild: 933, Moderate: 563, Severe: 4
- **Tuberculosis ($n=1,500$):** Silhouette: **0.2016** | Davies-Bouldin: **1.9419** | Calinski-Harabasz: **291.50**
  - Cluster 0: 574 cases (49.9% mean opacity)
  - Cluster 1: 595 cases (51.0% mean opacity)
  - Cluster 2: 331 cases (57.1% mean opacity)
  - COVID Centroid Proximity: Mild: 1497, Moderate: 3, Severe: 0

---

## 3. System Architecture & Processing Pipeline

```
Chest X-Ray Input (PA/AP view, JPG/PNG)
                    │
                    ▼
       [1. Technical Quality Gate]
    - Format, blur (Laplacian variance),
      contrast std, brightness bounds
                    │
                    ▼
       [2. Standard Preprocessing]
    - Resize to 224 × 224, Grayscale(3),
      ImageNet normalization
                    │
                    ▼
    [3. Four-Class Disease Classifier]
         (MobileNetV3-Small CNN)
                    │
       ┌────────────┼─────────────┬─────────────┐
       ▼            ▼             ▼             ▼
    Normal      Pneumonia    Tuberculosis    COVID-19
  (Detected)   (Detected)     (Detected)    (Detected)
       │            │             │             │
  [No Severity      │             │             ▼
    Bypassed]       │             │   [Supervised Severity]
                    │             │    MobileNetV3-Small
                    │             │    (Mild/Moderate/Severe)
                    └──────┬──────┘
                           ▼
            [Exploratory K-Means Clustering]
             - 1024-D MobileNetV3 Embeddings
             - Feature Standardization
             - 3 Unsupervised Radiographic Groups
             - Nearest COVID Reference Centroid
                           │
                           ▼
          [Grad-CAM Visual Explainability]
           Attribution on features[12] layer
                           │
                           ▼
           [Healthcare Research Dashboard]
             PDF Download & Live Results
```

---

## 4. Repository Structure

```
RespiraSeverity_Final/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI application entrypoint & middleware
│   │   ├── routes/
│   │   │   ├── validation.py        # POST /api/validate-image
│   │   │   ├── prediction.py        # POST /api/predict, /api/predict/disease, /api/predict/*
│   │   │   ├── explain.py           # POST /api/explain/gradcam
│   │   │   ├── metrics.py           # GET /api/metrics/*, GET /api/model-info
│   │   │   └── report.py            # POST /api/report/generate (PDF byte stream)
│   │   ├── services/
│   │   │   ├── ml_service.py        # Singleton model manager & inference engine
│   │   │   ├── image_validator.py   # Photometric, blur & resolution validator
│   │   │   ├── gradcam_service.py   # PyTorch Grad-CAM hook manager & overlay generator
│   │   │   └── report_service.py    # ReportLab academic PDF generator
│   │   └── schemas/
│   │       └── models.py            # Pydantic input/output schemas
│   └── main.py                      # Root uvicorn ASGI entrypoint
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.jsx           # Top header navigation & live backend status pill
│   │   │   └── Footer.jsx           # Institutional academic affiliations & disclaimer
│   │   ├── pages/
│   │   │   ├── LandingPage.jsx      # Page 1: Hero & methodology overview
│   │   │   ├── DashboardPage.jsx    # Page 2: KPI metrics, model status & dataset charts
│   │   │   ├── AnalysisPage.jsx     # Page 3: Radiograph upload, demo samples & quality gate
│   │   │   ├── ResultsPage.jsx      # Page 4: Diagnosis & condition-specific severity staging
│   │   │   ├── ExplainabilityPage.jsx# Page 5: Grad-CAM overlay & heatmaps
│   │   │   ├── ClusteringPage.jsx   # Page 6: 2D PCA feature space & cluster benchmarks
│   │   │   ├── PerformancePage.jsx  # Page 7: Holdout confusion matrices & loss curves
│   │   │   └── ReportPage.jsx       # Page 8: PDF report preview & download
│   │   ├── services/
│   │   │   └── api.js               # REST client
│   │   ├── App.jsx                  # Main application state router
│   │   ├── main.jsx                 # Vite React mounting
│   │   └── style.css                # Tailwind CSS & custom scrollbars
│   ├── public/
│   │   └── samples/                 # Verified demo radiographs for one-click review
│   ├── package.json
│   ├── tailwind.config.js
│   └── vite.config.js
│
├── models/
│   ├── disease_model.pt             # Trained PyTorch 4-class classifier (80.62%)
│   ├── disease_model_baseline_80_62.pt # Preserved immutable baseline
│   ├── covid_severity_model.pt      # Trained PyTorch COVID severity model
│   ├── pneumonia_kmeans.pkl         # K-Means model for Pneumonia (k=3)
│   ├── tb_kmeans.pkl                # K-Means model for Tuberculosis (k=3)
│   ├── feature_scaler.pkl           # StandardScaler for 1024-D embeddings
│   ├── pca_transform.pkl            # Fitted PCA (2 components)
│   ├── covid_reference_centroids.npy# Anchors for Mild, Moderate, Severe
│   └── pca_visualization_sample.json# Interactive 2D scatter coordinates
│
├── data/
│   ├── manifest.csv                 # 28,434 validated image paths and metadata
│   ├── raw/
│   │   └── dataset_locations.json   # Pointer to KaggleHub cache
│   └── severity_splits/
│       ├── train.csv                # 532 cases
│       ├── val.csv                  # 114 cases
│       └── test.csv                 # 114 cases
│
├── ml/
│   ├── run_clustering.py            # Feature extraction & clustering pipeline
│   ├── evaluate_disease.py          # Disease holdout evaluator
│   ├── evaluate_severity.py         # COVID severity test set evaluator
│   ├── train_disease.py             # Disease training script
│   └── train_severity.py            # COVID severity training script
│
├── reports/
│   ├── disease_classification_report.txt # Baseline holdout evaluation text
│   └── clustering_metrics.json      # Silhouette, Davies-Bouldin, and Calinski scores
│
├── requirements.txt
└── README.md
```

---

## 5. Installation & Execution Guide (Windows / VS Code)

### Prerequisites
- **Python 3.10+** (Python 3.13 verified)
- **Node.js v18+** (v22 verified) & **npm**

### Step 1: Start the FastAPI Backend
Open a PowerShell terminal in VS Code:
```powershell
# Navigate to the project root
cd "C:\Users\Disha C\OneDrive\Desktop\RespiraSeverity_Final\RespiraSeverity_Final"

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Launch the FastAPI backend server on port 8000
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
*The backend will initialize MobileNetV3-Small, load all PyTorch weights, K-means models, and reference centroids. You will see `Application startup complete`.*

### Step 2: Start the React Frontend
Open a **second** PowerShell terminal in VS Code:
```powershell
# Navigate to frontend folder
cd "C:\Users\Disha C\OneDrive\Desktop\RespiraSeverity_Final\RespiraSeverity_Final\frontend"

# Launch the Vite development server
npm run dev
```
*Open your browser and navigate to:* **`http://localhost:5173`** (or `http://127.0.0.1:5173`).

---

## 6. Project Demonstration Walkthrough for Academic Review

When presenting this project to your faculty review committee:

1. **Step 1 — Landing & Overview (`Home` tab):**
   - Introduce the project title and research motivation: moving beyond binary detection to disease-specific severity staging.
   - Show the 4-stage architecture flowchart.
2. **Step 2 — Research Analytics Dashboard (`Dashboard` tab):**
   - Highlight the dataset size: **28,434** validated radiographs across 3 repositories (Chest-Xray, BIMCV, RICORD).
   - Point out the active model indicators and the 80.62% baseline accuracy.
3. **Step 3 — Radiograph Analysis Portal (`X-Ray Analysis` tab):**
   - Click one of the **Quick Project Demo Samples** (e.g. *COVID-19 Sample*, *Pneumonia Sample*, *Normal Sample*, or *Tuberculosis Sample*).
   - Show the **Technical Quality Assessment** box: resolution, format, Laplacian blur variance, and photometric contrast.
   - Click **Run RespiraSeverity Pipeline** to demonstrate the live prediction.
4. **Step 4 — Intelligent Routing & Severity Staging (`Results` tab):**
   - Show how the disease prediction routes:
     - For **COVID-19**: Triggers the supervised 3-tier severity classifier (Mild/Moderate/Severe) with probability bars.
     - For **Normal**: Explicitly bypasses severity staging.
     - For **Pneumonia / TB**: Triggers exploratory clustering, displays the assigned cluster, relative radiographic opacity, and proximity to COVID reference centroids.
     - Emphasize the **Scientific Ethics Disclaimer** explaining why exploratory clusters are not falsely claimed as clinical severity.
5. **Step 5 — Model Visual Explainability (`Grad-CAM` tab):**
   - Toggle between **Original Radiograph**, **Activation Heatmap**, and **Diagnostic Overlay** (or view Side-by-Side).
   - Explain that gradients are backpropagated into MobileNetV3-Small's `features[12]` inverted residual block to localize predictive thoracic patterns.
6. **Step 6 — Unsupervised Feature Clustering (`Clustering` tab):**
   - Demonstrate the interactive **2D PCA feature scatter plots** for Pneumonia, TB, and COVID reference centroids.
   - Review the **Comparative Proximity** tab showing the severe domain divergence when mapping TB to COVID centroids, validating our paper's hypothesis.
7. **Step 7 — Research Evaluation Metrics (`Model Metrics` tab):**
   - Display the holdout evaluation tables, confusion matrices, and training/validation loss curves matching Table II & Table III from the paper.
8. **Step 8 — Clinical Research PDF Report (`PDF Report` tab):**
   - Click **Download PDF Report**. Show the generated, professionally formatted multi-page PDF document containing tables, metadata, and embedded Grad-CAM visuals.

---

## 7. API Documentation Summary

The FastAPI backend exposes interactive OpenAPI docs at **`http://127.0.0.1:8000/docs`**:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | System health check, device status (CPU/CUDA), and loaded models |
| `GET` | `/api/model-info` | Project metadata, architectures, class lists, and dataset sizes |
| `POST` | `/api/validate-image` | Evaluates file format, corrupt bytes, resolution, blur, and contrast |
| `POST` | `/api/predict` | Unified end-to-end endpoint: validation &rarr; disease &rarr; routed severity &rarr; Grad-CAM |
| `POST` | `/api/predict/disease` | 4-class disease classification (Normal, Pneumonia, TB, COVID) |
| `POST` | `/api/predict/covid-severity` | Supervised 3-class COVID severity classification (Mild, Moderate, Severe) |
| `POST` | `/api/predict/pneumonia-cluster` | Assigns Pneumonia radiograph to exploratory cluster ($k=3$) and COVID reference distance |
| `POST` | `/api/predict/tb-cluster` | Assigns TB radiograph to exploratory cluster ($k=3$) and COVID reference distance |
| `POST` | `/api/explain/gradcam` | Returns base64 original, heatmap, and overlay images from `features[12]` |
| `GET` | `/api/metrics/disease` | Holdout accuracy, macro F1, per-class report, and confusion matrix |
| `GET` | `/api/metrics/severity` | Test set accuracy, macro precision/recall/F1, and confusion matrix |
| `GET` | `/api/metrics/clustering` | Silhouette, Davies-Bouldin, Calinski-Harabasz scores & PCA coordinates |
| `POST` | `/api/report/generate` | Generates downloadable ReportLab PDF report stream |

---

## 8. Research Limitations & Future Work

1. **Class Imbalance in Multi-Source Collections:** Normal and TB cases significantly outnumber COVID-19 cases in public datasets. Balanced weighted cross-entropy mitigates this, but external multi-center validation is recommended.
2. **Severity Annotation Scarcity:** Supervised severity labels are currently available only for COVID-19 (760 cases). As clinical collaborators provide annotated opacity or lung involvement scores for Pneumonia and TB, supervised heads can be attached modularly without retraining the primary disease classifier.
3. **Auditing via Explainability:** Grad-CAM maps highlight regions driving network activations; they should be treated as visual auditing aids, not as definitive radiologist lesion contours.

---

## 9. Academic Disclaimer

*RespiraSeverity is an academic research software prototype engineered strictly for methodology demonstration and final-year dissertation studies. The predictions, probabilities, Grad-CAM heatmaps, and exploratory cluster assignments generated by this framework have **not** been cleared by regulatory health authorities and must **never** be used for clinical diagnosis, patient triage, or direct medical treatment.*
