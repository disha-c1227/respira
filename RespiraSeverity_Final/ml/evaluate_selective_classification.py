import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

PROJECT_ROOT = Path(r"C:\Users\Disha C\OneDrive\Desktop\RespiraSeverity_Final\RespiraSeverity_Final")
MANIFEST = PROJECT_ROOT / "data" / "manifest.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "disease_model.pt"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

C = ["Normal", "Pneumonia", "TB", "COVID"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EvalDS(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.transform = transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.path).convert("RGB")
        return self.transform(img), C.index(row.disease)

def run_evaluation():
    print("=" * 70)
    print("RespiraSeverity: Selective Classification & Uncertainty Threshold Analysis")
    print("=" * 70)
    
    df = pd.read_csv(MANIFEST)
    df = df[df.disease.isin(C)].copy()
    _, holdout = train_test_split(df, test_size=0.20, random_state=42, stratify=df.disease)
    
    loader = DataLoader(EvalDS(holdout), batch_size=32, shuffle=False, num_workers=0)
    
    # Load model
    m = models.mobilenet_v3_small(weights=None)
    m.classifier[-1] = torch.nn.Linear(m.classifier[-1].in_features, 4)
    ck = torch.load(MODEL_PATH, map_location=DEVICE)
    m.load_state_dict(ck["state_dict"])
    m.to(DEVICE).eval()
    
    all_targets = []
    all_probs = []
    all_preds = []
    all_confs = []
    
    print(f"Running inference over {len(holdout)} holdout radiographs...")
    with torch.no_grad():
        for x, y in loader:
            logits = m(x.to(DEVICE))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            confs = np.max(probs, axis=1)
            preds = np.argmax(probs, axis=1)
            
            all_targets.extend(y.numpy())
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_confs.extend(confs)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    confs = np.array(all_confs)
    probs = np.array(all_probs)
    
    # Overall baseline metrics
    base_acc = accuracy_score(y_true, y_pred)
    print(f"\nBaseline Accuracy (No Rejection): {base_acc * 100:.2f}%")
    
    thresholds = [0.0, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    results_by_thresh = []
    
    print("\n--- THRESHOLD VS COVERAGE & ACCURACY ---")
    print(f"{'Threshold (tau)':16s} | {'Coverage':10s} | {'Retained N':10s} | {'Selective Acc':14s} | {'Macro F1':10s} | {'Rejection %':12s}")
    print("-" * 85)
    
    for tau in thresholds:
        accepted_mask = confs >= tau
        retained_n = int(np.sum(accepted_mask))
        coverage = retained_n / len(y_true)
        rejection_rate = 1.0 - coverage
        
        if retained_n > 0:
            sel_acc = accuracy_score(y_true[accepted_mask], y_pred[accepted_mask])
            p, r, f1, _ = precision_recall_fscore_support(
                y_true[accepted_mask], y_pred[accepted_mask], average="macro", zero_division=0
            )
        else:
            sel_acc, f1 = 0.0, 0.0
            
        results_by_thresh.append({
            "threshold": tau,
            "coverage_percent": round(coverage * 100, 2),
            "retained_count": retained_n,
            "selective_accuracy": round(sel_acc * 100, 2),
            "macro_f1": round(f1 * 100, 2),
            "rejection_percent": round(rejection_rate * 100, 2)
        })
        
        print(f"{tau:<16.2f} | {coverage*100:9.2f}% | {retained_n:<10d} | {sel_acc*100:13.2f}% | {f1*100:9.2f}% | {rejection_rate*100:11.2f}%")
        
    # Save results to JSON
    out_file = REPORTS_DIR / "selective_classification_metrics.json"
    with open(out_file, "w") as f:
        json.dump(results_by_thresh, f, indent=2)
    print(f"\nSaved threshold analysis to {out_file}")

if __name__ == "__main__":
    run_evaluation()
