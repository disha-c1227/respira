"""
RespiraSeverity — Multi-Model Evaluation Script
Evaluates whichever of the four models are available and writes:
  reports/model_comparison.json        — structured metrics for all models
  reports/model_comparison_report.txt  — human-readable comparison report

Models evaluated (skipped if checkpoint not found):
  1. MobileNetV3-Small  — models/disease_model.pt           (original baseline)
  2. EfficientNetB0     — models/efficientnetb0_disease.pt
  3. ResNet50           — models/resnet50_disease.pt
  4. DenseNet121        — models/densenet121_disease.pt

All models use the same holdout split:
  train_test_split(df, test_size=0.20, random_state=42, stratify=df.disease)
  → 5,687 holdout images

IMPORTANT: This script never overwrites any model checkpoint.
           The COVID-19 severity model is not touched.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score
)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
MANIFEST   = ROOT / "data" / "manifest.csv"
MODELS_DIR = ROOT / "models"
REPORTS    = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

CLASSES      = ["Normal", "Pneumonia", "TB", "COVID"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
SEED         = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH        = 32


# ── Per-architecture preprocessing ───────────────────────────────────────────
# MobileNetV3 uses grayscale-repeated normalisation (must NOT change)
MOBILENET_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.485, 0.485],
                         std=[0.229, 0.229, 0.229]),
])

# EfficientNetB0, ResNet50, DenseNet121 all use ImageNet RGB
IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Dataset ───────────────────────────────────────────────────────────────────
class HoldoutDS(Dataset):
    def __init__(self, df, transform):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = CLASS_TO_IDX[row["disease"]]
        img   = Image.open(str(row["path"])).convert("RGB")
        return self.transform(img), label


# ── Model builders ────────────────────────────────────────────────────────────
def load_mobilenetv3(path):
    m = models.mobilenet_v3_small(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 4)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["state_dict"])
    return m


def load_efficientnetb0(path):
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 4)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["state_dict"])
    return m


def load_resnet50(path):
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 4)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["state_dict"])
    return m


def load_densenet121(path):
    m = models.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, 4)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["state_dict"])
    return m


# ── Registry ──────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "MobileNetV3-Small": {
        "path":        MODELS_DIR / "disease_model.pt",
        "loader":      load_mobilenetv3,
        "transform":   MOBILENET_TRANSFORM,
        "preprocessing": "Grayscale-repeat + [0.485,0.485,0.485]/[0.229,0.229,0.229]",
        "params_m":    2.5,
    },
    "EfficientNetB0": {
        "path":        MODELS_DIR / "efficientnetb0_disease.pt",
        "loader":      load_efficientnetb0,
        "transform":   IMAGENET_TRANSFORM,
        "preprocessing": "ImageNet RGB [0.485,0.456,0.406]/[0.229,0.224,0.225]",
        "params_m":    5.3,
    },
    "ResNet50": {
        "path":        MODELS_DIR / "resnet50_disease.pt",
        "loader":      load_resnet50,
        "transform":   IMAGENET_TRANSFORM,
        "preprocessing": "ImageNet RGB [0.485,0.456,0.406]/[0.229,0.224,0.225]",
        "params_m":    25.6,
    },
    "DenseNet121": {
        "path":        MODELS_DIR / "densenet121_disease.pt",
        "loader":      load_densenet121,
        "transform":   IMAGENET_TRANSFORM,
        "preprocessing": "ImageNet RGB [0.485,0.456,0.406]/[0.229,0.224,0.225]",
        "params_m":    8.1,
    },
}


# ── Evaluation function ───────────────────────────────────────────────────────
def evaluate_model(model_name, cfg, holdout_df):
    path = cfg["path"]
    if not path.exists():
        print(f"  [{model_name}] Checkpoint not found: {path.name} — SKIPPED")
        return None

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    # Load model
    t_load = time.time()
    model = cfg["loader"](path)
    model.to(DEVICE).eval()
    load_ms = (time.time() - t_load) * 1000
    print(f"  Loaded in {load_ms:.0f} ms")

    # Build loader
    ds     = HoldoutDS(holdout_df, cfg["transform"])
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)

    # Inference
    all_labels = []
    all_preds  = []
    all_probs  = []

    t_inf = time.time()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = probs.argmax(axis=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    total_inf_ms  = (time.time() - t_inf) * 1000
    per_img_ms    = total_inf_ms / len(holdout_df)

    y_true  = np.array(all_labels)
    y_pred  = np.array(all_preds)
    y_probs = np.array(all_probs)

    # Overall metrics
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # ROC-AUC (one-vs-rest)
    try:
        roc_auc = roc_auc_score(
            y_true, y_probs, multi_class="ovr", average="macro"
        )
    except Exception:
        roc_auc = None

    # Per-class
    per_class_p, per_class_r, per_class_f1, per_class_sup = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(4)), zero_division=0
    )

    per_class = {}
    for i, cls in enumerate(CLASSES):
        per_class[cls] = {
            "precision": round(float(per_class_p[i]) * 100, 2),
            "recall":    round(float(per_class_r[i]) * 100, 2),
            "f1":        round(float(per_class_f1[i]) * 100, 2),
            "support":   int(per_class_sup[i]),
        }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(4))).tolist()

    # Print results
    print(f"  Accuracy        : {acc*100:.2f}%")
    print(f"  Macro Precision : {prec_macro*100:.2f}%")
    print(f"  Macro Recall    : {rec_macro*100:.2f}%")
    print(f"  Macro F1        : {f1_macro*100:.2f}%")
    if roc_auc:
        print(f"  Macro ROC-AUC   : {roc_auc:.4f}")
    print(f"  Inference time  : {per_img_ms:.2f} ms/image")
    print()
    print("  Per-class performance:")
    print(f"  {'Class':<12} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
    print("  " + "-" * 50)
    for cls in CLASSES:
        pc = per_class[cls]
        print(f"  {cls:<12} {pc['precision']:>7.2f}% {pc['recall']:>7.2f}% {pc['f1']:>7.2f}% {pc['support']:>8}")
    print()
    print("  Confusion matrix (rows=true, cols=predicted):")
    print(f"  {'':<12}", end="")
    for c in CLASSES:
        print(f" {c:>12}", end="")
    print()
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:<12}", end="")
        for j in range(4):
            print(f" {cm[i][j]:>12}", end="")
        print()

    result = {
        "model":            model_name,
        "checkpoint":       str(path),
        "preprocessing":    cfg["preprocessing"],
        "params_million":   cfg["params_m"],
        "holdout_samples":  len(holdout_df),
        "accuracy":         round(acc * 100, 4),
        "macro_precision":  round(prec_macro * 100, 4),
        "macro_recall":     round(rec_macro * 100, 4),
        "macro_f1":         round(f1_macro * 100, 4),
        "weighted_f1":      round(f1_w * 100, 4),
        "roc_auc_macro":    round(roc_auc, 4) if roc_auc else None,
        "inference_ms_per_image": round(per_img_ms, 3),
        "per_class":        per_class,
        "confusion_matrix": {
            "labels": CLASSES,
            "matrix": cm,
        },
    }
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("RespiraSeverity — Multi-Model Comparison Evaluation")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Holdout split: test_size=0.20, random_state=42, stratified")
    print()

    if not MANIFEST.exists():
        raise SystemExit(f"Manifest not found: {MANIFEST}")

    df = pd.read_csv(MANIFEST)
    df = df[df["disease"].isin(CLASSES)].copy()

    _, holdout_df = train_test_split(
        df, test_size=0.20, random_state=SEED, stratify=df["disease"]
    )
    holdout_df = holdout_df.reset_index(drop=True)
    print(f"Holdout set: {len(holdout_df)} images")

    all_results = []
    for name, cfg in MODEL_REGISTRY.items():
        result = evaluate_model(name, cfg, holdout_df)
        if result:
            all_results.append(result)

    if not all_results:
        print("\nNo models were found. Train at least one model first.")
        return

    # ── Comparison table ──────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Model':<20} {'Accuracy':>9} {'MacroF1':>9} {'TB Recall':>10} {'COVID F1':>10} {'ms/img':>7}")
    print("-" * 70)
    for r in all_results:
        tb_recall = r["per_class"]["TB"]["recall"]
        covid_f1  = r["per_class"]["COVID"]["f1"]
        print(
            f"{r['model']:<20} {r['accuracy']:>8.2f}%"
            f" {r['macro_f1']:>8.2f}%"
            f" {tb_recall:>9.2f}%"
            f" {covid_f1:>9.2f}%"
            f" {r['inference_ms_per_image']:>7.2f}"
        )

    if len(all_results) > 1:
        best = max(all_results, key=lambda x: x["macro_f1"])
        print(f"\nBest model by Macro F1: {best['model']}  ({best['macro_f1']:.2f}%)")

    # ── Save JSON ─────────────────────────────────────────────────────────
    out_json = REPORTS / "model_comparison.json"
    output = {
        "evaluation_date":    pd.Timestamp.now().isoformat(),
        "holdout_samples":    len(holdout_df),
        "split":              "test_size=0.20, random_state=42, stratified by disease",
        "classes":            CLASSES,
        "preprocessing_note": (
            "MobileNetV3-Small uses Grayscale-repeat normalisation (consistent with its training). "
            "EfficientNetB0, ResNet50 and DenseNet121 use standard ImageNet RGB normalisation. "
            "Comparison against MobileNetV3 should note this preprocessing difference."
        ),
        "models": all_results,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nStructured results saved → {out_json}")

    # ── Save text report ──────────────────────────────────────────────────
    out_txt = REPORTS / "model_comparison_report.txt"
    lines   = []
    lines.append("RESPIRASEVERITY — MULTI-MODEL COMPARISON REPORT")
    lines.append(f"Generated: {pd.Timestamp.now()}")
    lines.append(f"Holdout  : {len(holdout_df)} images (20% stratified split, seed=42)")
    lines.append("")
    lines.append("PREPROCESSING NOTE")
    lines.append("  MobileNetV3-Small : Grayscale-repeat normalisation (preserved from original training)")
    lines.append("  EfficientNetB0    : ImageNet RGB normalisation")
    lines.append("  ResNet50          : ImageNet RGB normalisation")
    lines.append("  DenseNet121       : ImageNet RGB normalisation")
    lines.append("")
    for r in all_results:
        lines.append("=" * 60)
        lines.append(f"Model: {r['model']}")
        lines.append(f"  Accuracy        : {r['accuracy']:.2f}%")
        lines.append(f"  Macro Precision : {r['macro_precision']:.2f}%")
        lines.append(f"  Macro Recall    : {r['macro_recall']:.2f}%")
        lines.append(f"  Macro F1        : {r['macro_f1']:.2f}%")
        lines.append(f"  Weighted F1     : {r['weighted_f1']:.2f}%")
        if r["roc_auc_macro"]:
            lines.append(f"  ROC-AUC (macro) : {r['roc_auc_macro']:.4f}")
        lines.append(f"  Inference       : {r['inference_ms_per_image']:.2f} ms/image")
        lines.append("")
        lines.append("  Per-class:")
        for cls in CLASSES:
            pc = r["per_class"][cls]
            lines.append(
                f"    {cls:<12}  P={pc['precision']:>6.2f}%  R={pc['recall']:>6.2f}%"
                f"  F1={pc['f1']:>6.2f}%  n={pc['support']}"
            )
        lines.append("")
        lines.append("  Confusion matrix (rows=true, cols=predicted):")
        cm = r["confusion_matrix"]["matrix"]
        header = f"    {'':12}" + "".join(f"{c:>12}" for c in CLASSES)
        lines.append(header)
        for i, cls in enumerate(CLASSES):
            row_str = f"    {cls:<12}" + "".join(f"{cm[i][j]:>12}" for j in range(4))
            lines.append(row_str)
        lines.append("")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Text report saved        → {out_txt}")
    print()
    print("All done. The comparison data is now ready for backend/frontend integration.")


if __name__ == "__main__":
    main()
