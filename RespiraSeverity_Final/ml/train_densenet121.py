"""
RespiraSeverity — DenseNet121 Disease Classifier
4 classes: Normal, Pneumonia, TB, COVID

CHECKPOINT STRATEGY
-------------------
1. EPOCH SNAPSHOT  models/densenet121_checkpoints/densenet121_epoch_N.pt
   Permanent per-epoch record. Never overwritten.

2. LATEST          models/densenet121_checkpoints/densenet121_latest.pt
   Overwritten every epoch. Used for resume after any interruption.
   Stores: weights, optimizer, scheduler, epoch, history, no_improve,
           best_val_loss.

3. BEST MODEL      models/densenet121_disease.pt
   Updated only when val_loss strictly improves. Later worse epochs
   never overwrite it. Contains weights + inference metadata only.

Max epochs : 8  |  Early-stop patience : 3 (val_loss)
Preprocessing: ImageNet RGB (Option B, architecture-native)
NEVER touches MobileNetV3, severity model, or any other existing file.
"""

import os, sys, time, random
from pathlib import Path

# ---------------------------------------------------------------------------
# Dual-output logger
# ---------------------------------------------------------------------------
ROOT_DIR  = Path(__file__).resolve().parents[1]
_LOG_PATH = ROOT_DIR / "densenet121_training.log"
_log_fh   = open(_LOG_PATH, "a", encoding="utf-8", buffering=1)

import builtins as _builtins
_real_print = _builtins.print

def _tee(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _real_print(*args, **kwargs)
    sep  = kwargs.get("sep", " ")
    end  = kwargs.get("end", "\n")
    line = sep.join(str(a) for a in args) + end
    if not line.startswith("\r"):
        _log_fh.write(line)
        _log_fh.flush()

_builtins.print = _tee

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parents[1]
MANIFEST   = ROOT / "data" / "manifest.csv"
MODELS_DIR = ROOT / "models"
CKPT_DIR   = MODELS_DIR / "densenet121_checkpoints"
BEST_MODEL = MODELS_DIR / "densenet121_disease.pt"
LATEST_CK  = CKPT_DIR   / "densenet121_latest.pt"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
CLASSES             = ["Normal", "Pneumonia", "TB", "COVID"]
IMG_SIZE            = 224
BATCH_SIZE          = 32
NUM_EPOCHS          = 8
EARLY_STOP_PATIENCE = 3
LR                  = 1e-4
WEIGHT_DECAY        = 1e-4
DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Preprocessing — ImageNet RGB
# ---------------------------------------------------------------------------
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class XrayDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(str(row["path"])).convert("RGB")
        return self.transform(img), CLASS_TO_IDX[row["disease"]]

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model():
    m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    m.classifier = nn.Linear(m.classifier.in_features, len(CLASSES))
    return m

# ---------------------------------------------------------------------------
# Dual checkpoint saving
# ---------------------------------------------------------------------------
def save_checkpoints(model, optimizer, scheduler, epoch,
                     tr_loss, tr_acc, va_loss, va_acc,
                     best_val_loss, no_improve, history, is_best):
    epoch_state = {
        "epoch"         : epoch,
        "state_dict"    : model.state_dict(),
        "optimizer"     : optimizer.state_dict(),
        "scheduler"     : scheduler.state_dict(),
        "tr_loss"       : tr_loss,
        "tr_acc"        : tr_acc,
        "va_loss"       : va_loss,
        "va_acc"        : va_acc,
        "best_val_loss" : best_val_loss,
        "no_improve"    : no_improve,
        "history"       : history,
        "classes"       : CLASSES,
        "architecture"  : "DenseNet121",
        "img_size"      : IMG_SIZE,
        "preprocessing" : "ImageNet-RGB",
        "num_epochs"    : NUM_EPOCHS,
    }

    # 1. Epoch snapshot (permanent)
    epoch_path = CKPT_DIR / f"densenet121_epoch_{epoch}.pt"
    torch.save(epoch_state, epoch_path)
    print(f"  [EPOCH CKPT]  saved -> {epoch_path.name}  "
          f"(val_loss={va_loss:.4f}  val_acc={va_acc:.4f})")

    # 2. Latest (always-current, overwritten each epoch)
    torch.save(epoch_state, LATEST_CK)
    print(f"  [LATEST CKPT] updated -> {LATEST_CK.name}")

    # 3. Best model (only on improvement)
    if is_best:
        best_state = {
            "state_dict"    : model.state_dict(),
            "classes"       : CLASSES,
            "architecture"  : "DenseNet121",
            "img_size"      : IMG_SIZE,
            "preprocessing" : "ImageNet-RGB",
            "best_val_loss" : best_val_loss,
            "best_val_acc"  : va_acc,
            "best_epoch"    : epoch,
        }
        torch.save(best_state, BEST_MODEL)
        print(f"  [BEST MODEL]  updated -> {BEST_MODEL.name}  "
              f"(new best val_loss={va_loss:.4f})")
    else:
        print(f"  [BEST MODEL]  unchanged (best val_loss still {best_val_loss:.4f})")

# ---------------------------------------------------------------------------
# Train / eval helpers
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    run_loss = run_correct = total = 0
    n = len(loader)
    t0 = time.time()
    for bi, (images, labels) in enumerate(loader, 1):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(images)
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        run_loss    += loss.item() * len(labels)
        run_correct += (out.argmax(1) == labels).sum().item()
        total       += len(labels)
        if bi % 50 == 0 or bi == n:
            eta = (n - bi) / max(bi / max(time.time() - t0, 1e-6), 1e-6) / 60
            sys.stdout.write(
                f"\r  Epoch {epoch}/{NUM_EPOCHS}  batch {bi}/{n}  "
                f"loss={run_loss/total:.4f}  acc={run_correct/total:.4f}  ETA={eta:.1f}min"
            )
            sys.stdout.flush()
            if bi % 200 == 0 or bi == n:
                _log_fh.write(
                    f"  batch {bi}/{n}  loss={run_loss/total:.4f}  acc={run_correct/total:.4f}\n"
                )
                _log_fh.flush()
    sys.stdout.write("\n"); sys.stdout.flush()
    return run_loss / total, run_correct / total

def run_validation(model, loader, criterion):
    model.eval()
    run_loss = run_correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            out  = model(images)
            loss = criterion(out, labels)
            run_loss    += loss.item() * len(labels)
            run_correct += (out.argmax(1) == labels).sum().item()
            total       += len(labels)
    return run_loss / total, run_correct / total

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("RespiraSeverity -- DenseNet121  (max 8 epochs)")
    print("=" * 70)
    print(f"Log              : {_LOG_PATH}")
    print(f"Device           : {DEVICE}")
    print(f"Batch size       : {BATCH_SIZE}")
    print(f"Max epochs       : {NUM_EPOCHS}")
    print(f"Early-stop       : patience={EARLY_STOP_PATIENCE} on val_loss")
    print(f"LR / wd          : {LR} / {WEIGHT_DECAY}")
    print(f"Preprocessing    : ImageNet RGB (Option B)")
    print()
    print("Checkpoint locations:")
    print(f"  Epoch snapshots : {CKPT_DIR}/densenet121_epoch_N.pt")
    print(f"  Latest (resume) : {LATEST_CK}")
    print(f"  Best model      : {BEST_MODEL}")
    print()

    if not MANIFEST.exists():
        raise SystemExit(f"Manifest not found: {MANIFEST}")

    df = pd.read_csv(MANIFEST)
    df = df[df["disease"].isin(CLASSES)].copy()
    train_df, holdout_df = train_test_split(
        df, test_size=0.20, random_state=SEED, stratify=df["disease"]
    )
    train_df = train_df.reset_index(drop=True)
    holdout_df = holdout_df.reset_index(drop=True)

    print(f"Training set : {len(train_df)}")
    print(f"Holdout set  : {len(holdout_df)}  (same 5,687 images as MobileNetV3 baseline)")
    print()
    for c in CLASSES:
        print(f"  {c:<12}: {(train_df['disease']==c).sum()}")
    print()

    train_loader = DataLoader(XrayDataset(train_df, TRAIN_TRANSFORM),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(DEVICE.type=="cuda"))
    val_loader   = DataLoader(XrayDataset(holdout_df, VAL_TRANSFORM),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=(DEVICE.type=="cuda"))

    weights = compute_class_weight("balanced", classes=np.array(CLASSES),
                                   y=train_df["disease"].values)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    )
    print(f"Class weights : { {c: round(float(w),4) for c,w in zip(CLASSES,weights)} }")
    print()

    model     = build_model().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    # ── Resume from latest checkpoint ──────────────────────────────────
    start_epoch   = 0
    best_val_loss = float("inf")
    no_improve    = 0
    history       = []

    if LATEST_CK.exists():
        try:
            ck = torch.load(LATEST_CK, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ck["state_dict"])
            optimizer.load_state_dict(ck["optimizer"])
            scheduler.load_state_dict(ck["scheduler"])
            start_epoch   = int(ck["epoch"])
            best_val_loss = float(ck.get("best_val_loss", float("inf")))
            no_improve    = int(ck.get("no_improve", 0))
            history       = ck.get("history", [])
            print(f"Resumed from {LATEST_CK.name}")
            print(f"  Completed epochs : {start_epoch}/{NUM_EPOCHS}")
            print(f"  Best val_loss    : {best_val_loss:.4f}")
            print(f"  No-improve count : {no_improve}/{EARLY_STOP_PATIENCE}")
            print()
        except Exception as e:
            print(f"[Warning] Could not load latest checkpoint ({e}) -- starting fresh.")

    if start_epoch >= NUM_EPOCHS:
        print(f"Already completed {start_epoch}/{NUM_EPOCHS} epochs. Nothing to do.")
        _log_fh.close(); return

    if no_improve >= EARLY_STOP_PATIENCE:
        print("Early stopping already triggered before resume. Nothing to do.")
        _log_fh.close(); return

    # ── Training loop ───────────────────────────────────────────────────
    overall_start = time.time()
    print("=" * 70)
    print("Starting training ...")
    print("=" * 70)

    for epoch in range(start_epoch + 1, NUM_EPOCHS + 1):
        ep_t = time.time()
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}  [{time.strftime('%H:%M:%S')}]")

        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        va_loss, va_acc = run_validation(model, val_loader, criterion)
        ep_min = (time.time() - ep_t) / 60

        print(f"  train  loss={tr_loss:.4f}  acc={tr_acc:.4f}")
        print(f"  val    loss={va_loss:.4f}  acc={va_acc:.4f}  ({ep_min:.1f} min)")

        scheduler.step(va_loss)

        is_best = va_loss < best_val_loss
        if is_best:
            best_val_loss = va_loss
            no_improve    = 0
        else:
            no_improve += 1
            print(f"  no val_loss improvement  ({no_improve}/{EARLY_STOP_PATIENCE} patience used)")

        history.append({
            "epoch"   : epoch,
            "tr_loss" : round(tr_loss, 6),
            "tr_acc"  : round(tr_acc, 6),
            "va_loss" : round(va_loss, 6),
            "va_acc"  : round(va_acc, 6),
            "is_best" : is_best,
        })

        save_checkpoints(model, optimizer, scheduler, epoch,
                         tr_loss, tr_acc, va_loss, va_acc,
                         best_val_loss, no_improve, history, is_best)

        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping triggered at epoch {epoch} "
                  f"(val_loss unchanged for {EARLY_STOP_PATIENCE} epochs).")
            break

    total_min = (time.time() - overall_start) / 60
    print()
    print("=" * 70)
    print("DenseNet121 training complete.")
    print(f"  Best val_loss   : {best_val_loss:.4f}")
    print(f"  Total time      : {total_min:.1f} min")
    print(f"  Epoch snapshots : {CKPT_DIR}")
    print(f"  Latest ckpt     : {LATEST_CK}")
    print(f"  Best model      : {BEST_MODEL}")
    print()
    print("Run next:  python ml/evaluate_all_models.py")
    print("=" * 70)
    _log_fh.close()

if __name__ == "__main__":
    main()
