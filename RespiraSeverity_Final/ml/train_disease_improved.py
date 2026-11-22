import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ============================================================
# RespiraSeverity - Improved Disease Model Experiment
# Baseline model is NEVER overwritten
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MANIFEST = os.path.join(
    PROJECT_ROOT,
    "data",
    "manifest.csv"
)

BASELINE_MODEL = os.path.join(
    PROJECT_ROOT,
    "models",
    "disease_model_baseline_80_62.pt"
)

OUTPUT_MODEL = os.path.join(
    PROJECT_ROOT,
    "models",
    "disease_model_improved.pt"
)

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "disease_improved_checkpoints"
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 8

# Lower learning rate for fine-tuning an already trained model
BACKBONE_LR = 1e-5
CLASSIFIER_LR = 5e-5

CLASS_NAMES = [
    "Normal",
    "Pneumonia",
    "TB",
    "COVID"
]

CLASS_TO_IDX = {
    name: i for i, name in enumerate(CLASS_NAMES)
}

print("=" * 75)
print("RespiraSeverity - Improved Disease Model")
print("=" * 75)
print(f"Device: {DEVICE}")
print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print()
print("IMPORTANT:")
print("The original disease_model.pt will NOT be modified.")
print("Output model:", OUTPUT_MODEL)
print()

# ============================================================
# Load manifest
# ============================================================

df = pd.read_csv(MANIFEST)

df["disease"] = (
    df["disease"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df = df[
    df["disease"].isin(CLASS_NAMES)
].copy()

print("Dataset distribution:")
print(df["disease"].value_counts())
print(f"Total images: {len(df)}")
print()

# ============================================================
# Resolve paths
# ============================================================

def resolve_path(p):

    p = str(p)

    if os.path.isabs(p):
        return p

    return os.path.normpath(
        os.path.join(PROJECT_ROOT, p)
    )

df["image_path"] = df["path"].apply(resolve_path)

exists = df["image_path"].apply(os.path.isfile)

missing = (~exists).sum()

if missing:
    print(
        f"WARNING: {missing} images could not be found."
    )

df = df[exists].copy()

df["label"] = df["disease"].map(CLASS_TO_IDX)

# ============================================================
# SAME 80/20 HOLDOUT LOGIC
# ============================================================

train_df, val_df = train_test_split(
    df,
    test_size=0.20,
    random_state=SEED,
    stratify=df["label"]
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print("Dataset split:")
print(f"Training:   {len(train_df)}")
print(f"Holdout:    {len(val_df)}")
print()

print("Training distribution:")
print(train_df["disease"].value_counts())
print()

print("Holdout distribution:")
print(val_df["disease"].value_counts())
print()

# ============================================================
# Dataset
# ============================================================

class ChestXrayDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform=None
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = row["image_path"]

        label = int(row["label"])

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# Stronger but controlled augmentation
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=8
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.04, 0.04),
        scale=(0.95, 1.05)
    ),

    transforms.ColorJitter(
        brightness=0.12,
        contrast=0.12
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])

val_transform = transforms.Compose([

    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])

train_dataset = ChestXrayDataset(
    train_df,
    train_transform
)

val_dataset = ChestXrayDataset(
    val_df,
    val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ============================================================
# Load MobileNetV3 architecture
# ============================================================

print("Loading MobileNetV3-Small...")

model = models.mobilenet_v3_small(
    weights=None
)

in_features = (
    model.classifier[-1].in_features
)

model.classifier[-1] = nn.Linear(
    in_features,
    len(CLASS_NAMES)
)

# ============================================================
# Load YOUR EXISTING trained model
# ============================================================

if not os.path.isfile(BASELINE_MODEL):

    raise FileNotFoundError(
        "Baseline model not found:\n"
        + BASELINE_MODEL
    )

checkpoint = torch.load(
    BASELINE_MODEL,
    map_location="cpu"
)

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint

model.load_state_dict(
    state_dict,
    strict=True
)

print(
    "Existing 80.62% baseline model loaded."
)

model = model.to(DEVICE)

# ============================================================
# Fine-tune entire network with small learning rate
# ============================================================

for parameter in model.parameters():

    parameter.requires_grad = True

# Separate learning rates:
# backbone = very small
# classifier = larger

backbone_parameters = []

classifier_parameters = []

for name, parameter in model.named_parameters():

    if not parameter.requires_grad:
        continue

    if name.startswith("classifier"):

        classifier_parameters.append(
            parameter
        )

    else:

        backbone_parameters.append(
            parameter
        )

optimizer = torch.optim.AdamW(

    [
        {
            "params": backbone_parameters,
            "lr": BACKBONE_LR
        },

        {
            "params": classifier_parameters,
            "lr": CLASSIFIER_LR
        }
    ],

    weight_decay=1e-4
)

# ============================================================
# Class-weighted loss
# ============================================================

counts = (
    train_df["label"]
    .value_counts()
    .sort_index()
)

weights = []

for i in range(
    len(CLASS_NAMES)
):

    count = counts.get(
        i,
        1
    )

    weights.append(
        len(train_df)
        /
        (
            len(CLASS_NAMES)
            * count
        )
    )

class_weights = torch.tensor(
    weights,
    dtype=torch.float32,
    device=DEVICE
)

print(
    "Class weights:",
    class_weights.cpu().numpy()
)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=1
)

# ============================================================
# Training
# ============================================================

best_val_acc = 0.0
best_epoch = 0

print()
print("=" * 75)
print("Starting improvement experiment")
print("=" * 75)
print()

for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, (
        images,
        labels
    ) in enumerate(
        train_loader,
        start=1
    ):

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        optimizer.zero_grad()

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=2.0
        )

        optimizer.step()

        train_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = (
            outputs.argmax(
                dim=1
            )
        )

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += (
            labels.size(0)
        )

        if (
            batch_idx % 10 == 0
            or
            batch_idx == len(
                train_loader
            )
        ):

            live_acc = (
                train_correct
                /
                train_total
            )

            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"Batch {batch_idx}/{len(train_loader)} | "
                f"Train Acc {live_acc*100:.2f}%",
                end="\r"
            )

    train_loss /= train_total

    train_acc = (
        train_correct
        /
        train_total
    )

    # ========================================================
    # Validation
    # ========================================================

    model.eval()

    val_loss = 0.0

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            val_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += (
                labels.size(0)
            )

    val_loss /= val_total

    val_acc = (
        val_correct
        /
        val_total
    )

    scheduler.step(
        val_acc
    )

    print()

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc*100:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc*100:.2f}%"
    )

    # ========================================================
    # Save checkpoint every epoch
    # ========================================================

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"disease_improved_epoch_{epoch}.pt"
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "class_names": CLASS_NAMES,
            "img_size": IMG_SIZE
        },
        checkpoint_path
    )

    # ========================================================
    # Save BEST improved model
    # ========================================================

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        best_epoch = epoch

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "class_names":
                    CLASS_NAMES,

                "img_size":
                    IMG_SIZE,

                "best_val_accuracy":
                    best_val_acc
            },

            OUTPUT_MODEL
        )

        print(
            f"*** NEW BEST MODEL: "
            f"{best_val_acc*100:.2f}% "
            f"(Epoch {best_epoch}) ***"
        )

print()
print("=" * 75)
print("Improvement experiment complete")
print("=" * 75)

print(
    f"Best validation accuracy: "
    f"{best_val_acc*100:.2f}%"
)

print(
    f"Best epoch: {best_epoch}"
)

print()
print(
    "Original baseline remains untouched:"
)

print(
    "models\\disease_model.pt"
)

print(
    "Improved model:"
)

print(
    "models\\disease_model_improved.pt"
)

