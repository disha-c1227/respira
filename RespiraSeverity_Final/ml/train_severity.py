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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(PROJECT_ROOT, "data", "manifest.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "severity_checkpoints")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 8
LR = 0.0001

CLASS_NAMES = ["Mild", "Moderate", "Severe"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

print("=" * 70)
print("RespiraSeverity - COVID Severity Training")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print()

df = pd.read_csv(MANIFEST)

# Keep only explicit COVID severity labels
df["severity"] = df["severity"].fillna("").astype(str).str.strip()

df = df[df["severity"].isin(CLASS_NAMES)].copy()

# Severity labels in the current manifest are associated with COVID/RICORD cases.
print("Severity dataset:")
print(df["severity"].value_counts())
print(f"Total images: {len(df)}")
print()

# Resolve image paths
def resolve_path(p):
    p = str(p)

    if os.path.isabs(p):
        return p

    return os.path.normpath(os.path.join(PROJECT_ROOT, p))

df["image_path"] = df["path"].apply(resolve_path)

# Remove files that cannot be found
exists = df["image_path"].apply(os.path.isfile)

missing = (~exists).sum()

if missing > 0:
    print(f"WARNING: {missing} image files could not be found and will be skipped.")

df = df[exists].copy()

print(f"Usable images: {len(df)}")
print()

# Convert labels
df["label"] = df["severity"].map(CLASS_TO_IDX)

# Stratified train/validation/test split
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=SEED,
    stratify=df["label"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_df["label"]
)

print("Dataset split:")
print(f"Train: {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test: {len(test_df)}")
print()

print("Train distribution:")
print(train_df["severity"].value_counts())
print()

print("Validation distribution:")
print(val_df["severity"].value_counts())
print()

print("Test distribution:")
print(test_df["severity"].value_counts())
print()

# Save deterministic splits for evaluation
split_dir = os.path.join(PROJECT_ROOT, "data", "severity_splits")
os.makedirs(split_dir, exist_ok=True)

train_df.to_csv(os.path.join(split_dir, "train.csv"), index=False)
val_df.to_csv(os.path.join(split_dir, "val.csv"), index=False)
test_df.to_csv(os.path.join(split_dir, "test.csv"), index=False)

print(f"Saved splits to: {split_dir}")
print()

class SeverityDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row["image_path"]
        label = int(row["label"])

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Could not open image: {image_path}\n{e}")

        if self.transform:
            image = self.transform(image)

        return image, label

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(7),
    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = SeverityDataset(train_df, train_transform)
val_dataset = SeverityDataset(val_df, eval_transform)
test_dataset = SeverityDataset(test_df, eval_transform)

num_workers = 0

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers
)

# MobileNetV3-Small
print("Loading MobileNetV3-Small...")

try:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
except Exception:
    print("Could not load pretrained weights. Using untrained model.")
    model = models.mobilenet_v3_small(weights=None)

in_features = model.classifier[-1].in_features

model.classifier[-1] = nn.Linear(
    in_features,
    len(CLASS_NAMES)
)

model = model.to(DEVICE)

# Class weights for imbalance
train_counts = train_df["label"].value_counts().sort_index()

weights_list = []
for i in range(len(CLASS_NAMES)):
    count = train_counts.get(i, 1)
    weights_list.append(len(train_df) / (len(CLASS_NAMES) * count))

class_weights = torch.tensor(
    weights_list,
    dtype=torch.float32,
    device=DEVICE
)

print(f"Class weights: {class_weights.detach().cpu().numpy()}")

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)

best_val_acc = 0.0

print()
print("=" * 70)
print("Starting training")
print("=" * 70)

for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader, start=1):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 10 == 0 or batch_idx == len(train_loader):
            batch_acc = correct / total
            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"Batch {batch_idx}/{len(train_loader)} | "
                f"Train Acc {batch_acc * 100:.2f}%",
                end="\r"
            )

    train_loss = running_loss / total
    train_acc = correct / total

    # Validation
    model.eval()

    val_correct = 0
    val_total = 0
    val_loss_total = 0.0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss_total += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss_total / val_total
    val_acc = val_correct / val_total

    scheduler.step(val_acc)

    print()
    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc * 100:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc * 100:.2f}%"
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
        "class_names": CLASS_NAMES,
        "img_size": IMG_SIZE
    }

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"severity_epoch_{epoch}.pt"
    )

    torch.save(checkpoint, checkpoint_path)

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "img_size": IMG_SIZE
            },
            os.path.join(MODEL_DIR, "covid_severity_model.pt")
        )

        print(
            f"*** New best model saved: "
            f"{best_val_acc * 100:.2f}% validation accuracy ***"
        )

print()
print("=" * 70)
print("Training complete")
print("=" * 70)
print(f"Best validation accuracy: {best_val_acc * 100:.2f}%")
print(
    "Model saved to:",
    os.path.join(MODEL_DIR, "covid_severity_model.pt")
)
print()
print("IMPORTANT: The final evaluation should use data/severity_splits/test.csv")
print("which was kept separate from training.")
