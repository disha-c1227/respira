import os
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split


# ============================================================
# SETTINGS
# ============================================================

MANIFEST = "data/manifest.csv"

CHECKPOINT = (
    "models/disease_improved_checkpoints/"
    "disease_improved_epoch_4.pt"
)

CHECKPOINT_DIR = "models/disease_improved_checkpoints"

BEST_MODEL = "models/disease_model_improved.pt"

BATCH_SIZE = 8
TOTAL_EPOCHS = 8
IMG_SIZE = 224

CLASSES = [
    "Normal",
    "Pneumonia",
    "TB",
    "COVID"
]

CLASS_TO_IDX = {
    "Normal": 0,
    "Pneumonia": 1,
    "TB": 2,
    "COVID": 3
}


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading manifest...")

df = pd.read_csv(MANIFEST)

df = df[
    df["disease"].isin(CLASSES)
].copy()

df["label"] = df["disease"].map(CLASS_TO_IDX)

print("Total images:", len(df))


# ============================================================
# SAME 80/20 SPLIT
# ============================================================

train_df, val_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["label"],
    random_state=42
)

print("Training images:", len(train_df))
print("Validation images:", len(val_df))


# ============================================================
# DATASET
# ============================================================

class CXRDataset(Dataset):

    def __init__(self, dataframe, transform=None):

        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = row["path"]

        image = Image.open(
            image_path
        ).convert("RGB")

        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# TRANSFORMS
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
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([

    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = CXRDataset(
    train_df,
    train_transform
)

val_dataset = CXRDataset(
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
# CLASS WEIGHTS
# ============================================================
# Calculated manually.
# This avoids the sklearn compute_class_weight error.

print("\nCalculating class weights...")

class_counts = (
    train_df["label"]
    .value_counts()
    .sort_index()
)

total_samples = len(train_df)
num_classes = len(CLASSES)

weights = []

for class_index in range(num_classes):

    count = int(
        class_counts.get(
            class_index,
            0
        )
    )

    if count == 0:

        weight = 1.0

    else:

        weight = (
            total_samples /
            (num_classes * count)
        )

    weights.append(weight)


class_weights = torch.tensor(
    weights,
    dtype=torch.float32
).to(device)

print(
    "Class weights:",
    class_weights.cpu().numpy()
)


# ============================================================
# CREATE MODEL
# ============================================================

print("\nLoading MobileNetV3-Small...")

model = models.mobilenet_v3_small(
    weights=None
)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    len(CLASSES)
)

model = model.to(device)


# ============================================================
# LOAD EPOCH 4 CHECKPOINT
# ============================================================

print("\nLoading Epoch 4 checkpoint...")

if not os.path.exists(CHECKPOINT):

    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT}"
    )


checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

print(
    "Checkpoint keys:",
    checkpoint.keys()
)


# Load model weights
model.load_state_dict(
    checkpoint["model_state_dict"]
)


# ============================================================
# OPTIMIZER
# ============================================================

backbone_params = []
classifier_params = []

for name, param in model.named_parameters():

    if "classifier" in name:

        classifier_params.append(param)

    else:

        backbone_params.append(param)


optimizer = torch.optim.AdamW(

    [
        {
            "params": backbone_params,
            "lr": 1e-5
        },

        {
            "params": classifier_params,
            "lr": 5e-5
        }
    ],

    weight_decay=1e-4
)


# Restore optimizer state
optimizer.load_state_dict(
    checkpoint["optimizer_state_dict"]
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# RESUME INFORMATION
# ============================================================

previous_epoch = checkpoint["epoch"]

start_epoch = previous_epoch + 1

best_val_acc = checkpoint["best_val_acc"]


print("\n==========================================")
print("RESUMING IMPROVEMENT EXPERIMENT")
print("==========================================")

print(
    "Checkpoint epoch:",
    previous_epoch
)

print(
    "Starting epoch:",
    start_epoch
)

print(
    "Best validation accuracy so far:",
    f"{best_val_acc:.2f}%"
)

print(
    "Remaining epochs:",
    TOTAL_EPOCHS - previous_epoch
)

print("==========================================\n")


# ============================================================
# TRAINING
# ============================================================

for epoch in range(
    start_epoch,
    TOTAL_EPOCHS + 1
):

    model.train()

    running_correct = 0
    running_total = 0

    for batch_idx, (images, labels) in enumerate(
        train_loader,
        start=1
    ):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        _, predictions = torch.max(
            outputs,
            1
        )

        running_total += labels.size(0)

        running_correct += (
            predictions == labels
        ).sum().item()


        # Progress every 50 batches
        if batch_idx % 50 == 0:

            train_acc = (
                100.0 *
                running_correct /
                running_total
            )

            print(
                f"Epoch {epoch}/{TOTAL_EPOCHS} | "
                f"Batch {batch_idx}/{len(train_loader)} | "
                f"Train Acc {train_acc:.2f}%"
            )


    # ========================================================
    # TRAINING ACCURACY
    # ========================================================

    train_acc = (
        100.0 *
        running_correct /
        running_total
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predictions = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predictions == labels
            ).sum().item()


    val_acc = (
        100.0 *
        val_correct /
        val_total
    )


    # ========================================================
    # PRINT EPOCH RESULTS
    # ========================================================

    print("\n")
    print("==========================================")

    print(
        f"Epoch {epoch} COMPLETE"
    )

    print(
        f"Train Accuracy: {train_acc:.2f}%"
    )

    print(
        f"Validation Accuracy: {val_acc:.2f}%"
    )

    print(
        f"Previous Best: {best_val_acc:.2f}%"
    )

    print("==========================================")
    print("\n")


    # ========================================================
    # UPDATE BEST
    # ========================================================

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            BEST_MODEL
        )

        print(
            "NEW BEST MODEL SAVED!"
        )

        print(
            f"Best Validation Accuracy: "
            f"{best_val_acc:.2f}%"
        )

        print(
            f"Saved to: {BEST_MODEL}"
        )

    else:

        print(
            "No new best model this epoch."
        )


    # ========================================================
    # SAVE CHECKPOINT
    # ========================================================

    checkpoint_data = {

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "best_val_acc":
            best_val_acc,

        "class_names":
            CLASSES,

        "img_size":
            IMG_SIZE
    }


    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"disease_improved_epoch_{epoch}.pt"
    )


    torch.save(
        checkpoint_data,
        checkpoint_path
    )


    print(
        f"Checkpoint saved: "
        f"{checkpoint_path}"
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n")
print("==========================================")
print("RESUMED TRAINING COMPLETE")
print("==========================================")

print(
    f"Best Validation Accuracy: "
    f"{best_val_acc:.2f}%"
)

print(
    f"Best model: {BEST_MODEL}"
)

print("==========================================")