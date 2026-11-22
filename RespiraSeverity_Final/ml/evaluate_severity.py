import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEST_CSV = os.path.join(
    PROJECT_ROOT,
    "data",
    "severity_splits",
    "test.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "covid_severity_model.pt"
)

CLASS_NAMES = ["Mild", "Moderate", "Severe"]

IMG_SIZE = 224
BATCH_SIZE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("RespiraSeverity - COVID Severity Evaluation")
print("=" * 70)
print(f"Device: {DEVICE}")
print()

if not os.path.isfile(TEST_CSV):
    raise FileNotFoundError(
        f"Test split not found: {TEST_CSV}"
    )

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

test_df = pd.read_csv(TEST_CSV)

print(f"Test images: {len(test_df)}")
print()
print("Test class distribution:")
print(test_df["severity"].value_counts())
print()

class_to_idx = {
    name: i for i, name in enumerate(CLASS_NAMES)
}

test_df["label"] = test_df["severity"].map(class_to_idx)

class SeverityDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = str(row["image_path"])
        label = int(row["label"])

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = SeverityDataset(
    test_df,
    transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Loading model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model = models.mobilenet_v3_small(
    weights=None
)

in_features = model.classifier[-1].in_features

model.classifier[-1] = nn.Linear(
    in_features,
    len(CLASS_NAMES)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

all_labels = []
all_predictions = []
all_probabilities = []

print("Running evaluation...")
print()

with torch.no_grad():

    for images, labels in loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

y_true = np.array(all_labels)
y_pred = np.array(all_predictions)

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision, recall, f1, support = precision_recall_fscore_support(
    y_true,
    y_pred,
    labels=[0, 1, 2],
    zero_division=0
)

macro_precision = precision.mean()
macro_recall = recall.mean()
macro_f1 = f1.mean()

print("=" * 70)
print("COVID SEVERITY TEST RESULTS")
print("=" * 70)

print()
print(f"Test Accuracy : {accuracy * 100:.2f}%")
print(f"Macro Precision: {macro_precision * 100:.2f}%")
print(f"Macro Recall   : {macro_recall * 100:.2f}%")
print(f"Macro F1       : {macro_f1 * 100:.2f}%")

print()
print("Per-class results:")
print()

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print(report)

print("Confusion Matrix")
print("-" * 40)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1, 2]
)

print(
    "              Predicted"
)
print(
    "             Mild Moderate Severe"
)

for i, name in enumerate(CLASS_NAMES):

    print(
        f"{name:10s} "
        f"{cm[i,0]:5d} "
        f"{cm[i,1]:8d} "
        f"{cm[i,2]:7d}"
    )

print()

print("=" * 70)
print("Evaluation complete")
print("=" * 70)

print()
print("Model evaluated on the held-out test set:")
print(TEST_CSV)
