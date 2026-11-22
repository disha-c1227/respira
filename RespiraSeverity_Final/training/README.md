# Training & ML Scripts Directory

The machine learning and deep learning scripts are located in the `ml/` directory:

- `ml/train_disease.py`: Trains the four-class MobileNetV3-Small disease classifier with class-balanced cross-entropy.
- `ml/evaluate_disease.py`: Evaluates the disease classifier on the 20% holdout split (5,687 radiographs).
- `ml/train_severity.py`: Trains the supervised COVID-19 severity model on 760 cases.
- `ml/evaluate_severity.py`: Evaluates the COVID-19 severity model on the held-out test split.
- `ml/run_clustering.py`: Extracts 1024-D MobileNetV3 embeddings for COVID, Pneumonia, and TB, computes COVID reference centroids, fits K-Means (k=3), standardizes features, and exports PCA projections and clustering metrics.
