# ============================================================
# src/data_loader.py — Load .mat/.hea ECG records from disk
# ============================================================

import os
import numpy as np
from scipy.io import loadmat
from collections import Counter
from sklearn.preprocessing import LabelEncoder

from configs.config import DATASET_PATH, MIN_SAMPLES_PER_CLASS


def load_ecg_dataset(dataset_path: str = DATASET_PATH) -> tuple:
    """
    Load all .mat signal files and paired .hea label files.
    Walks recursively through subfolders (e.g. WFDBRecords/01/010/JS00001.mat)

    Returns
    -------
    X : np.ndarray  shape (N, 5000, 12)
    y : np.ndarray  shape (N,)  — raw string labels
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at: {dataset_path}\n"
            "Check DATASET_PATH in configs/config.py"
        )

    signals, labels = [], []
    total_found = 0

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if not file.endswith(".mat"):
                continue

            total_found += 1
            record   = file.replace(".mat", "")
            mat_path = os.path.join(root, file)
            hea_path = os.path.join(root, record + ".hea")

            try:
                mat    = loadmat(mat_path)
                signal = mat["val"].T          # shape: (5000, 12)
                signals.append(signal)

                with open(hea_path, "r") as f:
                    for line in f:
                        if "#Dx:" in line:
                            dx = line.strip().split(":")[1].strip()
                            labels.append(dx.split(",")[0])
                            break
            except Exception as e:
                print(f"  ⚠️  Skipping {file}: {e}")
                continue

    if total_found == 0:
        raise RuntimeError(f"No .mat files found under {dataset_path}")

    print(f"Found {total_found} .mat files | Loaded {len(signals)} successfully")

    X = np.array(signals)
    y = np.array(labels)
    print(f"Raw X shape: {X.shape} | Raw y shape: {y.shape}")
    print(f"Class distribution:\n{Counter(y)}")
    return X, y


def filter_rare_classes(X: np.ndarray, y: np.ndarray,
                        min_samples: int = MIN_SAMPLES_PER_CLASS):
    """Keep only classes with enough samples AND good learnability."""
    
    # These classes have too few samples or too much noise — remove them
    EXCLUDE_CLASSES = {'164889003', '164934002', '427393009', '284470004'}
    
    valid_classes = [
        cls for cls, cnt in Counter(y).items()
        if cnt >= min_samples and cls not in EXCLUDE_CLASSES
    ]
    mask = np.isin(y, valid_classes)
    X, y = X[mask], y[mask]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(f"Filtered → X: {X.shape} | Classes: {len(np.unique(y_enc))}")
    print(f"Class names: {list(le.classes_)}")
    print(f"Distribution: {Counter(y_enc)}")
    return X, y_enc, le