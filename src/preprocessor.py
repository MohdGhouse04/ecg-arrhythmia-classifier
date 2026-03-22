import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from configs.config import DOWNSAMPLE_FACTOR, TEST_SIZE, RANDOM_STATE


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def downsample_signals(X_train, X_test, factor=DOWNSAMPLE_FACTOR):
    X_train = X_train[:, ::factor, :]
    X_test  = X_test[:,  ::factor, :]
    print(f"After downsampling — Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test


def normalize_per_sample(X_train, X_test):
    X_train = X_train.astype("float32")
    X_test  = X_test.astype("float32")
    X_train = (X_train - np.mean(X_train, axis=1, keepdims=True)) / (np.std(X_train,  axis=1, keepdims=True) + 1e-8)
    X_test  = (X_test  - np.mean(X_test,  axis=1, keepdims=True)) / (np.std(X_test,   axis=1, keepdims=True) + 1e-8)
    print(f"Normalization done | Train mean≈{X_train.mean():.4f} std≈{X_train.std():.4f}")
    return X_train, X_test


def cap_majority_classes(X_train, y_train, cap=2000):
    """Cap majority classes so no single class dominates."""
    indices = []
    for cls in np.unique(y_train):
        cls_idx = np.where(y_train == cls)[0]
        if len(cls_idx) > cap:
            cls_idx = np.random.RandomState(42).choice(cls_idx, cap, replace=False)
        indices.extend(cls_idx)
    indices = np.array(indices)
    np.random.RandomState(42).shuffle(indices)
    print(f"After capping: {Counter(y_train[indices])}")
    return X_train[indices], y_train[indices]


def get_class_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw = dict(zip(classes, weights))
    print(f"Class weights computed for {len(classes)} classes")
    return cw


def full_preprocessing_pipeline(X, y):
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train, X_test = downsample_signals(X_train, X_test)
    X_train, X_test = normalize_per_sample(X_train, X_test)
    X_train, y_train = cap_majority_classes(X_train, y_train, cap=2000)
    class_weights = get_class_weights(y_train)
    return X_train, X_test, y_train, y_test, class_weights