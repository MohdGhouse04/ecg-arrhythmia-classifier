# ============================================================
# src/evaluator.py — Metrics, TTA, Ensemble, plotting
# ============================================================

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, precision_score, recall_score, f1_score
)

from configs.config import (
    TARGET_ACCURACY, TTA_N_AUGMENTS, TTA_NOISE_LEVEL,
    TRAINING_CURVES_PNG, CONFUSION_MATRIX_PNG, METRICS_JSON
)


# ── Basic evaluation ─────────────────────────────────────────────────────────

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   num_classes: int, class_names: list = None):
    """
    Print test accuracy, classification report, and macro/weighted metrics.
    Returns (test_acc, y_pred).
    """
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'='*42}")
    print(f"  TEST ACCURACY : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"  TEST LOSS     : {test_loss:.4f}")
    print(f"{'='*42}")

    if test_acc >= TARGET_ACCURACY:
        print(f"\n🎉 Target of {TARGET_ACCURACY*100:.0f}% ACHIEVED!")
    else:
        print(f"\n📍 Gap to target: {(TARGET_ACCURACY - test_acc)*100:.2f}%")

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    names = class_names or [f"Class {i}" for i in range(num_classes)]
    print("\n" + classification_report(y_test, y_pred,
                                       target_names=names, zero_division=0))

    metrics = {
        "test_accuracy"      : float(test_acc),
        "macro_precision"    : float(precision_score(y_test, y_pred, average="macro",    zero_division=0)),
        "macro_recall"       : float(recall_score(   y_test, y_pred, average="macro",    zero_division=0)),
        "macro_f1"           : float(f1_score(       y_test, y_pred, average="macro",    zero_division=0)),
        "weighted_precision" : float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "weighted_recall"    : float(recall_score(   y_test, y_pred, average="weighted", zero_division=0)),
        "weighted_f1"        : float(f1_score(       y_test, y_pred, average="weighted", zero_division=0)),
    }
    for k, v in metrics.items():
        print(f"  {k:<22}: {v:.4f}")

    return test_acc, y_pred, metrics


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_training_curves(history, save_path: str = TRAINING_CURVES_PNG):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"],     label="Train",      linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Validation", linewidth=2)
    axes[0].axhline(TARGET_ACCURACY, color="red", linestyle="--",
                    alpha=0.7, label=f"{int(TARGET_ACCURACY*100)}% target")
    axes[0].set_title("Model Accuracy", fontsize=14)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"],     label="Train",      linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Validation", linewidth=2)
    axes[1].set_title("Model Loss", fontsize=14)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Training curves saved → {save_path}")


def plot_confusion_matrix(y_test, y_pred, num_classes: int,
                          class_names: list = None,
                          save_path: str = CONFUSION_MATRIX_PNG):
    cm = confusion_matrix(y_test, y_pred)
    names = class_names or [f"Class {i}" for i in range(num_classes)]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                linewidths=0.5, linecolor="white",
                xticklabels=[f"Pred {n}" for n in names],
                yticklabels=[f"True {n}" for n in names])
    plt.title("Confusion Matrix — CNN+BiLSTM+Attention", fontsize=14)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Confusion matrix saved → {save_path}")


# ── Test-Time Augmentation ───────────────────────────────────────────────────

def tta_predict(model, X: np.ndarray,
                n_augments: int = TTA_N_AUGMENTS,
                noise_level: float = TTA_NOISE_LEVEL) -> np.ndarray:
    """Average predictions over `n_augments` slightly-noised copies of X."""
    all_preds = [model.predict(X, verbose=0)]
    for _ in range(n_augments - 1):
        X_aug = X + np.random.normal(0, noise_level, X.shape).astype("float32")
        all_preds.append(model.predict(X_aug, verbose=0))
    return np.mean(all_preds, axis=0)


def evaluate_tta(model, X_test: np.ndarray, y_test: np.ndarray) -> float:
    probs = tta_predict(model, X_test)
    y_pred = np.argmax(probs, axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"TTA Test Accuracy: {acc*100:.2f}%")
    return acc


# ── Ensemble ─────────────────────────────────────────────────────────────────

def ensemble_predict(models: list, X_test: np.ndarray,
                     y_test: np.ndarray, num_classes: int) -> float:
    """Simple average ensemble over a list of Keras models."""
    all_probs = [m.predict(X_test, verbose=0) for m in models]
    avg_probs  = np.mean(all_probs, axis=0)
    y_pred     = np.argmax(avg_probs, axis=1)
    acc        = accuracy_score(y_test, y_pred)
    print(f"Ensemble Test Accuracy: {acc*100:.2f}%")
    names = [f"Class {i}" for i in range(num_classes)]
    print(classification_report(y_test, y_pred, target_names=names, zero_division=0))
    return acc


# ── Save metrics ─────────────────────────────────────────────────────────────

def save_metrics(metrics: dict, path: str = METRICS_JSON):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {path}")
