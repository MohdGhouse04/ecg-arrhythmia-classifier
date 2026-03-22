#!/usr/bin/env python3
import os
import sys
import argparse
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from configs.config import (
    FINAL_MODEL_PATH, LABEL_ENCODER_PATH, METRICS_JSON,
    BASE_OUTPUT_DIR, LEARNING_RATE
)
from src.data_loader import load_ecg_dataset, filter_rare_classes
from src.preprocessor import full_preprocessing_pipeline
from src.model import build_cnn_bilstm_attention, compile_model
from src.trainer import train_model
from src.evaluator import (
    evaluate_model, plot_training_curves, plot_confusion_matrix,
    evaluate_tta, save_metrics
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--use-label-smoothing", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    print("\n" + "="*55)
    print("  ECG Arrhythmia Classification — Training Pipeline")
    print("="*55)
    print(f"  TensorFlow : {tf.__version__}")
    print(f"  GPUs       : {tf.config.list_physical_devices('GPU')}")
    print(f"  LR         : {LEARNING_RATE}")
    print("="*55 + "\n")

    # 1. Load & filter
    X, y_raw = load_ecg_dataset()
    X, y, le = filter_rare_classes(X, y_raw)
    NUM_CLASSES = len(np.unique(y))
    class_names = list(le.classes_)
    print(f"NUM_CLASSES: {NUM_CLASSES}")

    # 2. Preprocess
    X_train, X_test, y_train, y_test, class_weights = full_preprocessing_pipeline(X, y)

    # 3. Build & compile
    model = build_cnn_bilstm_attention(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        num_classes=NUM_CLASSES
    )
    model.summary()
    model = compile_model(model, NUM_CLASSES,
                          learning_rate=LEARNING_RATE,
                          use_focal_loss=not args.use_label_smoothing)

    # 4. Train
    history = train_model(model, X_train, y_train,
                          X_val=X_test, y_val=y_test,
                          class_weights=class_weights)

    # 5. Evaluate
    test_acc, y_pred, metrics = evaluate_model(
        model, X_test, y_test, NUM_CLASSES, class_names
    )
    plot_training_curves(history)
    plot_confusion_matrix(y_test, y_pred, NUM_CLASSES, class_names)

    # 6. TTA
    tta_acc = evaluate_tta(model, X_test, y_test)
    metrics["tta_accuracy"] = float(tta_acc)

    # 7. Save
    model.save(FINAL_MODEL_PATH)
    print(f"Model saved → {FINAL_MODEL_PATH}")

    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
    print(f"Label encoder saved → {LABEL_ENCODER_PATH}")

    save_metrics(metrics, METRICS_JSON)

    print("\n" + "="*50)
    print("           FINAL RESULTS SUMMARY")
    print("="*50)
    print(f"  Main Model Test Acc : {test_acc*100:.2f}%")
    print(f"  TTA Test Acc        : {tta_acc*100:.2f}%")
    print("="*50)


if __name__ == "__main__":
    main()