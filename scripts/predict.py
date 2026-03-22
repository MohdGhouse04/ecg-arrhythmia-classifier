#!/usr/bin/env python3
"""
scripts/predict.py
──────────────────
Run inference on a single .mat ECG file.

Usage:
  python scripts/predict.py --mat path/to/JS00001.mat
"""

import os
import sys
import argparse
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from scipy.io import loadmat
from configs.config import (
    FINAL_MODEL_PATH, LABEL_ENCODER_PATH, DOWNSAMPLE_FACTOR
)


def load_single(mat_path: str) -> np.ndarray:
    mat = loadmat(mat_path)
    signal = mat["val"].T                     # (5000, 12)
    signal = signal[::DOWNSAMPLE_FACTOR, :]   # (1000, 12)
    signal = signal.astype("float32")
    mean = np.mean(signal, axis=0, keepdims=True)
    std  = np.std(signal,  axis=0, keepdims=True)
    return (signal - mean) / (std + 1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", required=True, help="Path to .mat ECG file")
    parser.add_argument("--model", default=FINAL_MODEL_PATH)
    parser.add_argument("--encoder", default=LABEL_ENCODER_PATH)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model, compile=False)
    with open(args.encoder, "rb") as f:
        le = pickle.load(f)

    signal = load_single(args.mat)
    X = signal[np.newaxis, ...]               # (1, 1000, 12)
    probs = model.predict(X, verbose=0)[0]
    pred_idx = np.argmax(probs)
    pred_label = le.inverse_transform([pred_idx])[0]

    print(f"\nFile          : {args.mat}")
    print(f"Prediction    : {pred_label}  (class {pred_idx})")
    print(f"Confidence    : {probs[pred_idx]*100:.1f}%")
    print("\nAll class probabilities:")
    for i, (cls, p) in enumerate(zip(le.classes_, probs)):
        bar = "█" * int(p * 40)
        print(f"  {cls:<15} {p*100:5.1f}%  {bar}")


if __name__ == "__main__":
    main()
