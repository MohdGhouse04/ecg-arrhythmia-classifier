# ============================================================
# tests/test_pipeline.py — Smoke tests for the ECG pipeline
# ============================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.preprocessor import downsample_signals, normalize_per_sample
from src.model import build_cnn_bilstm_attention, focal_loss
import tensorflow as tf


# ── Preprocessor tests ───────────────────────────────────────────────────────

def test_downsample():
    X_train = np.random.randn(10, 5000, 12).astype("float32")
    X_test  = np.random.randn(4,  5000, 12).astype("float32")
    X_tr, X_te = downsample_signals(X_train, X_test, factor=5)
    assert X_tr.shape == (10, 1000, 12)
    assert X_te.shape == (4,  1000, 12)


def test_normalize_shape():
    X_train = np.random.randn(8, 1000, 12).astype("float32") * 100
    X_test  = np.random.randn(3, 1000, 12).astype("float32") * 100
    Xtr, Xte = normalize_per_sample(X_train, X_test)
    assert Xtr.shape == X_train.shape
    assert Xte.shape == X_test.shape


def test_normalize_range():
    """After per-sample normalization, most values should be in [-5, 5]."""
    X = np.random.randn(20, 1000, 12).astype("float32") * 500
    Xn, _ = normalize_per_sample(X, X.copy())
    assert np.abs(Xn).max() < 50   # generous upper bound


# ── Model tests ──────────────────────────────────────────────────────────────

def test_model_output_shape():
    model = build_cnn_bilstm_attention(input_shape=(1000, 12), num_classes=7)
    dummy = np.random.randn(2, 1000, 12).astype("float32")
    preds = model.predict(dummy, verbose=0)
    assert preds.shape == (2, 7)


def test_model_probabilities_sum_to_one():
    model = build_cnn_bilstm_attention(input_shape=(1000, 12), num_classes=7)
    dummy = np.random.randn(4, 1000, 12).astype("float32")
    preds = model.predict(dummy, verbose=0)
    np.testing.assert_allclose(preds.sum(axis=1), np.ones(4), atol=1e-5)


def test_focal_loss_not_negative():
    loss_fn = focal_loss(gamma=2.0, alpha=0.25, num_classes=7)
    y_true  = tf.constant([0, 1, 2, 3])
    y_pred  = tf.constant(np.random.dirichlet(np.ones(7), size=4), dtype=tf.float32)
    loss_val = loss_fn(y_true, y_pred)
    assert loss_val.numpy() >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
