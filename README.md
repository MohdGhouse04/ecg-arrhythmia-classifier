<<<<<<< HEAD
# 🫀 ECG Arrhythmia Classification — 95%+ Accuracy

**12-Lead ECG | CNN + BiLSTM + Self-Attention | WFDB_ChapmanShaoxing**

---

## Project Structure

```
ecg_project/
├── configs/
│   └── config.py              ← All hyperparameters & paths in one place
├── src/
│   ├── data_loader.py         ← Load .mat/.hea files, filter rare classes
│   ├── preprocessor.py        ← Split → Downsample → Normalize → SMOTE
│   ├── model.py               ← CNN + BiLSTM + Attention, Focal Loss
│   ├── trainer.py             ← Callbacks, training loop
│   └── evaluator.py           ← Metrics, TTA, ensemble, plots
├── scripts/
│   ├── train.py               ← End-to-end training entry point
│   └── predict.py             ← Single-file inference
├── notebooks/
│   └── ECG_95_Accuracy.ipynb  ← Original Colab notebook
├── models/                    ← (empty — populated after training)
├── outputs/                   ← Saved model, plots, metrics.json
├── tests/
│   └── test_pipeline.py       ← Smoke tests (pytest)
├── requirements.txt
└── README.md
```

---

## Dataset Setup

1. Download from PhysioNet:  
   `a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0`
2. Confirm the path in `configs/config.py`:
   ```python
   DATASET_PATH = "/Users/mohdghouse/Downloads/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/WFDB_ChapmanShaoxing"
   ```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run training (Focal Loss — default)
python scripts/train.py

# 3. Or run with Label Smoothing instead
python scripts/train.py --use-label-smoothing

# 4. Predict on a new ECG file
python scripts/predict.py --mat /path/to/JS00001.mat

# 5. Run tests
pytest tests/ -v
```

---

## Model Architecture

```
Input (1000, 12)
  → CNN Block 1 : Conv1D(64, k=7) → BN → MaxPool → Dropout(0.2)
  → CNN Block 2 : Conv1D(128, k=5) → BN → MaxPool → Dropout(0.3)
  → CNN Block 3 : Conv1D(256, k=3) → BN → MaxPool → Dropout(0.3)
  → BiLSTM(128, return_sequences=True) → Dropout(0.3)
  → Self-Attention (Bahdanau-style)
  → GlobalAveragePooling1D
  → Dense(256) → Dropout(0.5)
  → Softmax(num_classes)
```

---

## Key Design Decisions

| Issue | Fix |
|---|---|
| Test normalization bug | Per-sample normalization (no leakage) |
| EarlyStopping too aggressive | Patience = 8 |
| No LR scheduling | `ReduceLROnPlateau` (factor=0.5, patience=3) |
| Class imbalance | SMOTE + Focal Loss |
| Overfitting | Multi-level Dropout + BatchNorm |
| Low minority-class recall | Focal Loss (γ=2.0, α=0.25) |

---

## Expected Outputs

After training completes, the `outputs/` folder will contain:

```
outputs/
├── best_ecg_model.keras       ← Best checkpoint (by val_accuracy)
├── final_ecg_model.keras      ← Final saved model
├── label_encoder.pkl          ← Sklearn LabelEncoder
├── training_curves.png        ← Accuracy & loss curves
├── confusion_matrix.png       ← Heatmap across all classes
└── metrics.json               ← All numeric metrics
```

### metrics.json (example)
```json
{
  "test_accuracy":      0.9531,
  "macro_precision":    0.9412,
  "macro_recall":       0.9387,
  "macro_f1":           0.9399,
  "weighted_precision": 0.9538,
  "weighted_recall":    0.9531,
  "weighted_f1":        0.9533,
  "tta_accuracy":       0.9562
}
```

### Console output (example)
```
==========================================
  TEST ACCURACY : 0.9531  (95.31%)
  TEST LOSS     : 0.1842
==========================================

🎉 Target of 95% ACHIEVED!

              precision  recall  f1-score  support
  SNR             0.97    0.98      0.97      612
  AF              0.94    0.93      0.94      198
  IAVB            0.96    0.95      0.95      143
  LBBB            0.98    0.97      0.97       89
  RBBB            0.95    0.96      0.95      201
  PAC             0.91    0.90      0.90       74
  PVC             0.93    0.92      0.92       58

TTA Test Accuracy: 95.62%
```
=======
# ecg-arrhythmia-classifier
>>>>>>> d787db854b417b6de31a09ceeafd4169056f425c
