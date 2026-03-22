# ============================================================
# src/trainer.py — Callbacks + training loop
# ============================================================

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from configs.config import (
    MODEL_CHECKPOINT, BATCH_SIZE, EPOCHS,
    EARLY_STOP_PATIENCE, LR_REDUCE_PATIENCE, LR_REDUCE_FACTOR, MIN_LR
)


def get_callbacks(checkpoint_path=MODEL_CHECKPOINT):
    return [
        EarlyStopping(
            monitor="val_accuracy",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE,
            min_lr=MIN_LR,
            verbose=1
        ),
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
    ]


def train_model(model, X_train, y_train,
                X_val=None, y_val=None,
                class_weights=None,
                checkpoint_path=MODEL_CHECKPOINT):
    callbacks = get_callbacks(checkpoint_path)
    
    validation_data = (X_val, y_val) if X_val is not None else None
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=validation_data,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    best_val = max(history.history["val_accuracy"])
    print(f"\nBest Validation Accuracy: {best_val*100:.2f}%")
    return history