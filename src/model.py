# ============================================================
# src/model.py — CNN + BiLSTM + Self-Attention architecture
# ============================================================

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling1D,
    Bidirectional, LSTM, Multiply, Permute,
    RepeatVector, Flatten, Activation
)
from tensorflow.keras.models import Model

from configs.config import (
    CNN_FILTERS, CNN_KERNELS, LSTM_UNITS, DENSE_UNITS,
    DROPOUT_CNN, DROPOUT_LSTM, DROPOUT_DENSE,
    TIMESTEPS, NUM_LEADS
)


def build_cnn_bilstm_attention(input_shape: tuple = (TIMESTEPS, NUM_LEADS),
                                num_classes: int = 7) -> Model:
    inputs = Input(shape=input_shape, name="ecg_input")

    x = inputs
    for filters, kernel, dropout in zip(CNN_FILTERS, CNN_KERNELS, DROPOUT_CNN):
        x = Conv1D(filters, kernel, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(dropout)(x)

    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x)
    x = Dropout(DROPOUT_LSTM)(x)

    attention_scores  = Dense(1, activation="tanh")(x)
    attention_scores  = Flatten()(attention_scores)
    attention_weights = Activation("softmax")(attention_scores)
    attention_weights = RepeatVector(LSTM_UNITS * 2)(attention_weights)
    attention_weights = Permute([2, 1])(attention_weights)
    x = Multiply()([x, attention_weights])
    x = GlobalAveragePooling1D()(x)

    x = Dense(DENSE_UNITS, activation="relu")(x)
    x = Dropout(DROPOUT_DENSE)(x)
    outputs = Dense(num_classes, activation="softmax", name="predictions")(x)

    return Model(inputs, outputs, name="CNN_BiLSTM_Attention")


def focal_loss(gamma: float = 2.0, alpha: float = 0.25, num_classes: int = 7):
    def loss_fn(y_true, y_pred):
        y_true    = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_true_oh = tf.one_hot(y_true, num_classes)
        y_pred    = tf.clip_by_value(y_pred, 1e-8, 1.0)
        cross_ent = -y_true_oh * tf.math.log(y_pred)
        focal_wt  = alpha * tf.pow(1.0 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(focal_wt * cross_ent, axis=1))
    return loss_fn


def compile_model(model: Model, num_classes: int,
                  learning_rate: float = 1e-4,
                  use_focal_loss: bool = True) -> Model:
    if use_focal_loss:
        loss = focal_loss(gamma=2.0, alpha=0.25, num_classes=num_classes)
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy"]
    )
    print(f"Model compiled | loss={'FocalLoss' if use_focal_loss else 'SparseCategoricalCrossentropy'}")
    return model