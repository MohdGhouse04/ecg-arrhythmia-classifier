import os

DATASET_PATH = "/Users/mohdghouse/Downloads/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/WFDBRecords"

BASE_OUTPUT_DIR      = "outputs"
MODEL_CHECKPOINT     = os.path.join(BASE_OUTPUT_DIR, "best_ecg_model.keras")
FINAL_MODEL_PATH     = os.path.join(BASE_OUTPUT_DIR, "final_ecg_model.keras")
LABEL_ENCODER_PATH   = os.path.join(BASE_OUTPUT_DIR, "label_encoder.pkl")
TRAINING_CURVES_PNG  = os.path.join(BASE_OUTPUT_DIR, "training_curves.png")
CONFUSION_MATRIX_PNG = os.path.join(BASE_OUTPUT_DIR, "confusion_matrix.png")
METRICS_JSON         = os.path.join(BASE_OUTPUT_DIR, "metrics.json")

ORIGINAL_TIMESTEPS    = 5000
DOWNSAMPLE_FACTOR     = 5
TIMESTEPS             = 1000
NUM_LEADS             = 12
MIN_SAMPLES_PER_CLASS = 500   # only keep classes with 500+ real samples

CNN_FILTERS   = [64, 128, 256]
CNN_KERNELS   = [7, 5, 3]
LSTM_UNITS    = 128
DENSE_UNITS   = 256
DROPOUT_CNN   = [0.2, 0.3, 0.3]
DROPOUT_LSTM  = 0.3
DROPOUT_DENSE = 0.5

LEARNING_RATE       = 1e-4
BATCH_SIZE          = 32
EPOCHS              = 60
TEST_SIZE           = 0.2
RANDOM_STATE        = 42
EARLY_STOP_PATIENCE = 10
LR_REDUCE_PATIENCE  = 4
LR_REDUCE_FACTOR    = 0.5
MIN_LR              = 1e-7
VALIDATION_SPLIT    = 0.1

FOCAL_GAMMA  = 2.0
FOCAL_ALPHA  = 0.25

TTA_N_AUGMENTS = 5
TTA_NOISE_LEVEL = 0.02

TARGET_ACCURACY = 0.95