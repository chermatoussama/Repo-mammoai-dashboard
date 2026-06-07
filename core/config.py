# ─────────────────────────────────────────────────────────
# core/config.py  — All dashboard constants
# ─────────────────────────────────────────────────────────
import os

# Image
IMG_SIZE      = 512
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Model
MODEL_PATH    = os.path.join("assets", "model_weights.pth")

# Decision threshold used for the Abnormal/Normal call shown in the dashboard.
# Lower than the checkpoint's F1-optimal threshold (~0.68) on purpose: in a
# screening context, missing an abnormal case (false negative) is far costlier
# than flagging a normal case for review (false positive), so we trade some
# specificity for higher sensitivity toward "Abnormal".
SCREENING_THRESHOLD = 0.60

# Clinical encodings  (must match training notebook exactly)
DENSITY_MAP = {"DENSITY A": 0, "DENSITY B": 1, "DENSITY C": 2, "DENSITY D": 3}
VIEW_MAP    = {"CC": 0, "MLO": 1}
LABEL_MAP   = {0: "Normal ✅", 1: "Abnormal ⚠️"}

# Pre-computed asset paths
TEST_PRED_CSV     = os.path.join("assets", "precomputed", "test_predictions.csv")
TRAINING_CURVES   = os.path.join("assets", "precomputed", "training_curves.png")
EVALUATION_PLOT   = os.path.join("assets", "precomputed", "evaluation.png")
SAMPLE_CSV        = os.path.join("assets", "sample_data", "sample_breast_level.csv")
SAMPLE_IMAGE      = os.path.join("assets", "sample_data", "sample_image.png")

# Thesis metadata
THESIS_TITLE      = "Breast Cancer Detection Using Deep Learning"
STUDENT_NAME      = "Chermato Oussama"
UNIVERSITY        = "University — Thesis Project"
YEAR              = "2025–2026"
MODEL_FULL_NAME   = "EfficientNet-B3 NoisyStudent Dual-Stream"
DATASET_NAME      = "VinDr-Mammo"
