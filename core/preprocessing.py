# ─────────────────────────────────────────────────────────
# core/preprocessing.py  — val_transform + clinical encoder
# ─────────────────────────────────────────────────────────
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from core.config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, DENSITY_MAP, VIEW_MAP


val_transform = A.Compose([
    A.LongestMaxSize(max_size=IMG_SIZE),
    A.PadIfNeeded(
        min_height=IMG_SIZE, min_width=IMG_SIZE,
        border_mode=cv2.BORDER_CONSTANT, fill=0
    ),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


def build_clinical(density_str: str, view_str: str) -> np.ndarray:
    """
    Encode breast density + view position into a 6-dim one-hot vector.
    Indices 0-3 → DENSITY A/B/C/D
    Indices 4-5 → CC / MLO
    """
    vec = np.zeros(6, dtype=np.float32)
    vec[DENSITY_MAP[density_str]] = 1.0
    vec[4 + VIEW_MAP[view_str]] = 1.0
    return vec


def birads_to_label(birads_str) -> int | None:
    """Convert BI-RADS string to binary label (0=Normal, 1=Abnormal)."""
    try:
        s = str(birads_str).upper().replace('BI-RADS', '').replace('BIRADS', '').strip()
        digits = ''.join(ch for ch in s if ch.isdigit())
        score = int(digits)
        return 1 if score >= 3 else 0
    except Exception:
        return None
