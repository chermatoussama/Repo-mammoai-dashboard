# ─────────────────────────────────────────────────────────
# core/inference.py  — predict_single + predict_batch
# ─────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from core.preprocessing import val_transform, build_clinical
from core.config import LABEL_MAP


def predict_single(model, pil_image: Image.Image,
                   density: str, view: str,
                   threshold: float, device: str = "cpu") -> dict:
    """
    Run inference on a single PIL image with clinical metadata.

    Returns:
        dict with keys: label, label_name, prob_abnormal, prob_normal, threshold
    """
    # Convert to RGB numpy array (grayscale mammograms → 3-channel)
    img_np = np.array(pil_image.convert("RGB"))

    # Apply validation transform
    img_t  = val_transform(image=img_np)["image"].unsqueeze(0).to(device)
    clin_t = torch.from_numpy(build_clinical(density, view)).unsqueeze(0).to(device)

    with torch.no_grad():
        logits       = model(img_t, clin_t)
        probs        = F.softmax(logits, dim=1)[0]
        prob_normal  = probs[0].item()
        prob_abnorm  = probs[1].item()

    label = 1 if prob_abnorm >= threshold else 0

    return {
        "label":        label,
        "label_name":   LABEL_MAP[label],
        "prob_abnormal": prob_abnorm,
        "prob_normal":  prob_normal,
        "threshold":    threshold,
        "confidence":   max(prob_normal, prob_abnorm),
    }


def predict_batch(model, rows: list[dict], threshold: float,
                  device: str = "cpu") -> list[dict]:
    """
    Run inference on a list of dicts with keys:
        pil_image, density, view
    Returns a list of result dicts (same as predict_single).
    """
    results = []
    for row in rows:
        result = predict_single(
            model,
            row["pil_image"],
            row["density"],
            row["view"],
            threshold,
            device,
        )
        results.append(result)
    return results
