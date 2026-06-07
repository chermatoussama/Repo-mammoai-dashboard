# ─────────────────────────────────────────────────────────
# core/gradcam.py  — Grad-CAM implementation (from notebook)
# ─────────────────────────────────────────────────────────
import numpy as np
import cv2
import torch
from core.config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_gradcam(model, image_tensor: torch.Tensor,
                clinical_tensor: torch.Tensor,
                target_layer) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for the Abnormal class (index 1).

    Args:
        model          : EfficientNetB3DualStream (eval mode)
        image_tensor   : shape (C, H, W) — already normalised
        clinical_tensor: shape (6,) — one-hot clinical vector
        target_layer   : model.backbone.conv_head

    Returns:
        cam: np.ndarray of shape (IMG_SIZE, IMG_SIZE) in [0, 1]
    """
    gradients, activations = [], []

    def bwd_hook(m, gi, go):
        gradients.append(go[0])

    def fwd_hook(m, i, o):
        activations.append(o)

    hb = target_layer.register_full_backward_hook(bwd_hook)
    hf = target_layer.register_forward_hook(fwd_hook)

    model.eval()
    model.zero_grad()

    out   = model(image_tensor.unsqueeze(0), clinical_tensor.unsqueeze(0))
    score = out[0, 1]   # Abnormal logit
    score.backward()

    grads   = gradients[0].cpu().detach().numpy()[0]   # (C, H, W)
    fmap    = activations[0].cpu().detach().numpy()[0] # (C, H, W)
    weights = np.mean(grads, axis=(1, 2))              # (C,)

    cam = np.sum(weights[:, None, None] * fmap, axis=0)
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    denom = cam.max() - cam.min() + 1e-8
    cam = (cam - cam.min()) / denom

    hb.remove()
    hf.remove()
    return cam          # float32 in [0,1]


def overlay_gradcam(orig_img_np: np.ndarray, cam: np.ndarray,
                    alpha: float = 0.45) -> np.ndarray:
    """
    Blend a Grad-CAM heatmap with the original image.

    Args:
        orig_img_np : (H, W, 3) uint8 or float [0,1]
        cam         : (H, W) float [0,1]
        alpha       : blend weight for heatmap

    Returns:
        overlay: (H, W, 3) uint8 RGB
    """
    if orig_img_np.dtype != np.uint8:
        disp = np.clip(orig_img_np, 0.0, 1.0)
    else:
        disp = orig_img_np.astype(np.float32) / 255.0

    hmap  = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    hmap  = np.float32(hmap[:, :, ::-1]) / 255.0    # BGR→RGB

    overlay = alpha * hmap + (1.0 - alpha) * disp
    overlay = np.clip(overlay, 0.0, 1.0)
    return (overlay * 255).astype(np.uint8)


def denormalize(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Undo ImageNet normalisation, return (H, W, 3) float32 in [0, 1].
    """
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    img  = img_tensor.cpu().numpy().transpose(1, 2, 0)   # (H, W, C)
    img  = img * std + mean
    return np.clip(img, 0.0, 1.0)


def get_gradcam_outputs(model, image_tensor: torch.Tensor,
                        clinical_tensor: torch.Tensor):
    """
    Run Grad-CAM on a preprocessed (image_tensor, clinical_tensor) pair and
    build the four display panels used by the dashboard.

    Returns:
        processed_img : (H, W, 3) uint8 — denormalised input the model saw
        heatmap_img   : (H, W, 3) uint8 RGB — raw colour-mapped CAM
        overlay_img   : (H, W, 3) uint8 RGB — heatmap blended over the image
    """
    cam = get_gradcam(model, image_tensor, clinical_tensor,
                      target_layer=model.backbone.conv_head)

    processed = denormalize(image_tensor)                      # float [0,1]
    processed_img = (processed * 255).astype(np.uint8)

    overlay_img = overlay_gradcam(processed, cam)

    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_img = heatmap_bgr[:, :, ::-1]                       # BGR → RGB

    return processed_img, heatmap_img, overlay_img
