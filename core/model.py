# ─────────────────────────────────────────────────────────
# core/model.py  — EfficientNetB3DualStream (copied verbatim from notebook)
# ─────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import timm
from core.config import MODEL_PATH


class EfficientNetB3DualStream(nn.Module):
    """
    Visual Stream   : EfficientNet-B3 NoisyStudent → 1536
    Clinical Stream : MLP (6 → 64 → 128)           →  128
    Fusion          : Linear(1664 → 512 → 128 → 2)
    """
    def __init__(self, backbone_name='tf_efficientnet_b3_ns',
                 clinical_dim=6, num_classes=2, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, num_classes=0, in_chans=3,
            drop_rate=0.1, drop_path_rate=0.1,
        )
        feat_dim = self.backbone.num_features  # 1536

        self.clinical_mlp = nn.Sequential(
            nn.Linear(clinical_dim, 64),
            nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim + 128, 512),
            nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, image, clinical):
        v = self.backbone(image)
        c = self.clinical_mlp(clinical)
        return self.classifier(torch.cat([v, c], dim=1))


def load_model(path=MODEL_PATH, device="cpu"):
    """
    Load EfficientNetB3DualStream from a best_state .pth file.
    Returns (model, threshold).
    """
    # weights_only=False: this is our own trusted checkpoint (contains a state_dict
    # plus a float threshold) — PyTorch >= 2.6 defaults to weights_only=True, which
    # rejects the non-tensor entries in best_state.
    state = torch.load(path, map_location=device, weights_only=False)
    model = EfficientNetB3DualStream()
    model.load_state_dict(state["model"])
    model.eval()
    threshold = float(state.get("threshold", 0.5))
    return model, threshold
