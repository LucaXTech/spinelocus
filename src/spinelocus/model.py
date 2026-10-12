"""HRNet heatmaps; supports original W18/head checkpoint names."""

import torch
from torch import nn
from torch.nn import functional as F


class HRNetHeatmap(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.cfg = dict(cfg)
        if cfg["model_name"] == "smoke_cnn":
            # Offline numerical tests only. Never a research baseline.
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 8, 3, padding=1, stride=2),
                nn.ReLU(),
                nn.Conv2d(8, 16, 3, padding=1, stride=2),
                nn.ReLU(),
            )
            channels = 16
        else:
            import timm

            index = int(cfg.get("feature_index", 1))
            self.backbone = timm.create_model(
                cfg["model_name"], pretrained=pretrained, features_only=True, out_indices=(index,)
            )
            channels = self.backbone.feature_info.channels()[0]
        self.head = nn.Sequential(
            nn.Conv2d(channels, 128, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        if isinstance(features, (tuple, list)):
            features = features[0]
        return F.interpolate(
            self.head(features), (self.cfg["heatmap_size"],) * 2, mode="bilinear", align_corners=False
        )


def decode(logits, crop_size, temperature=0.35, convention="pixel_centers"):
    b, _, h, w = logits.shape
    if temperature <= 0:
        raise ValueError("Decode temperature must be positive")
    probs = (logits.float().flatten(1) / temperature).softmax(1).reshape(b, h, w)
    xs = torch.arange(w, device=logits.device, dtype=torch.float32)
    ys = torch.arange(h, device=logits.device, dtype=torch.float32)
    x = (probs * xs[None, None, :]).sum((1, 2))
    y = (probs * ys[None, :, None]).sum((1, 2))
    xy = torch.stack([x, y], dim=1)
    offset = 0.5 if convention == "pixel_centers" else 0.0
    return (xy + offset) * xy.new_tensor([crop_size / w, crop_size / h]) - offset


def heatmap_target(xy, cfg):
    h, c = int(cfg["heatmap_size"]), int(cfg["crop_size"])
    offset = 0.5 if cfg.get("coordinate_convention", "pixel_centers") == "pixel_centers" else 0.0
    centers = (xy.float() + offset) * h / c - offset
    xs = torch.arange(h, device=xy.device).float()[None, None, :]
    ys = torch.arange(h, device=xy.device).float()[None, :, None]
    distance = (xs - centers[:, 0, None, None]) ** 2 + (ys - centers[:, 1, None, None]) ** 2
    hm = torch.exp(-distance / (2 * float(cfg["sigma"]) ** 2))
    return hm[:, None]


def loss(logits, xy, cfg):
    target = heatmap_target(xy, cfg).flatten(1)
    target = target / target.sum(1, keepdim=True).clamp_min(1e-12)
    log_probs = F.log_softmax(logits.float().flatten(1) / cfg.get("temperature_ce", 0.5), dim=1)
    ce = -(target * log_probs).sum(1).mean()
    pred = decode(
        logits,
        cfg["crop_size"],
        cfg.get("temperature_decode", 0.35),
        cfg.get("coordinate_convention", "pixel_centers"),
    )
    coordinate = torch.abs(pred - xy).sum(1).mean()
    return ce + cfg.get("lambda_l1", 0.01) * coordinate


def load_model(path, device="cpu"):
    # Only open checkpoints from trusted sources. Training states contain RNG metadata.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "config" in checkpoint:
        cfg = checkpoint["config"]
    else:
        old = checkpoint.get("cfg", {})
        cfg = {
            "model_name": old.get("MODEL_NAME", "hrnet_w18"),
            "feature_index": 3,
            "crop_size": old.get("CROP_SIZE", 160),
            "heatmap_size": old.get("HEATMAP_SIZE", 64),
            "sigma": old.get("SIGMA", 1.5),
            "temperature_decode": 0.35,
            "coordinate_convention": "legacy_scale",
        }
    model = HRNetHeatmap(cfg, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, cfg
