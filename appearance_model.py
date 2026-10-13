"""A shared Fourier-KAN field with geometry, color and material heads."""

import torch
from torch import nn

from kan import KAN
from train import dequantize_state, quantize_tensor


class AppearanceKAN(nn.Module):
    def __init__(self, num_materials, hidden=128, hidden_layers=4,
                 omega_0=30.0, input_scale=1.0):
        super().__init__()
        if num_materials < 1 or hidden < 1 or hidden_layers < 0:
            raise ValueError("Invalid material count or network dimensions")
        if omega_0 <= 0 or input_scale <= 0:
            raise ValueError("omega_0 and input_scale must be positive")
        self.config = dict(num_materials=num_materials, hidden=hidden,
                           hidden_layers=hidden_layers, omega_0=omega_0,
                           input_scale=input_scale)
        backbone = KAN(hidden_features=hidden, hidden_layers=hidden_layers,
                       omega_0=omega_0, input_scale=input_scale)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.sdf_head = list(backbone.children())[-1]
        self.color_head = nn.Linear(hidden, 4)
        self.material_head = nn.Linear(hidden, num_materials)

    def forward(self, coords):
        features = self.backbone(coords)
        return {"sdf": self.sdf_head(features),
                "rgba": self.color_head(features).sigmoid(),
                "material_logits": self.material_head(features)}


def save_checkpoint(model, path, meta, precision="fp32"):
    """Each checkpoint includes normalization, material palette and architecture."""
    if precision not in ("fp32", "fp16", "int8"):
        raise ValueError(f"Unknown precision: {precision}")
    parameters = dict(model.named_parameters())
    state = {}
    for name, tensor in model.state_dict().items():
        if precision == "int8" and name in parameters:
            state[name] = quantize_tensor(tensor)
        else:
            dtype = torch.float16 if precision == "fp16" else torch.float32
            state[name] = tensor.detach().cpu().to(dtype)
    torch.save({"format": "tinyinr-appearance-v1", "precision": precision,
                "config": model.config, "meta": meta, "state_dict": state}, path)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "tinyinr-appearance-v1":
        raise ValueError("Expected a checkpoint from train_appearance.py")
    model = AppearanceKAN(**checkpoint["config"])
    model.load_state_dict(dequantize_state(checkpoint["state_dict"]), strict=True)
    return model.to(device).eval(), checkpoint["meta"]
