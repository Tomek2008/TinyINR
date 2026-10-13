"""Joint SDF, linear RGBA and material classification training, including INT8 QAT."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from appearance_data import (load_appearance_mesh, make_appearance_dataset, make_geometry_dataset,
                             normalized_bounds, validate_encoding_domain)
from appearance_model import AppearanceKAN, load_checkpoint, save_checkpoint
from train import ste_quantize


def _predict(model, coords, qat=False):
    if not qat:
        return model(coords)
    state = {name: ste_quantize(value) for name, value in model.named_parameters()}
    state.update(dict(model.named_buffers()))
    return functional_call(model, state, (coords,))


def _sdf_loss(pred, target, clamp):
    if clamp > 0:
        pred, target = pred.clamp(-clamp, clamp), target.clamp(-clamp, clamp)
    return F.mse_loss(pred, target)


@torch.no_grad()
def evaluate(model, geometry_loader, appearance_loader, device, qat=False, clamp=0.0):
    model.eval()
    sdf_sum = sdf_objective = color_sum = rgb_sum = correct = 0.0
    n_geometry = n_color = 0
    for coords, target in geometry_loader:
        coords, target = coords.to(device), target.to(device)
        pred = _predict(model, coords, qat)["sdf"]
        sdf_sum += F.mse_loss(pred, target).item() * len(coords)
        sdf_objective += _sdf_loss(pred, target, clamp).item() * len(coords)
        n_geometry += len(coords)
    material_sum = 0.0
    for coords, rgba, labels in appearance_loader:
        coords, rgba, labels = coords.to(device), rgba.to(device), labels.to(device)
        pred = _predict(model, coords, qat)
        color_sum += F.mse_loss(pred["rgba"], rgba).item() * len(coords)
        rgb_sum += F.mse_loss(pred["rgba"][:, :3], rgba[:, :3]).item() * len(coords)
        material_sum += F.cross_entropy(pred["material_logits"], labels).item() * len(coords)
        correct += (pred["material_logits"].argmax(-1) == labels).sum().item()
        n_color += len(coords)
    rgb_mse = rgb_sum / n_color
    return {"sdf_mse": sdf_sum / n_geometry, "sdf_objective": sdf_objective / n_geometry,
            "rgba_mse": color_sum / n_color, "rgb_psnr": -10 * math.log10(max(rgb_mse, 1e-12)),
            "material_ce": material_sum / n_color, "material_accuracy": correct / n_color}


def _split_loaders(dataset, batch_size, seed):
    if len(dataset) < 2:
        raise ValueError("Need at least two samples for training and validation")
    validation_count = max(1, len(dataset) // 10)
    train, validation = random_split(dataset, [len(dataset) - validation_count, validation_count],
                                     generator=torch.Generator().manual_seed(seed))
    return (DataLoader(train, batch_size=batch_size, shuffle=True,
                       generator=torch.Generator().manual_seed(seed)),
            DataLoader(validation, batch_size=batch_size))


@torch.no_grad()
def boundary_diagnostics(model, meta, device="cpu", resolution=32):
    """Check exterior sign independently of the random training/validation split."""
    low, high = normalized_bounds(meta)
    axes = [np.linspace(low[i], high[i], resolution, dtype=np.float32) for i in range(3)]
    panels = []
    for fixed in range(3):
        other = [i for i in range(3) if i != fixed]
        uv = np.stack(np.meshgrid(axes[other[0]], axes[other[1]], indexing="ij"), axis=-1).reshape(-1, 2)
        for side in (low[fixed], high[fixed]):
            points = np.empty((len(uv), 3), dtype=np.float32)
            points[:, fixed], points[:, other] = side, uv
            panels.append(points)
    points = torch.from_numpy(np.concatenate(panels)).to(device)
    values = torch.cat([model(batch)["sdf"] for batch in points.split(4096)])
    return {"boundary_min_sdf": values.min().item(),
            "boundary_negative_fraction": (values < 0).float().mean().item()}


def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def fit(model, loaders, epochs, lr, device, color_weight=0.1, material_weight=0.01,
        eikonal=0.0, clamp=0.0, qat=False, patience=30):
    geometry_train, geometry_val, color_train, color_val = loaders
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
    best_loss, best_state, stalled = float("inf"), None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        geometry_iter, color_iter = iter(geometry_train), iter(color_train)
        for _ in range(max(len(geometry_train), len(color_train))):
            geometry_batch, geometry_iter = _next_batch(geometry_iter, geometry_train)
            color_batch, color_iter = _next_batch(color_iter, color_train)
            coords, sdf = (value.to(device) for value in geometry_batch)
            surface, rgba, labels = (value.to(device) for value in color_batch)
            coords.requires_grad_(eikonal > 0)
            optimizer.zero_grad(set_to_none=True)
            geometry = _predict(model, coords, qat)["sdf"]
            appearance = _predict(model, surface, qat)
            loss = (_sdf_loss(geometry, sdf, clamp)
                    + color_weight * F.mse_loss(appearance["rgba"], rgba)
                    + material_weight * F.cross_entropy(appearance["material_logits"], labels))
            if eikonal > 0:
                gradient = torch.autograd.grad(geometry.sum(), coords, create_graph=True)[0]
                penalty = (gradient.norm(dim=-1) - 1).square()
                mask = (sdf[:, 0].abs() < clamp) if clamp > 0 else torch.ones_like(penalty)
                loss = loss + eikonal * (penalty * mask).sum() / mask.sum().clamp_min(1)
            if not torch.isfinite(loss):
                raise RuntimeError("Training loss is non-finite; check the mesh and learning rate")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        metrics = evaluate(model, geometry_val, color_val, device, qat=qat, clamp=clamp)
        objective = (metrics["sdf_objective"] + color_weight * metrics["rgba_mse"]
                     + material_weight * metrics["material_ce"])
        if not math.isfinite(objective):
            raise RuntimeError("Validation loss is non-finite")
        history.append({"epoch": epoch, "loss": objective, **metrics})
        if objective < best_loss:
            best_loss, stalled = objective, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stalled += 1
        scheduler.step(objective)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"{'QAT' if qat else 'FP32'} {epoch:4d}  val SDF {metrics['sdf_mse']:.6f}  "
                  f"RGB {metrics['rgb_psnr']:.2f} dB  material {metrics['material_accuracy']:.1%}", flush=True)
        if stalled >= patience:
            break
    model.load_state_dict(best_state)
    return {"epochs": len(history), "best_loss": best_loss, "history": history}


def train(mesh_path, out_dir="weights/appearance", hidden=128, hidden_layers=4,
          epochs=200, qat_epochs=100, samples=500_000, appearance_samples=200_000,
          batch_size=4096, color_weight=0.1, material_weight=0.01, eikonal=0.0,
          clamp=0.0, band=False, omega_0=30.0, input_scale=1.0, lr=2e-4,
          qat_lr=1e-5, seed=0, device="auto", patience=30):
    if epochs < 1 or qat_epochs < 0 or batch_size < 1 or samples < 10 or appearance_samples < 2 or patience < 1:
        raise ValueError("Invalid epoch, batch, sample count or patience")
    if any(not math.isfinite(x) or x <= 0 for x in (lr, qat_lr, color_weight, material_weight)):
        raise ValueError("Learning rates and appearance loss weights must be finite and positive")
    if any(not math.isfinite(x) or x < 0 for x in (eikonal, clamp)):
        raise ValueError("Eikonal weight and clamp must be finite and non-negative")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    mesh = load_appearance_mesh(mesh_path)
    validate_encoding_domain(mesh.meta, input_scale)
    geometry = make_geometry_dataset(mesh, samples, seed=seed, band=band)
    appearance = make_appearance_dataset(mesh, appearance_samples, seed=seed + 1)
    loaders = (*_split_loaders(geometry, batch_size, seed),
               *_split_loaders(appearance, batch_size, seed + 1))
    model = AppearanceKAN(len(mesh.materials), hidden, hidden_layers, omega_0, input_scale).to(device)
    print(f"Mesh: {mesh_path} | Materials: {len(mesh.materials)} | "
          f"Parameters: {sum(p.numel() for p in model.parameters()):,} | Device: {device}", flush=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(mesh_path).stem
    meta = {**mesh.meta, "seed": seed, "samples": samples, "appearance_samples": appearance_samples,
            "geometry_sampling": "reconstruction-domain-with-boundary-v1",
            "batch_size": batch_size, "lr": lr, "qat_lr": qat_lr,
            "loss_weights": {"sdf": 1.0, "rgba": color_weight, "material": material_weight,
                             "eikonal": eikonal}, "clamp": clamp, "band": band}
    settings = dict(color_weight=color_weight, material_weight=material_weight,
                    eikonal=eikonal, clamp=clamp, patience=patience)
    report = {"config": model.config, "meta": meta, "checkpoints": {}}
    report["fp32_fit"] = fit(model, loaders, epochs, lr, device, **settings)

    def save_and_measure(precision):
        path = out / f"{stem}_appearance_{precision}.pth"
        save_checkpoint(model, path, meta, precision)
        recovered, _ = load_checkpoint(path, device)
        metrics = evaluate(recovered, loaders[1], loaders[3], device, clamp=clamp)
        metrics.update(boundary_diagnostics(recovered, meta, device))
        report["checkpoints"][precision] = {"path": path.name, "bytes": path.stat().st_size, **metrics}

    save_and_measure("fp32")
    save_and_measure("fp16")
    if qat_epochs:
        report["qat_fit"] = fit(model, loaders, qat_epochs, qat_lr, device, qat=True, **settings)
        save_and_measure("int8")
    (out / "appearance_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved checkpoints and validation metrics to {out}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, dest="mesh_path")
    parser.add_argument("--out", default="weights/appearance", dest="out_dir")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4, dest="hidden_layers")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--qat-epochs", type=int, default=100)
    parser.add_argument("--samples", type=int, default=500_000)
    parser.add_argument("--appearance-samples", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--color-weight", type=float, default=0.1)
    parser.add_argument("--material-weight", type=float, default=0.01)
    parser.add_argument("--eikonal", type=float, default=0.0)
    parser.add_argument("--clamp", type=float, default=0.0)
    parser.add_argument("--band", action="store_true")
    parser.add_argument("--omega", type=float, default=30.0, dest="omega_0")
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--qat-lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()
    train(**vars(args))
