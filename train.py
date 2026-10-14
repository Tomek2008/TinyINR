import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call
from torch.utils.data import DataLoader, TensorDataset

from kan import KAN


def build_model(hidden=128, hidden_layers=4, omega_0=30.0, input_scale=3.0):
    return KAN(
        in_features=3,
        hidden_features=hidden,
        hidden_layers=hidden_layers,
        out_features=1,
        num_fourier_layers=1,
        grid_size=8,
        input_scale=input_scale,
        omega_0=omega_0,
    )


def save_weights(model, path, dtype):
    state = {key: value.detach().cpu().to(dtype) for key, value in model.state_dict().items()}
    torch.save(state, path)
    return path.stat().st_size


def quantize_tensor(tensor):
    tensor = tensor.detach().float().cpu()
    if tensor.ndim == 4:
        scales = tensor.abs().amax(dim=(0, 2, 3), keepdim=True) / 127.0
    elif tensor.ndim == 2 and tensor.shape[0] > 1:
        scales = tensor.abs().amax(dim=1, keepdim=True) / 127.0
    else:
        scales = torch.tensor([tensor.abs().max() / 127.0])
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    quantized = torch.clamp(torch.round(tensor / scales), -127, 127).to(torch.int8)
    return quantized, scales


def ste_quantize(weight):
    if weight.ndim == 4:
        scales = weight.abs().amax(dim=(0, 2, 3), keepdim=True) / 127.0
    elif weight.ndim == 2 and weight.shape[0] > 1:
        scales = weight.abs().amax(dim=1, keepdim=True) / 127.0
    else:
        scales = weight.abs().max().unsqueeze(0) / 127.0
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    quantized = torch.clamp(torch.round(weight / scales), -127, 127)
    dequant = quantized * scales
    return (dequant - weight).detach() + weight


def dequantize_state(state):
    recovered = {}
    for key, val in state.items():
        if isinstance(val, tuple):
            q, s = val
            recovered[key] = (q.float() * s).float()
        else:
            recovered[key] = val.float()
    return recovered


def save_int8(model, path):
    quantized_state = {}
    for key, value in model.state_dict().items():
        if "weight" in key or "fourier_coeffs" in key or "bias" in key:
            q, s = quantize_tensor(value)
            quantized_state[key] = (q, s.float())
        else:
            quantized_state[key] = value.detach().cpu()
    torch.save(quantized_state, path)
    return path.stat().st_size


def _epoch_loss(model, loader, optimizer, device, clip_grad=1.0, is_qat=False, eikonal=0.0, clamp=0.0):
    model.train()
    total_loss = 0.0
    count = 0
    names = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    for coords, target in loader:
        coords, target = coords.to(device), target.to(device)
        optimizer.zero_grad()

        if is_qat:
            ste_params = {k: ste_quantize(v) for k, v in names.items()}
            ste_params.update(buffers)
            pred = functional_call(model, ste_params, coords)
            loss = torch.nn.functional.mse_loss(pred, target)
        elif eikonal > 0 or clamp > 0:
            if eikonal > 0:
                coords = coords.detach().requires_grad_(True)
            pred = model(coords)
            if clamp > 0:
                data = torch.nn.functional.mse_loss(pred.clamp(-clamp, clamp), target.clamp(-clamp, clamp))
            else:
                surface = 1.0 + 4.0 * torch.exp(-target.abs() * 40.0)
                data = (surface * (pred - target).square()).mean()
            loss = data
            if eikonal > 0:
                grad = torch.autograd.grad(pred, coords, torch.ones_like(pred), create_graph=True)[0]
                deviation = (grad.norm(dim=-1) - 1.0).square()
                if clamp > 0:
                    mask = target.abs().squeeze(-1) < clamp
                    penalty = (deviation * mask).sum() / mask.sum().clamp(min=1.0)
                else:
                    penalty = deviation.mean()
                loss = loss + eikonal * penalty
        else:
            pred = model(coords)
            loss = torch.nn.functional.mse_loss(pred, target)

        loss.backward()

        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()
        total_loss += loss.item() * len(coords)
        count += len(coords)

    return total_loss / count


def _fit(model, loader, epochs, lr, is_qat=False, stop_patience=20, eikonal=0.0, clamp=0.0):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=25,
        threshold=1e-4,
        threshold_mode="rel",
        min_lr=1e-7,
    )

    best_loss = float("inf")
    best_state = None
    stall_count = 0
    hit_plateau = False

    for epoch in range(epochs):
        loss = _epoch_loss(
            model,
            loader,
            optimizer,
            device,
            clip_grad=0.25 if clamp > 0 and not is_qat else 1.0,
            is_qat=is_qat,
            eikonal=eikonal,
            clamp=0.0 if is_qat else clamp,
        )
        current_lr = optimizer.param_groups[0]["lr"]

        if loss < best_loss * (1.0 - 1e-4):
            best_loss = loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stall_count = 0
        else:
            stall_count += 1

        scheduler.step(loss)

        if epoch % 10 == 0 or epoch == epochs - 1:
            tag = "qat  " if is_qat else "epoch"
            print(f"{tag} {epoch:3d}  mse {loss:.6f}  best {best_loss:.6f}  lr {current_lr:.1e}")

        if math.isclose(current_lr, 1e-7, rel_tol=1e-3) and stall_count >= stop_patience:
            hit_plateau = True
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return {"epochs": epoch + 1, "plateau": hit_plateau, "best_mse": best_loss}


def train(
    epochs=200,
    batch_size=8192,
    num_samples=500_000,
    seed=0,
    out_dir="weights",
    hidden=128,
    hidden_layers=4,
    mesh_name="bunny",
    qat_epochs=100,
    stop_patience=20,
    eikonal=0.0,
    clamp=0.0,
    band=False,
    omega_0=30.0,
    input_scale=3.0,
):
    from data import load_mesh, make_sdf_dataset

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    mesh, ply_path = load_mesh(mesh_name)
    ply_bytes = Path(ply_path).stat().st_size

    coords, sdfs, meta = make_sdf_dataset(mesh, num_samples=num_samples, seed=seed, band=band)
    loader = DataLoader(
        TensorDataset(coords, sdfs),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(
        hidden=hidden,
        hidden_layers=hidden_layers,
        omega_0=omega_0,
        input_scale=input_scale,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Mesh: {mesh_name} | Hidden: {hidden} | Layers: {hidden_layers} | Params: {total_params:,} | Device: {device}")

    fp32_fit = _fit(
        model,
        loader,
        epochs=epochs,
        lr=4e-5 if (clamp > 0 or eikonal > 0) else 2e-4,
        is_qat=False,
        stop_patience=stop_patience,
        eikonal=eikonal,
        clamp=clamp,
    )

    fp32_bytes = save_weights(model, out / f"{mesh_name}_fp32.pth", torch.float32)
    fp16_bytes = save_weights(model, out / f"{mesh_name}_fp16.pth", torch.float16)

    if qat_epochs > 0:
        qat_fit = _fit(model, loader, epochs=qat_epochs, lr=1e-5, is_qat=True, stop_patience=stop_patience)
        int8_bytes = save_int8(model, out / f"{mesh_name}_int8.pth")
    else:
        qat_fit = {"epochs": 0, "plateau": False}
        int8_bytes = 0

    meta.update(
        {
            "ply_bytes": ply_bytes,
            "weight_bytes": {"fp32": fp32_bytes, "fp16": fp16_bytes, "int8": int8_bytes},
            "parameters": total_params,
            "hidden": hidden,
            "hidden_layers": hidden_layers,
            "fp32_epochs": fp32_fit["epochs"],
            "fp32_plateau": fp32_fit["plateau"],
            "qat_epochs": qat_fit["epochs"],
            "qat_plateau": qat_fit["plateau"],
        }
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Results for {mesh_name}:")
    print(f"  PLY:  {ply_bytes / 1024:.1f} KB")
    print(f"  FP32: {fp32_bytes / 1024:.1f} KB ({ply_bytes / max(fp32_bytes, 1):.1f}x compression)")
    print(f"  FP16: {fp16_bytes / 1024:.1f} KB ({ply_bytes / max(fp16_bytes, 1):.1f}x compression)")
    if int8_bytes:
        print(f"  INT8: {int8_bytes / 1024:.1f} KB ({ply_bytes / int8_bytes:.1f}x compression)")
    print(f"Saved checkpoints to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", default="bunny")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--qat-epochs", type=int, default=100)
    parser.add_argument("--samples", type=int, default=500_000)
    parser.add_argument("--eikonal", type=float, default=0.0)
    parser.add_argument("--clamp", type=float, default=0.0)
    parser.add_argument("--band", action="store_true")
    parser.add_argument("--omega", type=float, default=30.0)
    parser.add_argument("--input-scale", type=float, default=3.0)
    parser.add_argument("--out", default="weights")
    args = parser.parse_args()

    train(
        mesh_name=args.mesh,
        hidden=args.hidden,
        hidden_layers=args.layers,
        epochs=args.epochs,
        qat_epochs=args.qat_epochs,
        num_samples=args.samples,
        eikonal=args.eikonal,
        clamp=args.clamp,
        band=args.band,
        omega_0=args.omega,
        input_scale=args.input_scale,
        out_dir=args.out,
    )
