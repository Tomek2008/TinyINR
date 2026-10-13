import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from skimage import measure
import torch

from kan import KAN
from train import dequantize_state


def load_model(path, device="cuda"):
    blob = torch.load(path, map_location="cpu", weights_only=True)
    state = dequantize_state(blob)
    indices = sorted({int(key.split(".")[0]) for key in state})
    final = indices[-1]
    num_fourier = sum(f"{i}.fourier_coeffs" in state for i in indices if i != final)
    coeff = state["0.fourier_coeffs"]
    hidden, in_features, grid_size = coeff.shape[1], coeff.shape[2], coeff.shape[3]
    model = KAN(
        in_features=in_features,
        hidden_features=hidden,
        hidden_layers=final - 1,
        out_features=state[f"{final}.weight"].shape[0],
        num_fourier_layers=num_fourier,
        grid_size=grid_size,
        input_scale=3.0,
        omega_0=30.0,
    )
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def load_meta(path):
    raw = json.loads(Path(path).read_text())
    meta = {
        key: np.asarray(raw[key], dtype=np.float32)
        for key in ("center", "min_bound", "max_bound", "padding")
    }
    meta["scale"] = np.float32(raw["scale"])
    return meta


def sdf_grid(model, meta, resolution=128, device="cuda", batch_size=65536):
    low = meta["min_bound"] - meta["padding"]
    high = meta["max_bound"] + meta["padding"]
    axes = [np.linspace(low[i], high[i], resolution, dtype=np.float32) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    norm_grid = (grid - meta["center"]) / meta["scale"]
    values = []
    with torch.no_grad():
        for start in range(0, len(norm_grid), batch_size):
            batch = torch.from_numpy(norm_grid[start : start + batch_size]).float().to(device)
            values.append(model(batch).cpu().numpy())

    sdf = np.vstack(values).reshape(resolution, resolution, resolution) * meta["scale"]
    return sdf, axes


def mesh_from_grid(sdf, axes):
    step = [axes[i][1] - axes[i][0] for i in range(3)]
    padded_sdf = np.pad(sdf, 1, mode="constant", constant_values=1.0)
    verts, faces, _, _ = measure.marching_cubes(
        padded_sdf,
        level=0.0,
        spacing=tuple(step),
    )
    verts = verts + np.array([axes[0][0] - step[0], axes[1][0] - step[1], axes[2][0] - step[2]])
    return verts, faces


def save_obj(verts, faces, path):
    lines = [f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n" for v in verts]
    lines.extend(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n" for f in faces)
    Path(path).write_text("".join(lines))


def evaluate_chamfer(mesh_gt, verts_pred, faces_pred, num_samples=100_000):
    import open3d as o3d

    pred_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(verts_pred, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(faces_pred, dtype=np.int32)),
    )
    p_gt = np.asarray(mesh_gt.sample_points_uniformly(num_samples).points, dtype=np.float64)
    p_pred = np.asarray(pred_mesh.sample_points_uniformly(num_samples).points, dtype=np.float64)

    fwd = cKDTree(p_pred).query(p_gt)[0]
    bwd = cKDTree(p_gt).query(p_pred)[0]
    chamfer_mean = float(np.concatenate([fwd, bwd]).mean())
    return chamfer_mean


if __name__ == "__main__":
    from data import load_mesh

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--meta", default=None)
    parser.add_argument("--mesh", default="bunny")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--out", default="reconstruction.obj")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta_path = args.meta or str(Path(args.weights).parent / "meta.json")

    meta = load_meta(meta_path)
    model = load_model(args.weights, device=device)

    print(f"Sampling SDF on {args.resolution}^3 grid...")
    sdf, axes = sdf_grid(model, meta, resolution=args.resolution, device=device)

    verts, faces = mesh_from_grid(sdf, axes)
    save_obj(verts, faces, args.out)
    print(f"Saved reconstructed mesh ({len(verts):,} vertices, {len(faces):,} faces) to {args.out}")

    gt_mesh, _ = load_mesh(args.mesh)
    chamfer = evaluate_chamfer(gt_mesh, verts, faces)
    print(f"Symmetric Chamfer Distance (100k samples): {chamfer:.6f}")
