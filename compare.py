import argparse
from pathlib import Path

import DracoPy
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import torch

from data import load_mesh
from reconstruct import evaluate_chamfer, load_meta, load_model, mesh_from_grid, sdf_grid


def draco_benchmark(mesh, bits_list=(1, 2, 3, 4, 5, 6, 8, 9, 11, 15), level=10):
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.uint32)

    results = []
    for bits in bits_list:
        encoded = DracoPy.encode(
            verts,
            faces,
            quantization_bits=bits,
            compression_level=level,
        )
        size_kb = len(encoded) / 1024.0
        decoded = DracoPy.decode(encoded)

        dec_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(decoded.points, dtype=np.float64)),
            o3d.utility.Vector3iVector(np.asarray(decoded.faces, dtype=np.int32)),
        )

        chamfer = evaluate_chamfer(mesh, np.asarray(dec_mesh.vertices), np.asarray(dec_mesh.triangles))
        results.append((f"Draco ({bits}-bit)", size_kb, chamfer))

    return results


def inr_benchmark(weights_map, mesh_name="bunny"):
    gt_mesh, _ = load_mesh(mesh_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = []
    for name, pth_path, meta_path in weights_map:
        size_kb = Path(pth_path).stat().st_size / 1024.0
        meta = load_meta(meta_path)
        model = load_model(pth_path, device=device)
        sdf, axes = sdf_grid(model, meta, resolution=128, device=device)
        verts, faces = mesh_from_grid(sdf, axes)
        chamfer = evaluate_chamfer(gt_mesh, verts, faces)
        results.append((name, size_kb, chamfer))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", default="bunny", choices=["bunny", "armadillo", "knot"])
    args = parser.parse_args()

    mesh, ply_path = load_mesh(args.mesh)
    ply_kb = Path(ply_path).stat().st_size / 1024.0
    print(f"Original Mesh: {args.mesh} (PLY Size: {ply_kb:.1f} KB)\n")

    print(f"{'Method':<25} | {'Size (KB)':<10} | {'Chamfer Error':<15}")
    print("-" * 56)

    draco_results = draco_benchmark(mesh)
    for name, size, err in draco_results:
        print(f"{name:<25} | {size:<10.1f} | {err:<15.6f}")
