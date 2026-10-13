"""Reconstruct a colored PLY or a GLB with learned colors and material assignments."""

import argparse
from pathlib import Path
import warnings

import numpy as np
import torch
import trimesh
from trimesh.visual.material import PBRMaterial

from appearance_data import linear_to_srgb, validate_encoding_domain
from appearance_model import load_checkpoint
from reconstruct import mesh_from_grid, sdf_grid


class _GeometryField(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, coords):
        return self.model(coords)["sdf"]


@torch.no_grad()
def predict_appearance(model, points, meta, device="cpu", batch_size=65536):
    colors, labels = [], []
    for start in range(0, len(points), batch_size):
        coords = (points[start:start + batch_size] - np.asarray(meta["center"])) / meta["scale"]
        pred = model(torch.as_tensor(coords, dtype=torch.float32, device=device))
        colors.append(pred["rgba"].cpu().numpy())
        labels.append(pred["material_logits"].argmax(-1).cpu().numpy())
    return np.concatenate(colors), np.concatenate(labels)


def export_mesh(vertices, faces, rgba, material_ids, materials, path):
    path = Path(path)
    if path.suffix.lower() not in (".glb", ".ply"):
        raise ValueError("Use .glb to preserve materials, or .ply for vertex colors")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".ply":
        colors = rgba.copy()
        colors[:, :3] = linear_to_srgb(colors[:, :3])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False,
                               vertex_colors=np.round(np.clip(colors, 0, 1) * 255).astype(np.uint8))
        warnings.warn("PLY exports vertex RGBA only; use GLB to retain PBR materials.")
        mesh.export(path)
        return
    scene = trimesh.Scene()
    for i, material in enumerate(materials):
        selected = faces[material_ids == i]
        if not len(selected):
            continue
        indices, inverse = np.unique(selected, return_inverse=True)
        # The field already includes baseColorFactor, texture and vertex color.
        # Keep the exported factor white so it does not tint the prediction twice.
        pbr = PBRMaterial(name=f"{i}_{material['name']}", baseColorFactor=[255, 255, 255, 255],
                          roughnessFactor=material["roughness"], metallicFactor=material["metallic"],
                          emissiveFactor=material["emissive"], alphaMode=material["alpha_mode"],
                          alphaCutoff=material["alpha_cutoff"], doubleSided=material["double_sided"])
        visual = trimesh.visual.TextureVisuals(material=pbr)
        # glTF COLOR_0 is linear, unlike the sRGB bytes used for PLY.
        visual.vertex_attributes["color"] = np.round(np.clip(rgba[indices], 0, 1) * 255).astype(np.uint8)
        part = trimesh.Trimesh(vertices=vertices[indices], faces=inverse.reshape(-1, 3),
                              visual=visual, process=False)
        scene.add_geometry(part, geom_name=f"material_{i}")
    path.write_bytes(scene.export(file_type="glb"))


def reconstruct(weights, out="reconstruction.glb", resolution=128, batch_size=65536, device="auto"):
    if resolution < 3 or batch_size < 1:
        raise ValueError("Resolution must be at least 3 and batch size must be positive")
    if Path(out).suffix.lower() not in (".glb", ".ply"):
        raise ValueError("Output must be .glb or .ply")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_checkpoint(weights, device)
    try:
        validate_encoding_domain(meta, model.config["input_scale"])
    except ValueError as error:
        warnings.warn(str(error))
    grid_meta = {key: np.asarray(meta[key], dtype=np.float32)
                 for key in ("center", "min_bound", "max_bound", "padding")}
    grid_meta["scale"] = float(meta["scale"])
    print(f"Sampling SDF on a {resolution}^3 grid...", flush=True)
    sdf, axes = sdf_grid(_GeometryField(model), grid_meta, resolution, device, batch_size)
    if not np.isfinite(sdf).all() or not sdf.min() < 0 < sdf.max():
        raise ValueError("The learned SDF has no zero crossing in the reconstruction bounds; "
                         "train longer or inspect the input mesh's orientation and closure")
    boundary = np.concatenate([sdf[0].ravel(), sdf[-1].ravel(), sdf[:, 0].ravel(),
                               sdf[:, -1].ravel(), sdf[:, :, 0].ravel(), sdf[:, :, -1].ravel()])
    if np.any(boundary < 0):
        warnings.warn(f"SDF is negative on {np.mean(boundary < 0):.1%} of the reconstruction "
                      "boundary; marching cubes padding will create artificial caps. "
                      "Check the Fourier scale and exterior supervision before using this mesh.")
    vertices, faces = mesh_from_grid(sdf, axes)
    rgba, _ = predict_appearance(model, vertices, meta, device, batch_size)
    # Classify face centroids, then split primitives by material for GLB export.
    _, face_materials = predict_appearance(model, vertices[faces].mean(axis=1), meta, device, batch_size)
    export_mesh(vertices, faces, rgba, face_materials, meta["materials"], out)
    print(f"Saved {len(vertices):,} vertices and {len(faces):,} triangles to {out}", flush=True)
    return vertices, faces


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", default="reconstruction.glb")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--device", default="auto")
    reconstruct(**vars(parser.parse_args()))
