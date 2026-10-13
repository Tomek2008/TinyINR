from pathlib import Path
import numpy as np
import open3d as o3d
import point_cloud_utils as pcu
import torch


def load_mesh(name="bunny"):
    assets_dir = Path(__file__).parent / "assets"
    if name == "sword":
        path = assets_dir / "sword.obj"
    elif name == "shield":
        path = assets_dir / "shield.obj"
    elif name in ("bunny", "armadillo", "knot"):
        source = {
            "bunny": o3d.data.BunnyMesh,
            "armadillo": o3d.data.ArmadilloMesh,
            "knot": o3d.data.KnotMesh,
        }[name]()
        path = source.path
    else:
        path = Path(name)

    mesh = o3d.io.read_triangle_mesh(str(path))
    if not mesh.has_triangles():
        raise RuntimeError(f"Failed to load mesh from {path}")
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()
    return mesh, str(path)


def make_sdf_dataset(mesh, num_samples=500_000, seed=0, band=False):
    rng = np.random.default_rng(seed)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)

    if band:
        n_surface = int(num_samples * 0.45)
        n_near = int(num_samples * 0.35)
        n_shell = int(num_samples * 0.1)
        n_uniform = num_samples - n_surface - n_near - n_shell
    else:
        n_surface = int(num_samples * 0.6)
        n_near = int(num_samples * 0.2)
        n_shell = 0
        n_uniform = num_samples - n_surface - n_near

    surface = np.asarray(mesh.sample_points_uniformly(n_surface).points, dtype=np.float32)
    surface = surface + rng.normal(scale=1e-4 if band else 1e-5, size=surface.shape).astype(np.float32)

    near = np.asarray(mesh.sample_points_uniformly(n_near).points, dtype=np.float32)
    near = near + rng.normal(scale=0.004 if band else 0.005, size=near.shape).astype(np.float32)
    if n_shell:
        shell = np.asarray(mesh.sample_points_uniformly(n_shell).points, dtype=np.float32)
        shell = shell + rng.normal(scale=0.02, size=shell.shape).astype(np.float32)
        near = np.vstack([near, shell])

    bbox = mesh.get_axis_aligned_bounding_box()
    min_bound = np.asarray(bbox.min_bound, dtype=np.float32)
    max_bound = np.asarray(bbox.max_bound, dtype=np.float32)
    center = (min_bound + max_bound) / 2.0
    scale = float(np.max(max_bound - min_bound)) / 2.0
    padding = (max_bound - min_bound) * 0.1

    uniform = rng.uniform(center - scale * 1.15, center + scale * 1.15, size=(n_uniform, 3)).astype(np.float32)

    points = np.vstack([surface, near, uniform])

    sdf, _, _ = pcu.signed_distance_to_mesh(points, verts, faces)

    norm_points = (points - center) / scale
    norm_sdf = sdf / scale

    meta = {
        "center": center.tolist(),
        "scale": scale,
        "min_bound": min_bound.tolist(),
        "max_bound": max_bound.tolist(),
        "padding": padding.tolist(),
    }

    return (
        torch.from_numpy(norm_points).float(),
        torch.from_numpy(norm_sdf[:, None]).float(),
        meta,
    )
