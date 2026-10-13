import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import scipy.ndimage as ndi
import torch

from reconstruct import load_model, mesh_from_grid


def yaw_matrix(degrees):
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def sample_field(model, resolution=128, bound=1.05, device="cuda", batch_size=65536, rotation=None):
    axis = np.linspace(-bound, bound, resolution, dtype=np.float32)
    coords = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    if rotation is not None:
        coords = np.ascontiguousarray(coords @ rotation, dtype=np.float32)

    values = []
    with torch.no_grad():
        for start in range(0, len(coords), batch_size):
            batch = torch.from_numpy(coords[start : start + batch_size]).float().to(device)
            values.append(model(batch).cpu().numpy())

    return np.vstack(values).reshape(resolution, resolution, resolution)


def solid_half(sdf, bound):
    mask = sdf <= 0.0
    if not np.any(mask):
        mask = sdf <= np.percentile(sdf, 2)
    n = sdf.shape[0]
    half = []
    for axis_ids in np.nonzero(mask):
        coord = (axis_ids / (n - 1)) * (2.0 * bound) - bound
        half.append(float(np.percentile(np.abs(coord), 99)))
    return np.maximum(np.asarray(half, dtype=np.float32), 0.03)


def scale_field(grid, src_half, dst_half, bound):
    n = grid.shape[0]
    axis = np.linspace(-bound, bound, n, dtype=np.float64)
    ratio = np.maximum(src_half, 1e-3) / np.maximum(dst_half, 1e-3)
    index = []
    for i in range(3):
        coord = axis * ratio[i]
        index.append((coord + bound) / (2.0 * bound) * (n - 1))
    xx, yy, zz = np.meshgrid(index[0], index[1], index[2], indexing="ij")
    return ndi.map_coordinates(grid, [xx, yy, zz], order=1, mode="constant", cval=bound)


def morph_models(
    model_a_path,
    model_b_path,
    out_mp4="morph.mp4",
    steps=90,
    fps=30,
    resolution=128,
    bound=1.05,
    yaw_b=0.0,
    smooth_sigma=1.0,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_a = load_model(model_a_path, device=device)
    model_b = load_model(model_b_path, device=device)

    rot_b = yaw_matrix(yaw_b) if yaw_b != 0 else None

    print(f"Sampling source model SDF...")
    grid_a = sample_field(model_a, resolution=resolution, bound=bound, device=device)
    print(f"Sampling target model SDF (yaw: {yaw_b} deg)...")
    grid_b = sample_field(model_b, resolution=resolution, bound=bound, device=device, rotation=rot_b)

    axis = np.linspace(-bound, bound, resolution, dtype=np.float32)
    axes = [axis, axis, axis]
    half_a = solid_half(grid_a, bound)
    half_b = solid_half(grid_b, bound)
    print(f"Source extent {half_a.round(3).tolist()} -> target extent {half_b.round(3).tolist()}")

    print(f"Rendering {steps} transition frames ({steps / fps:.1f}s at {fps} fps)...")
    renderer = o3d.visualization.rendering.OffscreenRenderer(1280, 720)
    renderer.scene.set_background([0.071, 0.078, 0.094, 1.0])
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.base_color = [0.80, 0.82, 0.88, 1.0]
    center = np.zeros(3)
    eye = np.array([1.85, 0.95, 2.05])

    images = []
    for step in range(steps):
        linear = step / (steps - 1)
        alpha = linear * linear * (3.0 - 2.0 * linear)

        mid = (1.0 - alpha) * half_a + alpha * half_b
        warped_a = scale_field(grid_a, half_a, mid, bound)
        warped_b = scale_field(grid_b, half_b, mid, bound)
        band = 0.12
        blended_sdf = (1.0 - alpha) * np.clip(warped_a, -band, band) + alpha * np.clip(warped_b, -band, band)
        if smooth_sigma > 0:
            blended_sdf = ndi.gaussian_filter(blended_sdf, sigma=smooth_sigma)

        verts, faces = mesh_from_grid(blended_sdf, axes)

        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(verts, dtype=np.float64)),
            o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32)),
        )
        mesh.compute_vertex_normals()
        triangle_ids, counts, _ = mesh.cluster_connected_triangles()
        counts = np.asarray(counts)
        triangle_ids = np.asarray(triangle_ids)
        if len(counts):
            cutoff = max(int(counts.max() * 0.12), 1)
            mesh.remove_triangles_by_mask(counts[triangle_ids] < cutoff)
            mesh.remove_unreferenced_vertices()
            mesh.compute_vertex_normals()
        renderer.scene.clear_geometry()
        renderer.scene.add_geometry("shape", mesh, material)
        renderer.setup_camera(48.0, center, eye, [0.0, 1.0, 0.0])

        images.append(np.asarray(renderer.render_to_image()))

    import subprocess

    height, width = images[0].shape[:2]
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(out_mp4),
        ],
        input=b"".join(frame.tobytes() for frame in images),
        check=True,
    )
    print(f"Exported continuous morph video to {out_mp4}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--out", default="morph.mp4")
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--bound", type=float, default=1.05)
    parser.add_argument("--yaw-b", type=float, default=0.0)
    parser.add_argument("--smooth", type=float, default=1.0)
    args = parser.parse_args()

    morph_models(
        args.model_a,
        args.model_b,
        out_mp4=args.out,
        steps=args.steps,
        fps=args.fps,
        resolution=args.resolution,
        bound=args.bound,
        yaw_b=args.yaw_b,
        smooth_sigma=args.smooth,
    )
