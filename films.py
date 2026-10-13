import argparse
import subprocess
from pathlib import Path

import numpy as np
import open3d as o3d

from data import load_mesh
from morph import yaw_matrix


def fit_mesh(mesh, height=1.45):
    bounds = mesh.get_axis_aligned_bounding_box()
    center = (np.asarray(bounds.min_bound) + np.asarray(bounds.max_bound)) * 0.5
    mesh.translate(-center)
    span = float(np.asarray(bounds.get_extent())[1])
    mesh.scale(height / max(span, 1e-6), center=np.zeros(3))
    mesh.compute_vertex_normals()
    return mesh


def prepared(name, subdivisions=0):
    mesh, _ = load_mesh(name)
    fit_mesh(mesh)
    for _ in range(subdivisions):
        mesh = mesh.subdivide_midpoint(number_of_iterations=1)
        mesh.compute_vertex_normals()
    return mesh


def smoothstep(linear):
    return linear * linear * (3.0 - 2.0 * linear)


def section_segments(verts, faces, axis, value):
    edges = ((0, 1), (1, 2), (2, 0))
    crossings = []
    masks = []
    for a, b in edges:
        p0 = verts[faces[:, a]]
        p1 = verts[faces[:, b]]
        d0 = p0[:, axis] - value
        d1 = p1[:, axis] - value
        hit = d0 * d1 < 0.0
        t = d0 / (d0 - d1 + 1e-12)
        crossings.append(p0 + (p1 - p0) * t[:, None])
        masks.append(hit)
    mask = np.stack(masks, axis=1)
    keep = mask.sum(axis=1) == 2
    if not np.any(keep):
        return np.zeros((0, 2, 3))
    points = np.stack(crossings, axis=1)[keep]
    which = mask[keep]
    first = np.argmax(which, axis=1)
    which[np.arange(len(first)), first] = False
    second = np.argmax(which, axis=1)
    row = np.arange(len(first))
    return np.stack([points[row, first], points[row, second]], axis=1)


def mesh_from_arrays(verts, faces):
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(verts, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32)),
    )
    mesh.compute_vertex_normals()
    return mesh


def apply_yaw(mesh, degrees):
    turned = o3d.geometry.TriangleMesh(mesh)
    turned.rotate(yaw_matrix(degrees), center=np.zeros(3))
    turned.compute_vertex_normals()
    return turned


def apply_yaw_lines(segments, degrees):
    if len(segments) == 0:
        return segments
    flat = segments.reshape(-1, 3) @ yaw_matrix(degrees).T
    return flat.reshape(-1, 2, 3)


class Stage:
    def __init__(self):
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(1280, 720)
        self.renderer.scene.set_background([0.071, 0.078, 0.094, 1.0])
        self.body = o3d.visualization.rendering.MaterialRecord()
        self.body.shader = "defaultLit"
        self.body.base_color = [0.80, 0.82, 0.88, 1.0]
        self.cut = o3d.visualization.rendering.MaterialRecord()
        self.cut.shader = "unlitLine"
        self.cut.line_width = 5.0
        self.cut.base_color = [0.96, 0.97, 1.0, 1.0]
        self.eye = np.array([2.15, 0.0, 3.55])
        self.center = np.zeros(3)

    def snap(self, mesh, lines=None, fov=40.0):
        self.renderer.scene.clear_geometry()
        self.renderer.scene.add_geometry("shape", mesh, self.body)
        if lines is not None and len(lines) > 0:
            cloud = lines.reshape(-1, 3)
            pairs = np.arange(len(cloud), dtype=np.int32).reshape(-1, 2)
            drawing = o3d.geometry.LineSet(
                o3d.utility.Vector3dVector(cloud),
                o3d.utility.Vector2iVector(pairs),
            )
            self.renderer.scene.add_geometry("cut", drawing, self.cut)
        self.renderer.setup_camera(fov, self.center, self.eye, [0.0, 1.0, 0.0])
        return np.asarray(self.renderer.render_to_image())


def write_mp4(frames, path, fps):
    height, width = frames[0].shape[:2]
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
            str(path),
        ],
        input=b"".join(frame.tobytes() for frame in frames),
        check=True,
    )
    print(f"Exported {path}")


def film_orbit(stage, steps_each=120, fps=30):
    frames = []
    for name in ("sword", "shield"):
        mesh = prepared(name, subdivisions=1 if name == "sword" else 0)
        for step in range(steps_each):
            yaw = 18.0 + 360.0 * step / steps_each
            frames.append(stage.snap(apply_yaw(mesh, yaw), fov=32.0))
    return frames


def sample_cloud(mesh, count):
    mesh.compute_vertex_normals()
    cloud = mesh.sample_points_uniformly(count)
    return np.asarray(cloud.points, dtype=np.float64), np.asarray(cloud.normals, dtype=np.float64)


def pair_index(src, dst, bands=64):
    count = min(len(src), len(dst))
    src_order = np.argsort(src[:count, 1], kind="stable")
    dst_order = np.argsort(dst[:count, 1], kind="stable")
    choice = np.empty(count, dtype=np.int64)
    edges = np.linspace(0, count, bands + 1, dtype=int)
    for start, stop in zip(edges[:-1], edges[1:]):
        sband = src[src_order[start:stop]]
        didx = dst_order[start:stop]
        dband = dst[didx]
        srank = np.argsort(np.argsort(np.arctan2(sband[:, 2], sband[:, 0])))
        drank = np.argsort(np.arctan2(dband[:, 2], dband[:, 0]))
        choice[src_order[start:stop]] = didx[drank[srank]]
    return choice


def cloud_mesh(points, normals):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.normals = o3d.utility.Vector3dVector(normals)
    return cloud


def yaw_arrays(points, normals, degrees):
    matrix = yaw_matrix(degrees).astype(np.float64)
    return points @ matrix.T, normals @ matrix.T


def film_flow(stage, steps=180, fps=30):
    sword = prepared("sword")
    shield = prepared("shield")
    src, src_n = sample_cloud(sword, 180_000)
    dst, dst_n = sample_cloud(shield, 180_000)
    choice = pair_index(src, dst)
    count = len(choice)
    src, src_n = src[:count], src_n[:count]
    dst, dst_n = dst[choice], dst_n[choice]
    stage.body.point_size = 6.0
    hold = 0 if steps < 40 else 18
    frames = []
    for step in range(steps):
        if step < hold:
            alpha = 0.0
        elif step >= steps - hold:
            alpha = 1.0
        else:
            alpha = smoothstep((step - hold) / (steps - 2 * hold - 1))
        points = (1.0 - alpha) * src + alpha * dst
        normals = (1.0 - alpha) * src_n + alpha * dst_n
        length = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(length, 1e-6)
        radial = points.copy()
        radial[:, 1] = 0.0
        radial_len = np.linalg.norm(radial, axis=1, keepdims=True)
        points = points + (0.06 * np.sin(np.pi * alpha)) * radial / np.maximum(radial_len, 1e-6)
        yaw = 16.0 + 18.0 * alpha
        points, normals = yaw_arrays(points, normals, yaw)
        frames.append(stage.snap(cloud_mesh(points, normals), fov=32.0))
    return frames


def film_slice(stage, steps_each=100, fps=30):
    frames = []
    pairs = (
        (prepared("sword", subdivisions=1), 1),
        (prepared("shield"), 0),
    )
    hold = 12
    for mesh, axis in pairs:
        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles, dtype=np.int32)
        top = float(verts[:, axis].max())
        bottom = float(verts[:, axis].min())
        centroids = verts[faces].mean(axis=1)[:, axis]
        for step in range(steps_each):
            travel = 0.0 if step < hold else (step - hold) / (steps_each - hold - 1)
            value = top + (bottom - top) * travel * 0.97
            keep = faces[centroids <= value]
            shown = mesh_from_arrays(verts, keep if len(keep) else faces[:1])
            lines = section_segments(verts, faces, axis, value)
            yaw = 24.0
            frames.append(stage.snap(apply_yaw(shown, yaw), apply_yaw_lines(lines, yaw), fov=32.0))
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=".")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stage = Stage()
    print("Rendering surface flow...")
    write_mp4(film_flow(stage, fps=args.fps), out / "flow_sword_shield.mp4", args.fps)
    print("Rendering orbit...")
    write_mp4(film_orbit(stage, fps=args.fps), out / "orbit_sword_shield.mp4", args.fps)
    print("Rendering slice...")
    write_mp4(film_slice(stage, fps=args.fps), out / "slice_sword_shield.mp4", args.fps)


if __name__ == "__main__":
    main()
