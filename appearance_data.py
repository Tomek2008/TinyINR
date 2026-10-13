"""Surface supervision retaining face materials, UV seams and scene transforms."""

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import point_cloud_utils as pcu
import torch
from torch.utils.data import TensorDataset
import trimesh
from trimesh.visual.material import MultiMaterial, PBRMaterial


def srgb_to_linear(rgb):
    rgb = np.asarray(rgb, dtype=np.float32)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb):
    rgb = np.clip(rgb, 0, 1)
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - 0.055)


def _rgba(values):
    values = np.asarray(values)
    result = values.astype(np.float32)
    if np.issubdtype(values.dtype, np.integer):
        result /= 255.0
    if result.shape[-1] == 3:
        result = np.concatenate([result, np.ones((*result.shape[:-1], 1))], axis=-1)
    return np.clip(result, 0, 1).astype(np.float32)


def _material(material, index):
    is_pbr = isinstance(material, PBRMaterial)
    color = np.ones(4, dtype=np.float32)
    image = None
    roughness, metallic = 1.0, 0.0
    alpha_mode, alpha_cutoff, double_sided = "OPAQUE", 0.5, False
    emissive = [0.0, 0.0, 0.0]
    if is_pbr:
        if material.baseColorFactor is not None:
            color = _rgba(material.baseColorFactor)
        image = material.baseColorTexture
        roughness = 1.0 if material.roughnessFactor is None else material.roughnessFactor
        metallic = 1.0 if material.metallicFactor is None else material.metallicFactor
        alpha_mode = material.alphaMode or "OPAQUE"
        alpha_cutoff = 0.5 if material.alphaCutoff is None else material.alphaCutoff
        double_sided = bool(material.doubleSided)
        if material.emissiveFactor is not None:
            emissive = material.emissiveFactor.tolist()
        ignored = [key for key in ("normalTexture", "occlusionTexture",
                   "metallicRoughnessTexture", "emissiveTexture")
                   if getattr(material, key, None) is not None]
        if ignored:
            warnings.warn(f"Material {material.name}: unsupported maps {ignored}; "
                          "only base color textures and constant PBR factors are retained.")
    elif material is not None:
        color = _rgba(material.diffuse)
        color[:3] = srgb_to_linear(color[:3])
        image = material.image
        roughness = (2.0 / (float(material.glossiness) + 2.0)) ** 0.25
        # trimesh retains MTL opacity in kwargs rather than diffuse alpha.
        opacity = material.kwargs.get("d")
        transparency = material.kwargs.get("tr")
        if opacity is not None:
            color[3] = float(np.asarray(opacity).reshape(-1)[0])
        elif transparency is not None:
            color[3] = 1.0 - float(np.asarray(transparency).reshape(-1)[0])
        if color[3] < 1 or (image is not None and image.convert("RGBA").getextrema()[3][0] < 255):
            alpha_mode = "BLEND"
    pixels = None
    if image is not None:
        pixels = _rgba(np.asarray(image.convert("RGBA")))
        pixels[..., :3] = srgb_to_linear(pixels[..., :3])
    record = {"name": getattr(material, "name", None) or f"material_{index}",
              "base_color": np.clip(color, 0, 1).tolist(),
              "roughness": float(np.clip(roughness, 0, 1)),
              "metallic": float(np.clip(metallic, 0, 1)),
              "emissive": emissive, "alpha_mode": alpha_mode,
              "alpha_cutoff": float(alpha_cutoff), "double_sided": double_sided,
              "has_base_color_texture": pixels is not None}
    return record, pixels


def sample_texture(pixels, uv):
    """Bilinear sampling with repeat wrapping; UV origin is bottom left."""
    h, w = pixels.shape[:2]
    x = (uv[:, 0] % 1) * w - 0.5
    y = ((1 - uv[:, 1]) % 1) * h - 0.5
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    tx, ty = (x - x0)[:, None], (y - y0)[:, None]
    return ((1 - tx) * (1 - ty) * pixels[y0 % h, x0 % w]
            + tx * (1 - ty) * pixels[y0 % h, (x0 + 1) % w]
            + (1 - tx) * ty * pixels[(y0 + 1) % h, x0 % w]
            + tx * ty * pixels[(y0 + 1) % h, (x0 + 1) % w])


@dataclass
class AppearanceMesh:
    vertices: np.ndarray
    faces: np.ndarray
    face_materials: np.ndarray
    corner_colors: np.ndarray
    corner_uvs: np.ndarray
    materials: list
    textures: list
    meta: dict

    def sample_surface(self, count, rng):
        if count < 1:
            raise ValueError("Surface sample count must be positive")
        triangles = self.vertices[self.faces]
        areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                        triangles[:, 2] - triangles[:, 0]), axis=1)
        face_ids = rng.choice(len(self.faces), size=count, p=areas / areas.sum())
        root = np.sqrt(rng.random(count))
        second = rng.random(count)
        bary = np.column_stack([1 - root, root * (1 - second), root * second])
        points = np.einsum("ni,nij->nj", bary, triangles[face_ids])
        colors = np.einsum("ni,nij->nj", bary, self.corner_colors[face_ids])
        uv = np.einsum("ni,nij->nj", bary, self.corner_uvs[face_ids])
        labels = self.face_materials[face_ids]
        for i, material in enumerate(self.materials):
            mask = labels == i
            colors[mask] *= material["base_color"]
            if self.textures[i] is not None:
                colors[mask] *= sample_texture(self.textures[i], uv[mask])
        # Normalize coordinates before casting them to float32 in the datasets.
        return points, colors.astype(np.float32), labels.copy()


def load_appearance_mesh(path):
    path = Path(path)
    if not path.is_file():
        alias = Path(__file__).parent / "assets" / f"{path}.obj"
        if alias.is_file():
            path = alias
        else:
            raise FileNotFoundError(f"Mesh not found: {path}")
    scene = trimesh.load_scene(path, process=False)
    vertices, faces, ids, colors, uvs, palette, textures = [], [], [], [], [], [], []
    material_ids = {}
    offset = 0
    has_appearance = False
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        source = scene.geometry[geometry_name]
        if not isinstance(source, trimesh.Trimesh) or not len(source.faces):
            continue
        mesh = source.copy()
        # TextureVisuals.copy() copies UVs but drops glTF COLOR_0 attributes.
        if source.visual.kind == "texture":
            for key, value in source.visual.vertex_attributes.items():
                mesh.visual.vertex_attributes[key] = value.copy()
        mesh.apply_transform(transform)
        triangles = np.asarray(mesh.triangles)
        area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                      triangles[:, 2] - triangles[:, 0]), axis=1)
        mesh.update_faces(np.isfinite(area) & (area > 0))
        if not len(mesh.faces):
            continue
        visual = mesh.visual
        corner_color = np.ones((len(mesh.faces), 3, 4), dtype=np.float32)
        corner_uv = np.zeros((len(mesh.faces), 3, 2), dtype=np.float32)
        source_material = getattr(source.visual, "material", None)
        local_materials = (list(source_material.materials) if isinstance(source_material, MultiMaterial)
                           else [source_material])
        local_ids = getattr(visual, "face_materials", None)
        if local_ids is None:
            local_ids = np.zeros(len(mesh.faces), dtype=int)
        local_ids = np.asarray(local_ids, dtype=int)
        if local_ids.shape != (len(mesh.faces),) or np.any(local_ids < 0) or np.any(local_ids >= len(local_materials)):
            raise ValueError("Invalid face material indices")
        mapping = []
        for material in local_materials:
            key = id(material)
            if key not in material_ids:
                index = len(palette)
                record, texture = _material(material, index)
                palette.append(record)
                textures.append(texture)
                material_ids[key] = index
            mapping.append(material_ids[key])
        if visual.kind == "vertex":
            vertex_color = _rgba(visual.vertex_colors)
            if path.suffix.lower() not in (".glb", ".gltf"):
                vertex_color[:, :3] = srgb_to_linear(vertex_color[:, :3])
            corner_color = vertex_color[mesh.faces]
            has_appearance = True
        elif visual.kind == "face":
            face_color = _rgba(visual.face_colors)
            face_color[:, :3] = srgb_to_linear(face_color[:, :3])
            corner_color = np.repeat(face_color[:, None, :], 3, axis=1)
            has_appearance = True
        elif visual.kind == "texture":
            has_appearance = True
            if visual.uv is not None:
                corner_uv = np.asarray(visual.uv, dtype=np.float32)[mesh.faces]
            elif any(textures[i] is not None for i in mapping):
                raise ValueError(f"Mesh {geometry_name} has a texture but no UV coordinates")
            if "color" in visual.vertex_attributes:
                corner_color = _rgba(visual.vertex_attributes["color"])[mesh.faces]
        if not np.isfinite(corner_uv).all() or not np.isfinite(corner_color).all():
            raise ValueError("Mesh contains non-finite colors or UV coordinates")
        if source_material is None and np.any(corner_color[..., 3] < 1):
            palette[mapping[0]]["alpha_mode"] = "BLEND"
        vertices.append(np.asarray(mesh.vertices))
        faces.append(np.asarray(mesh.faces) + offset)
        ids.append(np.asarray(mapping)[local_ids])
        colors.append(corner_color)
        uvs.append(corner_uv)
        offset += len(mesh.vertices)
    if not faces:
        raise ValueError(f"No non-degenerate triangles in {path}")
    vertices = np.concatenate(vertices).astype(np.float64)
    faces = np.concatenate(faces).astype(np.int32)
    used = vertices[np.unique(faces)]
    low, high = used.min(axis=0), used.max(axis=0)
    scale = float((high - low).max() / 2)
    if not np.isfinite(vertices).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Mesh has invalid coordinates or zero extent")
    if not has_appearance:
        warnings.warn("Mesh has no colors or materials; using opaque white.")
    if not trimesh.Trimesh(vertices=vertices, faces=faces, process=True).is_watertight:
        warnings.warn("Mesh is not watertight; signed distances may be unreliable.")
    meta = {"source": str(path.resolve()), "center": ((low + high) / 2).tolist(),
            "scale": scale, "min_bound": low.tolist(), "max_bound": high.tolist(),
            "padding": np.maximum((high - low) * 0.1, scale * 0.05).tolist(),
            "materials": palette, "color_space": "linear", "has_appearance": has_appearance}
    return AppearanceMesh(vertices, faces, np.concatenate(ids), np.concatenate(colors),
                          np.concatenate(uvs), palette, textures, meta)


def make_geometry_dataset(mesh, count=500_000, seed=0, band=False):
    if count < 10:
        raise ValueError("At least 10 SDF samples are required")
    rng = np.random.default_rng(seed)
    center, scale = np.asarray(mesh.meta["center"]), mesh.meta["scale"]
    vertices = np.ascontiguousarray((mesh.vertices - center) / scale)
    n_surface, n_near = int(count * (0.45 if band else 0.6)), int(count * (0.35 if band else 0.2))
    n_shell = int(count * 0.1) if band else 0
    surface = (mesh.sample_surface(n_surface + n_near + n_shell, rng)[0] - center) / scale
    # Exact surface points are supervised at zero. Perturbations use normalized units.
    near = surface[n_surface:n_surface + n_near] + rng.normal(0, 0.005, (n_near, 3))
    shell = surface[n_surface + n_near:] + rng.normal(0, 0.02, (n_shell, 3))
    low, high = normalized_bounds(mesh.meta)
    # Supervise the full reconstruction domain, including its boundary.
    uniform = rng.uniform(low, high, (count - len(surface), 3))
    n_boundary = max(1, len(uniform) // 4)
    axes = rng.integers(0, 3, n_boundary)
    sides = rng.integers(0, 2, n_boundary)
    uniform[np.arange(n_boundary), axes] = np.where(sides, high[axes], low[axes])
    off_surface = np.ascontiguousarray(np.concatenate([near, shell, uniform]), dtype=np.float64)
    distances, _, _ = pcu.signed_distance_to_mesh(off_surface, vertices, mesh.faces)
    coords = np.concatenate([surface[:n_surface], off_surface]).astype(np.float32)
    sdf = np.concatenate([np.zeros(n_surface), distances]).astype(np.float32)[:, None]
    if not np.isfinite(sdf).all():
        raise ValueError("Signed distance computation produced non-finite values")
    return TensorDataset(torch.from_numpy(coords), torch.from_numpy(sdf))


def normalized_bounds(meta):
    center = np.asarray(meta["center"])
    padding = np.asarray(meta["padding"])
    return ((np.asarray(meta["min_bound"]) - padding - center) / meta["scale"],
            (np.asarray(meta["max_bound"]) + padding - center) / meta["scale"])


def validate_encoding_domain(meta, input_scale):
    """A Fourier-only front end identifies x with x + 2π / input_scale."""
    if not np.isfinite(input_scale) or input_scale <= 0:
        raise ValueError("input_scale must be finite and positive")
    low, high = normalized_bounds(meta)
    span = float((high - low).max())
    if input_scale * span >= 2 * np.pi:
        raise ValueError(
            f"Fourier period {2 * np.pi / input_scale:.3f} overlaps the reconstruction "
            f"domain (width {span:.3f}), producing periodic replicas. "
            f"Use --input-scale below {2 * np.pi / span:.3f} (default: 1.0).")


def make_appearance_dataset(mesh, count=200_000, seed=1):
    points, rgba, material_ids = mesh.sample_surface(count, np.random.default_rng(seed))
    coords = ((points - np.asarray(mesh.meta["center"])) / mesh.meta["scale"]).astype(np.float32)
    return TensorDataset(torch.from_numpy(coords), torch.from_numpy(rgba),
                         torch.from_numpy(material_ids).long())
