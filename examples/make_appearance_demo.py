"""Generate a closed sphere with a red metal half and a blue matte half."""

import argparse
from pathlib import Path

import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial


def make_demo(path):
    sphere = trimesh.creation.icosphere(subdivisions=3)
    labels = sphere.triangles_center[:, 0] >= 0
    materials = [PBRMaterial(name="red_metal", baseColorFactor=[220, 30, 15, 255],
                             metallicFactor=0.9, roughnessFactor=0.2),
                 PBRMaterial(name="blue_matte", baseColorFactor=[15, 65, 220, 255],
                             metallicFactor=0.0, roughnessFactor=0.85)]
    scene = trimesh.Scene()
    for label, material in enumerate(materials):
        faces = sphere.faces[labels == label]
        indices, inverse = np.unique(faces, return_inverse=True)
        part = trimesh.Trimesh(vertices=sphere.vertices[indices], faces=inverse.reshape(-1, 3),
                              visual=trimesh.visual.TextureVisuals(material=material), process=False)
        scene.add_geometry(part, geom_name=material.name)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(scene.export(file_type="glb"))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="weights/appearance_demo/source.glb")
    print(make_demo(parser.parse_args().out))
