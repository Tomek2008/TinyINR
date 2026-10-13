"""Measure reconstructed geometry as well as the field's exterior sign."""

import argparse
import json
from pathlib import Path

import numpy as np
import point_cloud_utils as pcu
import torch
import trimesh

from appearance_data import load_appearance_mesh
from appearance_model import load_checkpoint
from reconstruct_appearance import reconstruct
from train_appearance import boundary_diagnostics


def geometry_metrics(source, vertices, faces, samples=20000, seed=0):
    """Symmetric point-to-triangle distances in normalized coordinates."""
    center, scale = np.asarray(source.meta["center"]), source.meta["scale"]
    original_vertices = np.ascontiguousarray((source.vertices - center) / scale)
    predicted_vertices = np.ascontiguousarray((vertices - center) / scale, dtype=np.float64)
    predicted_faces = np.ascontiguousarray(faces, dtype=np.int32)
    original_points = (source.sample_surface(samples, np.random.default_rng(seed))[0] - center) / scale
    predicted_mesh = trimesh.Trimesh(predicted_vertices, predicted_faces, process=False)
    predicted_points, _ = trimesh.sample.sample_surface(predicted_mesh, samples, seed=seed + 1)
    forward, _, _ = pcu.closest_points_on_mesh(np.ascontiguousarray(original_points), predicted_vertices, predicted_faces)
    backward, _, _ = pcu.closest_points_on_mesh(np.ascontiguousarray(predicted_points), original_vertices, source.faces)
    combined = np.concatenate([forward, backward])
    precision, recall = float(np.mean(backward < 0.01)), float(np.mean(forward < 0.01))
    return {"mean_surface_distance": float(combined.mean()),
            "p95_surface_distance": float(np.percentile(combined, 95)),
            "fscore_at_0_01": 2 * precision * recall / max(precision + recall, 1e-12),
            "components": len(predicted_mesh.split(only_watertight=False)),
            "vertices": len(vertices), "triangles": len(faces)}


def evaluate_run(run, source_path=None, resolution=96, samples=20000):
    run = Path(run)
    report = json.loads((run / "appearance_metrics.json").read_text())
    source = load_appearance_mesh(source_path or report["meta"]["source"])
    results = {"resolution": resolution, "surface_samples_each_direction": samples,
               "distance_units": "normalized coordinates (source longest half-extent = 1)",
               "checkpoints": {}}
    for precision, saved in report["checkpoints"].items():
        weights = run / saved["path"]
        vertices, faces = reconstruct(weights, run / f"evaluated_{precision}_{resolution}.glb",
                                      resolution=resolution, device="cpu")
        model, meta = load_checkpoint(weights)
        values = geometry_metrics(source, vertices, faces, samples)
        values.update(boundary_diagnostics(model, meta, resolution=64))
        results["checkpoints"][precision] = values
        print(f"{precision}: mean surface distance {values['mean_surface_distance']:.6f}, "
              f"F-score {values['fscore_at_0_01']:.1%}, components {values['components']}, "
              f"negative boundary {values['boundary_negative_fraction']:.1%}", flush=True)
    (run / "geometry_metrics.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--source", default=None, dest="source_path")
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.samples < 1 or args.resolution < 3 or args.threads < 1:
        parser.error("Samples and threads must be positive; resolution must be at least 3")
    torch.set_num_threads(args.threads)
    arguments = vars(args)
    arguments.pop("threads")
    evaluate_run(**arguments)
