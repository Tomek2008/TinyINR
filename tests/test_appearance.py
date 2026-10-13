import json

import numpy as np
from PIL import Image
import pytest
import torch
import trimesh
from trimesh.visual.material import PBRMaterial

from appearance_data import (linear_to_srgb, load_appearance_mesh, make_appearance_dataset,
                             make_geometry_dataset, sample_texture, srgb_to_linear,
                             normalized_bounds, validate_encoding_domain)
from appearance_model import AppearanceKAN, load_checkpoint, save_checkpoint
from examples.make_appearance_demo import make_demo
from evaluate_appearance import geometry_metrics
from reconstruct_appearance import export_mesh, reconstruct
from reconstruct import load_model
from train import build_model, save_int8, save_weights
from train_appearance import _predict, train, boundary_diagnostics


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_texture_sampling_uv_orientation_repeat_and_interpolation():
    pixels = np.array([[[1, 0, 0, 1], [0, 1, 0, 1]],
                       [[0, 0, 1, 1], [1, 1, 1, 1]]], dtype=np.float32)
    uv = np.array([[0.25, 0.75], [0.75, 0.75], [0.25, 0.25], [0.75, 0.25],
                   [1.25, -0.25], [0.5, 0.5]])
    expected = np.vstack([pixels.reshape(-1, 4), pixels[0, 0], [0.5, 0.5, 0.5, 1]])
    np.testing.assert_allclose(sample_texture(pixels, uv), expected)
    np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(np.linspace(0, 1, 100))),
                               np.linspace(0, 1, 100), atol=1e-6)


def test_obj_mtl_texture_and_face_labels(tmp_path):
    Image.new("RGBA", (2, 2), (128, 64, 255, 255)).save(tmp_path / "color.png")
    (tmp_path / "material.mtl").write_text(
        "newmtl textured\nKd 1 1 1\nNs 30\nmap_Kd color.png\n"
        "newmtl green\nKd 0 1 0\nNs 0\n")
    path = tmp_path / "multi.obj"
    path.write_text("mtllib material.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\n"
                    "v 2 0 0\nv 3 0 0\nv 2 1 0\n"
                    "vt 0 0\nvt 1 0\nvt 0 1\n"
                    "usemtl textured\nf 1/1 2/2 3/3\nusemtl green\nf 4/1 5/2 6/3\n")
    with pytest.warns(UserWarning, match="watertight"):
        mesh = load_appearance_mesh(path)
    points, colors, labels = mesh.sample_surface(200, np.random.default_rng(0))
    left = points[:, 0] < 1.5
    assert left.any() and (~left).any()
    np.testing.assert_allclose(colors[left, :3], np.tile(srgb_to_linear([128 / 255, 64 / 255, 1]), (left.sum(), 1)), atol=1e-6)
    np.testing.assert_allclose(colors[~left, :3], np.tile([0, 1, 0], ((~left).sum(), 1)))
    assert len(np.unique(labels)) == 2
    assert len(np.unique(labels[left])) == len(np.unique(labels[~left])) == 1
    assert labels[left][0] != labels[~left][0]
    assert all(material["alpha_mode"] == "OPAQUE" for material in mesh.materials)


def test_vertex_colors_and_seed(tmp_path):
    mesh = trimesh.creation.box()
    mesh.visual.vertex_colors = [128, 64, 255, 255]
    path = tmp_path / "vertex.ply"
    mesh.export(path)
    loaded = load_appearance_mesh(path)
    first = make_appearance_dataset(loaded, 100, seed=8)
    second = make_appearance_dataset(loaded, 100, seed=8)
    for a, b in zip(first.tensors, second.tensors):
        assert torch.equal(a, b)
    expected = srgb_to_linear([128 / 255, 64 / 255, 1])
    np.testing.assert_allclose(first.tensors[1][:, :3], np.tile(expected, (100, 1)), atol=1e-6)


def test_gltf_combines_texture_vertex_color_factor_and_alpha(tmp_path):
    material = PBRMaterial(baseColorFactor=[128, 255, 255, 128], alphaMode="BLEND",
                           baseColorTexture=Image.new("RGBA", (2, 2), (128, 64, 255, 192)))
    visual = trimesh.visual.TextureVisuals(uv=np.full((3, 2), 0.5), material=material)
    visual.vertex_attributes["color"] = np.array([[0, 0, 255, 255], [255, 0, 255, 255],
                                                  [0, 255, 255, 255]], dtype=np.uint8)
    triangle = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                               faces=[[0, 1, 2]], visual=visual, process=False)
    path = tmp_path / "combined.glb"
    path.write_bytes(trimesh.Scene(triangle).export(file_type="glb"))
    with pytest.warns(UserWarning, match="watertight"):
        loaded = load_appearance_mesh(path)
    points, rgba, _ = loaded.sample_surface(100, np.random.default_rng(1))
    expected = np.column_stack([points[:, :2], np.ones((100, 2))])
    expected *= [128 / 255, 1, 1, 128 / 255]
    expected[:, :3] *= srgb_to_linear([128 / 255, 64 / 255, 1])
    expected[:, 3] *= 192 / 255
    np.testing.assert_allclose(rgba, expected, atol=1e-6)
    assert loaded.materials[0]["alpha_mode"] == "BLEND"


def test_scene_instances_transform_and_shared_material(tmp_path):
    box = trimesh.creation.box()
    box.visual = trimesh.visual.TextureVisuals(material=PBRMaterial(
        baseColorFactor=[255, 0, 0, 255], roughnessFactor=0.25, metallicFactor=0.8))
    scene = trimesh.Scene()
    scene.add_geometry(box, geom_name="box", node_name="left")
    matrix = np.eye(4)
    matrix[:3, 3] = [3, 0, 0]
    scene.graph.update(frame_to="right", matrix=matrix, geometry="box")
    path = tmp_path / "instances.glb"
    path.write_bytes(scene.export(file_type="glb"))
    loaded = load_appearance_mesh(path)
    assert len(loaded.faces) == 24
    assert len(loaded.materials) == 1
    np.testing.assert_allclose(loaded.meta["min_bound"], [-0.5] * 3)
    np.testing.assert_allclose(loaded.meta["max_bound"], [3.5, 0.5, 0.5])
    assert loaded.materials[0]["metallic"] == pytest.approx(0.8)
    assert loaded.materials[0]["roughness"] == pytest.approx(0.25)


def test_normalized_sdf_is_scale_invariant(tmp_path):
    datasets = []
    for scale, translation in [(1, 0), (20, 100)]:
        box = trimesh.creation.box()
        box.apply_scale(scale)
        box.apply_translation([translation] * 3)
        box.visual.vertex_colors = [255, 255, 255, 255]
        path = tmp_path / f"box_{scale}.ply"
        box.export(path)
        datasets.append(make_geometry_dataset(load_appearance_mesh(path), 300, seed=5, band=True))
    for a, b in zip(datasets[0].tensors, datasets[1].tensors):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-4)
    coords, sdf = datasets[0].tensors
    q = coords.abs() - 1
    exact = q.clamp_min(0).norm(dim=-1) + q.max(dim=-1).values.clamp_max(0)
    torch.testing.assert_close(sdf[:, 0], exact, atol=2e-6, rtol=1e-4)


@pytest.mark.parametrize("precision", ["fp32", "fp16", "int8"])
def test_checkpoint_and_qat_roundtrip(tmp_path, precision):
    torch.manual_seed(0)
    model = AppearanceKAN(3, hidden=12, hidden_layers=1, omega_0=12, input_scale=2)
    path = tmp_path / f"{precision}.pth"
    meta = {"center": [1, 2, 3], "scale": 2, "materials": [{"name": "a"}] * 3}
    save_checkpoint(model, path, meta, precision)
    recovered, loaded_meta = load_checkpoint(path)
    assert meta == loaded_meta and model.config == recovered.config
    coords = torch.randn(20, 3)
    expected = _predict(model, coords, qat=precision == "int8")
    for key, value in recovered(coords).items():
        torch.testing.assert_close(value, expected[key], atol=0.002 if precision == "fp16" else 1e-6, rtol=1e-4)
    if precision == "int8":
        sum(value.sum() for value in expected.values()).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_glb_export_keeps_linear_colors_and_pbr_factors(tmp_path):
    source = load_appearance_mesh(make_demo(tmp_path / "source.glb"))
    rgba = np.tile([0.2, 0.4, 0.6, 1.0], (len(source.vertices), 1))
    path = tmp_path / "reconstruction.glb"
    export_mesh(source.vertices, source.faces, rgba, source.face_materials, source.materials, path)
    loaded = load_appearance_mesh(path)
    _, colors, _ = loaded.sample_surface(100, np.random.default_rng(1))
    np.testing.assert_allclose(colors, np.tile(rgba[0], (100, 1)), atol=1 / 255)
    assert sorted(m["metallic"] for m in loaded.materials) == pytest.approx([0, 0.9])
    assert sorted(m["roughness"] for m in loaded.materials) == pytest.approx([0.2, 0.85])


def test_ply_export_encodes_srgb_and_preserves_alpha(tmp_path):
    mesh = trimesh.creation.box()
    rgba = np.tile([0.2, 0.4, 0.6, 0.5], (len(mesh.vertices), 1))
    path = tmp_path / "colors.ply"
    with pytest.warns(UserWarning, match="PLY"):
        export_mesh(mesh.vertices, mesh.faces, rgba, np.zeros(len(mesh.faces)), [], path)
    loaded = load_appearance_mesh(path)
    _, colors, _ = loaded.sample_surface(50, np.random.default_rng(0))
    np.testing.assert_allclose(colors, np.tile(rgba[0], (50, 1)), atol=0.005)
    assert loaded.materials[0]["alpha_mode"] == "BLEND"


def test_legacy_geometry_checkpoints_still_load(tmp_path):
    model = build_model(hidden=8, hidden_layers=1).eval()
    coords = torch.randn(10, 3)
    for name, save in [("fp32", lambda m, p: save_weights(m, p, torch.float32)), ("int8", save_int8)]:
        path = tmp_path / f"legacy_{name}.pth"
        save(model, path)
        loaded = load_model(path, device="cpu")
        assert loaded(coords).shape == (10, 1)
        assert torch.isfinite(loaded(coords)).all()
        if name == "fp32":
            torch.testing.assert_close(loaded(coords), model(coords))


def test_full_training_qat_and_reconstruction_without_source(tmp_path):
    path = make_demo(tmp_path / "demo.glb")
    report = train(str(path), str(tmp_path / "out"), hidden=32, hidden_layers=2,
                   epochs=35, qat_epochs=2, samples=2048, appearance_samples=1024,
                   batch_size=256, lr=1e-3, seed=4, device="cpu")
    history = report["fp32_fit"]["history"]
    assert history[-1]["loss"] < history[0]["loss"] * 0.5
    assert report["checkpoints"]["fp32"]["material_accuracy"] > 0.9
    assert report["checkpoints"]["fp32"]["rgb_psnr"] > 16
    assert set(report["checkpoints"]) == {"fp32", "fp16", "int8"}
    assert json.loads((tmp_path / "out/appearance_metrics.json").read_text())["config"]["num_materials"] == 2
    path.unlink()
    output = tmp_path / "reconstruction.glb"
    vertices, faces = reconstruct(tmp_path / "out/demo_appearance_int8.pth", output,
                                  resolution=24, batch_size=2048, device="cpu")
    assert len(vertices) > 100 and len(faces) > 100
    assert len(trimesh.load_scene(output).geometry) == 2


@pytest.mark.parametrize("qat_epochs", [0, 1])
def test_tiny_batches_and_regularizers_in_qat(tmp_path, qat_epochs):
    path = make_demo(tmp_path / "tiny.glb")
    result = train(str(path), str(tmp_path / "out"), hidden=8, hidden_layers=1,
                   epochs=1, qat_epochs=qat_epochs, samples=20, appearance_samples=10,
                   batch_size=8192, eikonal=0.01, clamp=0.1, device="cpu")
    assert ("int8" in result["checkpoints"]) == bool(qat_epochs)
    assert np.isfinite(result["checkpoints"]["int8" if qat_epochs else "fp16"]["sdf_mse"])


@pytest.mark.parametrize("kwargs", [{"epochs": 0}, {"batch_size": 0}, {"samples": 0},
                                   {"appearance_samples": 1}, {"color_weight": -1}])
def test_invalid_training_arguments_fail_before_loading(kwargs):
    with pytest.raises(ValueError):
        train("does-not-exist.glb", **kwargs)


def test_periodic_replicas_are_rejected_before_training(tmp_path):
    path = make_demo(tmp_path / "demo.glb")
    mesh = load_appearance_mesh(path)
    with pytest.raises(ValueError, match="periodic replicas"):
        train(str(path), str(tmp_path / "out"), input_scale=3.0, samples=20)
    validate_encoding_domain(mesh.meta, AppearanceKAN(2).config["input_scale"])
    # A legacy network cannot distinguish inside from outside at these points.
    model = AppearanceKAN(2, hidden=8, hidden_layers=1, input_scale=3.0)
    coords = torch.tensor([[-0.94, 0.2, 0], [-0.94 + 2 * np.pi / 3, 0.2, 0]])
    output = model(coords)
    for values in output.values():
        torch.testing.assert_close(values[0], values[1], atol=1e-6, rtol=1e-5)
    assert coords[0].norm() < 1 < coords[1].norm()


def test_geometry_training_covers_reconstruction_boundary(tmp_path):
    mesh = load_appearance_mesh(make_demo(tmp_path / "demo.glb"))
    coords, sdf = make_geometry_dataset(mesh, 1000).tensors
    low, high = normalized_bounds(mesh.meta)
    boundary = np.isclose(coords, low).any(axis=1) | np.isclose(coords, high).any(axis=1)
    assert boundary.sum() >= 40
    assert (sdf[boundary] > 0).all()
    for axis in range(3):
        assert np.isclose(coords[:, axis], low[axis]).any()
        assert np.isclose(coords[:, axis], high[axis]).any()


def test_sphere_has_one_surface_without_periodic_caps(tmp_path):
    path = make_demo(tmp_path / "sphere.glb")
    train(str(path), str(tmp_path / "out"), hidden=32, hidden_layers=2,
          epochs=60, qat_epochs=3, samples=8192, appearance_samples=4096,
          batch_size=512, lr=1e-3, seed=0, device="cpu")
    weights = tmp_path / "out/sphere_appearance_int8.pth"
    model, meta = load_checkpoint(weights)
    assert boundary_diagnostics(model, meta)["boundary_min_sdf"] > 0
    vertices, faces = reconstruct(weights, tmp_path / "sphere.glb", resolution=48, device="cpu")
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    assert len(mesh.split(only_watertight=False)) == 1
    radii = np.linalg.norm(vertices, axis=1)
    assert np.abs(radii - 1).max() < 0.04


def test_geometry_metrics_use_normalized_unsquared_distances(tmp_path):
    triangle = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                               faces=[[0, 1, 2]], vertex_colors=[255, 255, 255, 255], process=False)
    path = tmp_path / "triangle.ply"
    triangle.export(path)
    with pytest.warns(UserWarning, match="watertight"):
        source = load_appearance_mesh(path)
    result = geometry_metrics(source, source.vertices + [0, 0, 0.1], source.faces, samples=100)
    assert result["mean_surface_distance"] == pytest.approx(0.2)
    assert result["p95_surface_distance"] == pytest.approx(0.2)
    assert result["fscore_at_0_01"] == 0
