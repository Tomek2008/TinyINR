"""Render appearance checkpoints into reproducible README comparisons.

Requires Blender on PATH in addition to the project's Python dependencies.
The Blender worker uses the imported GLB materials without changing geometry.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def blender_worker(manifest_path):
    import bpy
    from mathutils import Vector

    manifest = json.loads(Path(manifest_path).read_text())
    settings = manifest["render"]
    camera_direction = Vector(settings.get("view", (0.6, -6.0, 1.4))).normalized()

    def point_at(obj, target):
        obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()

    def light(name, position, energy, size, size_y):
        data = bpy.data.lights.new(name, "AREA")
        data.energy = energy
        data.shape = "RECTANGLE"
        data.size, data.size_y = size, size_y
        obj = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(obj)
        obj.location = position
        point_at(obj, (0, 0, 0))

    for entry in manifest["entries"]:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = settings["samples"]
        scene.cycles.use_denoising = True
        scene.cycles.seed = 0
        scene.render.threads_mode = "FIXED"
        scene.render.threads = settings["threads"]
        scene.render.resolution_x = scene.render.resolution_y = settings["tile_size"]
        scene.render.resolution_percentage = 100
        scene.render.film_transparent = True
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
        scene.world = bpy.data.worlds.new("Studio world")
        scene.world.use_nodes = True
        background = scene.world.node_tree.nodes.get("Background")
        background.inputs["Color"].default_value = (0.20, 0.23, 0.30, 1)
        background.inputs["Strength"].default_value = 0.45

        bpy.ops.import_scene.gltf(filepath=entry["mesh"])
        imported = list(scene.objects)
        root = bpy.data.objects.new("Shared normalization", None)
        scene.collection.objects.link(root)
        for obj in imported:
            if obj.parent is None:
                obj.parent = root
        center = manifest["normalization"]["center"]
        scale = manifest["normalization"]["scale"]
        # Blender imports glTF's Y-up coordinates as Z-up.
        root.location = Vector((-center[0], center[2], -center[1])) / scale
        root.scale = (1 / scale,) * 3

        camera_data = bpy.data.cameras.new("Shared orthographic camera")
        camera = bpy.data.objects.new("Camera", camera_data)
        scene.collection.objects.link(camera)
        scene.camera = camera
        camera_data.type = "ORTHO"
        camera_data.ortho_scale = settings["ortho_scale"]
        camera.location = camera_direction * 6
        point_at(camera, (0, 0, 0))
        light("Large key", (-3.0, -4.5, 5.0), 420, 3.0, 5.0)
        light("Soft fill", (4.0, -2.0, 1.5), 130, 4.0, 4.0)
        light("Top rim", (0.0, 3.0, 4.0), 350, 3.0, 3.0)

        def render(mode):
            scene.render.filepath = str(Path(manifest["cache"]) / f"{entry['id']}_{mode}.png")
            bpy.ops.render.render(write_still=True)
            print(f"RENDERED {entry['id']} {mode}", flush=True)

        render("studio")
        # Route the existing base-color expression into emission, keeping both
        # vertex colors and glTF baseColorFactor. This removes illumination only.
        for material in bpy.data.materials:
            if not material.use_nodes:
                continue
            nodes, links = material.node_tree.nodes, material.node_tree.links
            principled = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
            output = next((node for node in nodes if node.type == "OUTPUT_MATERIAL"), None)
            if principled is None or output is None:
                continue
            emission = nodes.new("ShaderNodeEmission")
            color = principled.inputs["Base Color"]
            if color.is_linked:
                links.new(color.links[0].from_socket, emission.inputs["Color"])
            else:
                emission.inputs["Color"].default_value = color.default_value
            links.new(emission.outputs[0], output.inputs["Surface"])
        scene.cycles.samples = 16
        scene.cycles.use_denoising = False
        render("color")
        target = Vector(settings["detail_center"])
        camera.location = target + camera_direction * 6
        camera_data.ortho_scale = settings["detail_scale"]
        point_at(camera, target)
        render("detail")


def compose(manifest, output):
    from PIL import Image, ImageDraw, ImageFont

    background = (17, 19, 22)
    panel = (23, 26, 32)
    border = (45, 50, 60)
    white, muted, accent = (236, 240, 246), (155, 165, 181), (98, 217, 199)
    width, margin, gap = 2400, 80, 24
    column = (width - margin * 2 - gap * 3) // 4

    def font(size, bold=False):
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            return ImageFont.load_default(size=size)

    def text(draw, xy, value, size=24, color=white, bold=False):
        draw.text(xy, value, font=font(size, bold), fill=color)

    def tile(canvas, entry, mode, xy, size):
        path = Path(manifest["cache"]) / f"{entry['id']}_{mode}.png"
        with Image.open(path) as source:
            rgba = source.convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
        canvas.paste(rgba, xy, rgba)

    def heading(draw, title, subtitle):
        text(draw, (margin, 38), "TINYINR  /  APPEARANCE", 22, accent, True)
        text(draw, (margin - 2, 83), title, 58, bold=True)
        text(draw, (margin, 164), subtitle, 25, muted)

    canvas = Image.new("RGB", (width, 1200), background)
    draw = ImageDraw.Draw(canvas)
    heading(draw, manifest.get("title", "Color & material reconstruction"),
            "Original mesh and three stored precisions. Same camera, lighting and scale.")
    for i, entry in enumerate(manifest["entries"]):
        x = margin + i * (column + gap)
        draw.rounded_rectangle((x, 234, x + column, 1008), radius=20, fill=panel,
                               outline=accent if entry["id"] == "int8" else border, width=2)
        text(draw, (x + 28, 261), entry["title"], 34, bold=True)
        text(draw, (x + 28, 311), entry["subtitle"], 20, muted)
        tile(canvas, entry, "studio", (x + 6, 340), column - 12)
        # The same image viewport is used for all four meshes, including artifacts.
        draw.line((x + 28, 887, x + column - 28, 887), fill=border, width=2)
        text(draw, (x + 28, 909), f"{entry['bytes'] / 1024:.1f} KiB", 31, bold=True)
        suffix = "source GLB" if entry["id"] == "original" else "checkpoint"
        text(draw, (x + 245, 920), suffix, 19, muted)
        if entry["id"] == "original":
            stats = "Reference colors and material assignments"
        else:
            stats = f"RGB {entry['rgb_psnr']:.2f} dB  ·  materials {entry['material_accuracy']:.1%}"
        text(draw, (x + 28, 961), stats, 19, accent if entry["id"] == "int8" else muted)

    for i, material in enumerate(manifest.get("materials", [])[:4]):
        x = margin + i * (column + gap)
        text(draw, (x, 1041), material["name"].replace("_", " ").upper(), 20, accent, True)
        description = ("base-color texture · " if material.get("has_base_color_texture") else "")
        description += f"metal {material['metallic']:.2f} · roughness {material['roughness']:.2f}"
        text(draw, (x, 1074), description, 17, muted)
    text(draw, (margin, 1131),
         f"{manifest['resolution']}³ marching cubes · unfiltered meshes · RGB / material scores use held-out surface samples", 21, muted)
    prefix = manifest.get("prefix", "appearance")
    canvas.save(output / f"{prefix}_comparison.png", optimize=True)

    canvas = Image.new("RGB", (width, 1520), background)
    draw = ImageDraw.Draw(canvas)
    heading(draw, "Base color, without lighting",
            "The same meshes with unlit base color. Lower row: the same surface detail in every reconstruction.")
    for i, entry in enumerate(manifest["entries"]):
        x = margin + i * (column + gap)
        text(draw, (x + 20, 243), entry["title"], 31, bold=True)
        draw.rounded_rectangle((x, 306, x + column, 806), radius=16, fill=panel, outline=border, width=2)
        tile(canvas, entry, "color", (x + (column - 490) // 2, 311), 490)
        draw.rounded_rectangle((x, 900, x + column, 1400), radius=16, fill=panel, outline=border, width=2)
        tile(canvas, entry, "detail", (x + (column - 490) // 2, 905), 490)
    zoom = manifest["render"]["ortho_scale"] / manifest["render"]["detail_scale"]
    text(draw, (margin, 844), f"SURFACE DETAIL  /  {zoom:.1f}× CLOSER", 23, accent, True)
    text(draw, (margin, 1444),
         "Linear base color → sRGB display · identical framing · no geometry smoothing, color retouching or component removal", 21, muted)
    canvas.save(output / f"{prefix}_colors.png", optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("weights/appearance_demo"))
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("docs"))
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--tile-size", type=int, default=768)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--prefix", default="appearance")
    parser.add_argument("--title", default="Color & material reconstruction")
    parser.add_argument("--view", nargs=3, type=float, default=[0.6, -6.0, 1.4])
    parser.add_argument("--detail-center", nargs=3, type=float, default=[0, -0.955, 0.23],
                        help="Close-up center in normalized Blender (Z-up) coordinates")
    parser.add_argument("--ortho-scale", type=float, default=3.35)
    parser.add_argument("--detail-scale", type=float, default=1.1)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--compose-only", action="store_true", help="Reuse previously rendered tiles")
    args = parser.parse_args()
    if min(args.tile_size, args.samples, args.threads, args.ortho_scale, args.detail_scale) <= 0 or args.resolution < 3:
        parser.error("Render settings must be positive")
    if Path(args.prefix).name != args.prefix or not args.prefix:
        parser.error("--prefix must be a filename prefix, not a path")
    if not any(args.view):
        parser.error("--view must be nonzero")
    args.out.mkdir(parents=True, exist_ok=True)
    cache = (args.run / "renders").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = cache / "manifest.json"
    if args.compose_only:
        manifest = json.loads(manifest_path.read_text())
    else:
        import torch
        from reconstruct_appearance import reconstruct

        blender = shutil.which(args.blender)
        if blender is None:
            parser.error("Blender is required for rendering; install it or pass --blender /path/to/blender")
        torch.set_num_threads(args.threads)
        report = json.loads((args.run / "appearance_metrics.json").read_text())
        source = args.source or Path(report["meta"]["source"])
        if not source.is_file():
            parser.error("Source GLB is missing; pass --source /path/to/source.glb")
        entries = [{"id": "original", "title": "Original", "subtitle": f"{len(report['meta']['materials'])} source materials",
                    "mesh": str(source.resolve()), "bytes": source.stat().st_size}]
        for precision in ("fp32", "fp16", "int8"):
            stats = report["checkpoints"][precision]
            checkpoint = args.run / stats["path"]
            mesh = cache / f"reconstruction_{precision}_{args.resolution}.glb"
            reconstruct(checkpoint, mesh, resolution=args.resolution, device="cpu")
            entries.append({**stats, "id": precision, "title": precision.upper(),
                            "subtitle": "Quantization-aware training" if precision == "int8" else "Floating-point weights",
                            "mesh": str(mesh)})
        manifest = {"entries": entries, "cache": str(cache), "resolution": args.resolution,
                    "materials": report["meta"]["materials"], "prefix": args.prefix, "title": args.title,
                    "normalization": {k: report["meta"][k] for k in ("center", "scale")},
                    "render": {"samples": args.samples, "tile_size": args.tile_size,
                               "threads": args.threads, "ortho_scale": args.ortho_scale,
                               "detail_scale": args.detail_scale, "detail_center": args.detail_center,
                               "view": args.view}}
        manifest_path.write_text(json.dumps(manifest, indent=2))
        environment = {**os.environ, "ALSOFT_DRIVERS": "null", "SDL_AUDIODRIVER": "dummy"}
        log_path = cache / "blender.log"
        print(f"Rendering with Blender; progress log: {log_path}", flush=True)
        with log_path.open("w") as log:
            result = subprocess.run([blender, "-noaudio", "--background", "--factory-startup",
                                     "--threads", str(args.threads), "--python-exit-code", "1",
                                     "--python", str(Path(__file__).resolve()), "--",
                                     "--blender-worker", str(manifest_path)],
                                    env=environment, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Blender exited with {result.returncode}. See {log_path}")
    compose(manifest, args.out)
    print(f"Saved comparison figures to {args.out} (prefix: {manifest.get('prefix', 'appearance')})")


if __name__ == "__main__":
    if "--blender-worker" in sys.argv:
        blender_worker(sys.argv[sys.argv.index("--blender-worker") + 1])
    else:
        main()
