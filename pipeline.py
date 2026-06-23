"""Orchestrator — front concept -> 4-layer 2.5D pack(s), NO mesh.

Single character:
    <TRELLIS2_PY> pipeline.py --concept front.png --slug noble --subject "royal-blue coat with gold trim" --out out/

Roster (two phases so ComfyUI's Qwen-Edit and the diffusers estimators never fight for VRAM):
    <TRELLIS2_PY> pipeline.py --roster roster.json --out out/
  roster.json = [{"slug": "...", "concept": "path.png", "subject": "..."}, ...]

Preconditions: ComfyUI live on COMFY_URL with the Qwen-Edit model; watchdog armed; ollama stopped.
Phase 1 runs NVS (ComfyUI). Phase 2 POSTs /free (unloads Qwen-Edit ~20GB) then runs the
estimators in-process — peak stays well under the 32GB ceiling.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, urllib.request
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import COMFY_PY, COMFY_URL

HERE = Path(__file__).parent


def _free():
    try:
        urllib.request.urlopen(urllib.request.Request(
            f"{COMFY_URL}/free", json.dumps({"unload_models": True, "free_memory": True}).encode(),
            {"Content-Type": "application/json"}), timeout=30).read()
    except Exception:
        pass


def nvs_one(concept: Path, subject: str, nvs_dir: Path):
    """Shell the NVS step under ComfyUI's embedded python (ComfyUI must be live)."""
    subprocess.run([COMFY_PY, str(HERE / "nvs.py"), "--image", str(concept),
                    "--out", str(nvs_dir), "--subject", subject], check=True)


def run_roster(roster: list, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    nvs_root = out / "_nvs"
    # Phase 1 — NVS all (ComfyUI / Qwen-Edit)
    print(f"== Phase 1: NVS for {len(roster)} characters ==")
    for c in roster:
        print(f"-- NVS {c['slug']} --")
        nvs_one(Path(c["concept"]), c["subject"], nvs_root / c["slug"])
    # Phase 2 — free Qwen-Edit, run estimators in-process (load models ONCE, loop)
    print("== Phase 2: free ComfyUI models + build 2.5D packs ==")
    _free()
    import build2p5d
    models = build2p5d.load_models()
    # monkey-free: reuse the loaded models by calling the internals per character
    for c in roster:
        slug = c["slug"]; vd = nvs_root / slug
        print(f"-- build {slug} --")
        _build_with_models(build2p5d, models, vd, slug, out)
    print(f"DONE roster -> {out}")


def _build_with_models(build2p5d, models, views_dir, slug, out):
    """Replicates build2p5d.process_and_assemble but with pre-loaded models (roster reuse)."""
    import numpy as np
    from PIL import Image
    from config import DIRECTIONS, LAYERS, TILE, SCHEMA_VERSION
    asset = out / "assets" / slug
    for l in LAYERS:
        (asset / l).mkdir(parents=True, exist_ok=True)
    per = {}
    for d in DIRECTIONS:
        f = views_dir / f"{d}.png"
        if not f.exists():
            continue
        rgb = Image.open(f).convert("RGB")
        mask = build2p5d.alpha_of(rgb, models)
        depth = np.asarray(models["depth"](rgb)["depth"], np.float32)
        nrm = np.asarray(models["normals"](rgb).prediction[0], np.float32); nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-6
        iid = models["iid"](rgb)
        vis = models["iid"].image_processor.visualize_intrinsics(iid.prediction, models["iid"].target_properties)[0]
        ao = build2p5d.ao_from_depth(depth, mask)
        per[d] = {"mask": mask, "albedo": np.asarray(vis["albedo"].convert("RGB")),
                  "rough": np.asarray(vis["roughness"].convert("L")),
                  "nrm": ((nrm * 0.5 + 0.5) * 255).astype(np.uint8),
                  "depth": ((depth - depth.min()) / (np.ptp(depth) + 1e-6) * 255).astype(np.uint8),
                  "ao": (ao * 255).astype(np.uint8)}
    bbox = build2p5d._union_bbox([per[d]["mask"] for d in per])
    files = {}
    for d, L in per.items():
        af = build2p5d._frame((L["mask"] * 255).astype(np.uint8), bbox, TILE, Image.LANCZOS, "L")
        def rgba(arr, rs=Image.LANCZOS):
            rgb = build2p5d._frame(arr, bbox, TILE, rs, "RGB"); return Image.merge("RGBA", [*rgb.split(), af])
        rgba(L["albedo"]).save(asset / "albedo" / f"{d}.png")
        rgba(L["nrm"], Image.BOX).save(asset / "normal" / f"{d}.png")
        rgba(np.dstack([L["ao"], L["rough"], np.zeros_like(L["ao"])]), Image.BOX).save(asset / "mask" / f"{d}.png")
        rgba(np.dstack([L["depth"]] * 3), Image.BOX).save(asset / "depth" / f"{d}.png")
        for layer in LAYERS:
            files[f"{layer}/{d}.png"] = build2p5d._sha(asset / layer / f"{d}.png")
    manifest = {"schema_version": SCHEMA_VERSION, "slug": slug, "name": slug.replace("-", " ").title(),
                "tileSize": TILE, "directions": list(per.keys()), "pivot": [0.5, 1.0], "layers": LAYERS,
                "maskChannels": {"r": "ao", "g": "roughness", "b": "emissive"}, "maskStatus": "real",
                "pipeline": "no-mesh-2p5d", "license": "commercial-OK",
                "engineNotes": build2p5d.ENGINE_NOTES, "files": files}
    (asset / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    build2p5d.relight_qa(asset, out / f"_relit_{slug}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roster", type=Path)
    ap.add_argument("--concept", type=Path)
    ap.add_argument("--slug")
    ap.add_argument("--subject")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    if a.roster:
        run_roster(json.loads(a.roster.read_text(encoding="utf-8")), a.out)
    else:
        if not (a.concept and a.slug and a.subject):
            raise SystemExit("single mode needs --concept --slug --subject (or use --roster)")
        run_roster([{"slug": a.slug, "concept": str(a.concept), "subject": a.subject}], a.out)


if __name__ == "__main__":
    main()
