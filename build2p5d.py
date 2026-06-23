"""Step 2 — turn the 8 directional 2D views into a 4-layer 2.5D pack (NO mesh).

Per view, from single-image estimation:
  alpha    <- BiRefNet matte (clean cut; threshold fallback)
  albedo   <- Marigold-IID-appearance (DE-LIT base colour; strips baked shading)
  roughness<- Marigold-IID-appearance
  normal   <- Marigold-normals (real surface orientation)
  depth    <- Depth-Anything-V2-Base
  AO       <- screen-space approx from depth (recessed = darker)
Then union-bbox foot-anchor to 512px and write albedo/normal/mask(R=AO,G=rough,B=emissive)/
depth + manifest (schema 2.0.0). Plus a relit QA turnaround.

    <TRELLIS2_PY> build2p5d.py --views <nvs_dir> --slug noble --out <pack_dir>
"""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", "E:/AI-Models/hf-cache"))
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from config import (DIRECTIONS, TILE, SCHEMA_VERSION, LAYERS,
                    MARIGOLD_NORMALS, MARIGOLD_IID, DEPTH_ANYTHING, BIREFNET)

ENGINE_NOTES = {
    "godot": "albedo + normal via CanvasTexture; mask.r=AO -> ambient, mask.g=roughness -> specular; depth=parallax",
    "unity_urp": "_BaseMap + _NormalMap + _MaskMap (R=AO,G=roughness); depth tooling-only",
    "phaser": "albedo + normal (_n); mask/depth tooling-only",
}


def load_models():
    import torch
    from transformers import pipeline as hf_pipeline, AutoModelForImageSegmentation
    from diffusers import MarigoldNormalsPipeline, MarigoldIntrinsicsPipeline
    m = {}
    m["depth"] = hf_pipeline("depth-estimation", model=DEPTH_ANYTHING, device=0)
    m["normals"] = MarigoldNormalsPipeline.from_pretrained(MARIGOLD_NORMALS, torch_dtype=torch.float16).to("cuda")
    m["iid"] = MarigoldIntrinsicsPipeline.from_pretrained(MARIGOLD_IID, torch_dtype=torch.float16).to("cuda")
    try:
        bm = AutoModelForImageSegmentation.from_pretrained(BIREFNET, trust_remote_code=True).to("cuda").eval().half()
        m["birefnet"] = bm
    except Exception as e:
        print(f"  (BiRefNet unavailable: {e}; using pale-bg threshold alpha)", flush=True)
        m["birefnet"] = None
    return m


def alpha_of(rgb, models):
    bm = models.get("birefnet")
    if bm is not None:
        import torch
        from torchvision import transforms
        t = transforms.Compose([transforms.Resize((1024, 1024)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        x = t(rgb).unsqueeze(0).to("cuda").half()
        with torch.no_grad():
            pred = bm(x)[-1].sigmoid().cpu()[0, 0].float().numpy()
        a = np.asarray(Image.fromarray((pred * 255).astype(np.uint8)).resize(rgb.size), np.float32) / 255.0
        return a
    a = np.asarray(rgb, np.float32)
    d = np.linalg.norm(a - np.array([200, 200, 195]), axis=-1)
    m = (d > 34).astype(np.uint8) * 255
    m = np.asarray(Image.fromarray(m).filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3)), np.float32) / 255.0
    return m


def ao_from_depth(depth, mask):
    d = depth.astype(np.float32); d = (d - d.min()) / (np.ptp(d) + 1e-6)
    blur = np.asarray(Image.fromarray((d * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(9)), np.float32) / 255.0
    ao = np.clip(1.0 - np.clip(blur - d, 0, 1) * 3.0, 0.35, 1.0)
    return ao * mask + (1 - mask)


def _union_bbox(alphas):
    box = None
    for a in alphas:
        ys, xs = np.where(a > 0.5)
        if len(xs) == 0:
            continue
        bb = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        box = bb if box is None else (min(box[0], bb[0]), min(box[1], bb[1]), max(box[2], bb[2]), max(box[3], bb[3]))
    return box


def _frame(arr, bbox, side, resample, mode):
    im = Image.fromarray(arr, mode).crop(bbox)
    w, h = im.size; s = max(w, h)
    canvas = Image.new(mode, (s, s), 0 if mode == "L" else (0, 0, 0))
    canvas.paste(im, ((s - w) // 2, s - h))   # foot-anchored, bottom-center
    return canvas.resize((side, side), resample)


def _sha(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()


def process_and_assemble(views_dir: Path, slug: str, out: Path):
    models = load_models()
    asset = out / "assets" / slug
    for l in LAYERS:
        (asset / l).mkdir(parents=True, exist_ok=True)

    per = {}   # dir -> dict of native-res arrays
    for d in DIRECTIONS:
        f = views_dir / f"{d}.png"
        if not f.exists():
            print(f"  WARN missing view {d}"); continue
        rgb = Image.open(f).convert("RGB")
        mask = alpha_of(rgb, models)
        depth = np.asarray(models["depth"](rgb)["depth"], np.float32)
        nrm = np.asarray(models["normals"](rgb).prediction[0], np.float32)
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-6
        iid = models["iid"](rgb)
        vis = models["iid"].image_processor.visualize_intrinsics(iid.prediction, models["iid"].target_properties)[0]
        albedo = np.asarray(vis["albedo"].convert("RGB"))
        rough = np.asarray(vis["roughness"].convert("L"))
        ao = ao_from_depth(depth, mask)
        per[d] = {"mask": mask, "albedo": albedo, "rough": rough,
                  "nrm": ((nrm * 0.5 + 0.5) * 255).astype(np.uint8),
                  "depth": ((depth - depth.min()) / (np.ptp(depth) + 1e-6) * 255).astype(np.uint8),
                  "ao": (ao * 255).astype(np.uint8)}
        print(f"  {d}: layers estimated", flush=True)

    bbox = _union_bbox([per[d]["mask"] for d in per])
    files = {}
    for d, L in per.items():
        a8 = (L["mask"] * 255).astype(np.uint8)
        af = _frame(a8, bbox, TILE, Image.LANCZOS, "L")
        def rgba(arr, resample=Image.LANCZOS):
            rgb = _frame(arr, bbox, TILE, resample, "RGB")
            return Image.merge("RGBA", [*rgb.split(), af])
        rgba(L["albedo"]).save(asset / "albedo" / f"{d}.png")
        rgba(L["nrm"], Image.BOX).save(asset / "normal" / f"{d}.png")
        maskrgb = np.dstack([L["ao"], L["rough"], np.zeros_like(L["ao"])])
        rgba(maskrgb, Image.BOX).save(asset / "mask" / f"{d}.png")
        rgba(np.dstack([L["depth"]] * 3), Image.BOX).save(asset / "depth" / f"{d}.png")
        for layer in LAYERS:
            files[f"{layer}/{d}.png"] = _sha(asset / layer / f"{d}.png")

    manifest = {"schema_version": SCHEMA_VERSION, "slug": slug, "name": slug.replace("-", " ").title(),
                "tileSize": TILE, "directions": list(per.keys()), "pivot": [0.5, 1.0], "layers": LAYERS,
                "maskChannels": {"r": "ao", "g": "roughness", "b": "emissive"}, "maskStatus": "real",
                "pipeline": "no-mesh-2p5d (Qwen-Edit NVS + Marigold-IID/normals + DepthAnythingV2 + BiRefNet)",
                "license": "commercial-OK", "engineNotes": ENGINE_NOTES, "files": files}
    (asset / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  pack -> {asset} ({len(files)} files)")
    return asset


def relight_qa(asset: Path, out_png: Path):
    """Tuned engine relight (de-lit albedo + normal + AO, key+fill+spec) across the directions."""
    def load(layer, d):
        return np.asarray(Image.open(asset / layer / f"{d}.png").convert("RGBA"), np.float32) / 255.0
    dirs = [d for d in DIRECTIONS if (asset / "albedo" / f"{d}.png").exists()]
    cols = 4; rows = (len(dirs) + cols - 1) // cols
    cv = Image.new("RGB", (cols * 320, rows * 340 + 28), (20, 20, 24)); dr = ImageDraw.Draw(cv)
    dr.text((6, 6), f"{asset.name} — 2.5D relit (no mesh)", fill=(250, 250, 180))
    key = np.array([-0.5, 0.45, 0.75], np.float32); key /= np.linalg.norm(key)
    fill = np.array([0.6, -0.1, 0.6], np.float32); fill /= np.linalg.norm(fill)
    for i, d in enumerate(dirs):
        alb = load("albedo", d); nrm = load("normal", d); msk = load("mask", d)
        a = alb[..., 3:4]; base = alb[..., :3]; ao = msk[..., 0:1]; rough = msk[..., 1:2]
        n = np.empty_like(base); n[..., 0] = nrm[..., 0] * 2 - 1; n[..., 1] = nrm[..., 1] * 2 - 1; n[..., 2] = nrm[..., 2] * 2 - 1
        n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6
        ndk = np.clip((n * key).sum(-1, keepdims=True), 0, 1); ndf = np.clip((n * fill).sum(-1, keepdims=True), 0, 1)
        R = 2 * ndk * n - key; rdv = np.clip((R * np.array([0, 0, 1.], np.float32)).sum(-1, keepdims=True), 0, 1)
        spec = (1 - rough) * (rdv ** 24) * 0.5
        lit = np.clip(base * (0.22 * ao + 0.95 * ndk + 0.30 * ndf) + spec, 0, 1)
        comp = lit * a + np.array([0.13, 0.13, 0.16]) * (1 - a)
        r, c = divmod(i, cols); y = 28 + r * 340
        dr.text((c * 320 + 6, y), d, fill=(200, 225, 160))
        cv.paste(Image.fromarray((comp * 255).astype(np.uint8)).resize((314, 314), Image.LANCZOS), (c * 320 + 3, y + 16))
    cv.save(out_png); print(f"  relit QA -> {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--views", required=True, type=Path, help="dir with the 8 directional PNGs")
    ap.add_argument("--slug", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--no-qa", action="store_true")
    a = ap.parse_args()
    asset = process_and_assemble(a.views, a.slug, a.out)
    if not a.no_qa:
        relight_qa(asset, a.out / f"_relit_{a.slug}.png")
    print(f"DONE 2.5D pack -> {asset}")


if __name__ == "__main__":
    main()
