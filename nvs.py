"""Step 1 — 2D directional views via Qwen-Image-Edit-2511 (NO mesh).

Rotates the camera around a front concept to synthesize the other 7 of the 8 sprite
directions, holding identity. Keeps the concept's 2D fidelity (faces survive — the exact
thing TRELLIS mesh reconstruction destroyed). Runs against a live ComfyUI on COMFY_URL
with the Qwen-Edit text-encoder pinned to CPU (the VRAM fix).

    <COMFY_PY> nvs.py --image front.png --out dir/ --subject "<garment/identity descriptor>"
"""
from __future__ import annotations
import argparse, json, shutil, sys, os, time, urllib.request, uuid
from pathlib import Path
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # ComfyUI embedded python's ._pth omits the script dir
from config import (COMFY_URL, COMFY_OUTPUT_DIR, COMFY_INPUT_DIR,
                    QWEN_EDIT_UNET, QWEN_CLIP, QWEN_VAE, NVS_VIEWS)

NEG = ("anime, manga, cel shaded, multiple characters, two people, duplicate, different character, different "
       "outfit, changed clothing, blurry, low quality, deformed, extra limbs, text, watermark, cropped, cut off, "
       "dark background, harsh shadows, 3d render, photorealistic")

_TEMPLATES = {
 "front_right": "Rotate the camera to show this exact same character from a three-quarter FRONT-RIGHT angle, turned so we see the front and the right side of the figure.",
 "right":       "Rotate the camera to show this exact same character from the RIGHT side, a clean full right-profile view.",
 "back_right":  "Rotate the camera to show this exact same character from a three-quarter REAR-RIGHT angle, mostly the back of the figure with the right side visible.",
 "back":        "Rotate the camera to show this exact same character from directly BEHIND, a full rear view, the back of the figure facing away, the back of the {subject} and the hair visible.",
 "back_left":   "Rotate the camera to show this exact same character from a three-quarter REAR-LEFT angle, mostly the back of the figure with the left side visible.",
 "left":        "Rotate the camera to show this exact same character from the LEFT side, a clean full left-profile view.",
 "front_left":  "Rotate the camera to show this exact same character from a three-quarter FRONT-LEFT angle, turned so we see the front and the left side of the figure.",
}


def view_prompt(view: str, subject: str) -> str:
    keep = (f" Keep the IDENTICAL character, identical {subject}, identical proportions, identical painterly "
            f"hand-painted art style, identical plain pale grey background, full body, both feet on the ground, "
            f"standing, single figure.")
    return _TEMPLATES[view].format(subject=subject) + keep


def build_graph(image_filename: str, positive: str, seed: int, steps: int, cfg: float, prefix: str) -> dict:
    """Qwen-Edit-2511 NVS graph. CLIPLoader device:'cpu' is the load-bearing VRAM fix."""
    return {
        "37": {"class_type": "UNETLoader", "inputs": {"unet_name": QWEN_EDIT_UNET, "weight_dtype": "default"}},
        "38": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN_CLIP, "type": "qwen_image", "device": "cpu"}},
        "39": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN_VAE}},
        "10": {"class_type": "LoadImage", "inputs": {"image": image_filename}},
        "11": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["10", 0]}},
        "6": {"class_type": "TextEncodeQwenImageEditPlus",
              "inputs": {"clip": ["38", 0], "prompt": positive, "vae": ["39", 0], "image1": ["11", 0]}},
        "7": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["38", 0], "prompt": NEG, "vae": ["39", 0]}},
        "12": {"class_type": "VAEEncode", "inputs": {"pixels": ["11", 0], "vae": ["39", 0]}},
        "66": {"class_type": "ModelSamplingAuraFlow", "inputs": {"shift": 3.1, "model": ["37", 0]}},
        "3": {"class_type": "KSampler",
              "inputs": {"seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple",
                         "denoise": 1.0, "model": ["66", 0], "positive": ["6", 0], "negative": ["7", 0],
                         "latent_image": ["12", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["39", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["8", 0]}},
    }


def _free():
    try:
        urllib.request.urlopen(urllib.request.Request(
            f"{COMFY_URL}/free", json.dumps({"unload_models": True, "free_memory": True}).encode(),
            {"Content-Type": "application/json"}), timeout=30).read()
    except Exception:
        pass


def _submit(graph: dict, dst: Path) -> Path:
    pid = json.loads(urllib.request.urlopen(urllib.request.Request(
        f"{COMFY_URL}/prompt", json.dumps({"prompt": graph, "client_id": uuid.uuid4().hex}).encode(),
        {"Content-Type": "application/json"}), timeout=120).read())["prompt_id"]
    for _ in range(2400):
        hist = json.loads(urllib.request.urlopen(f"{COMFY_URL}/history/{pid}", timeout=30).read())
        if pid in hist:
            st = hist[pid].get("status", {})
            if st.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI error: {st.get('messages')}")
            if hist[pid].get("outputs"):
                img = hist[pid]["outputs"]["9"]["images"][0]
                src = Path(COMFY_OUTPUT_DIR) / img["subfolder"] / img["filename"]
                for _w in range(120):
                    if src.exists():
                        Image.open(src).convert("RGB").save(dst); return dst
                    time.sleep(0.5)
        time.sleep(0.5)
    raise RuntimeError("timed out")


def synthesize(image: Path, out: Path, subject: str, steps: int = 20, cfg: float = 3.0, seed: int = 770700) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    Image.open(image).convert("RGB").save(out / "front.png")          # front = the concept itself
    staged = Path(COMFY_INPUT_DIR) / f"nvs_src_{image.stem}.png"
    shutil.copy(image, staged); _free()
    receipt = {"src": image.name, "subject": subject, "views": ["front"]}
    s = seed
    for v in NVS_VIEWS:
        dst = out / f"{v}.png"
        _submit(build_graph(staged.name, view_prompt(v, subject), s, steps, cfg, f"nvs_{image.stem}_{v}"), dst)
        print(f"  {v} (seed {s}) -> {dst.name}", flush=True)
        receipt["views"].append(v); s += 1
    (out / "nvs.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--subject", required=True, help="garment/identity descriptor woven into every view")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=770700)
    a = ap.parse_args()
    synthesize(a.image, a.out, a.subject, steps=a.steps, cfg=a.cfg, seed=a.seed)
    print(f"DONE 8 directional views -> {a.out}")


if __name__ == "__main__":
    main()
