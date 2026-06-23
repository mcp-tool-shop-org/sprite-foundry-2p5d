"""Single source of truth for the no-mesh 2.5D sprite pipeline.

Pipeline (NO MESH — supersedes the retired trellis-sprite-pipeline):
  front concept (Qwen-Image + house LoRA)
   -> nvs.py        : Qwen-Image-Edit-2511 8-direction 2D directional views (ComfyUI)
   -> build2p5d.py  : BiRefNet alpha + Marigold-IID de-lit albedo/roughness
                      + Marigold-normals + Depth-Anything-V2 depth -> AO -> 4-layer pack
   -> relight QA preview

Every model here is commercial-usable: Qwen (Apache), Marigold (Apache code / OpenRAIL
SD2 weights — commercial OK), Depth-Anything-V2-**Base** (Apache; the Large variant is
CC-BY-NC — never use Large), BiRefNet (MIT). NO InsightFace / IP-Adapter / FLUX.1-dev /
IC-Light (all non-commercial) and NO lossy single-image 3D mesh.
"""
import os

# --- runtimes -------------------------------------------------------------
# build2p5d (diffusers estimators) runs under the trellis2 venv torch; nvs runs under
# ComfyUI's embedded python against the live ComfyUI server.
TRELLIS2_PY = os.environ.get("TRELLIS2_PY", "E:/AI-Models/trellis2-env/Scripts/python.exe")
COMFY_PY = os.environ.get("COMFY_PY", "E:/AI-Models/ComfyUI_windows_portable/python_embeded/python.exe")
HF_HOME = os.environ.get("HF_HOME", "E:/AI-Models/hf-cache")

# --- ComfyUI (NVS step) ---------------------------------------------------
COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
_COMFY_ROOT = "E:/AI-Models/ComfyUI_windows_portable/ComfyUI"
COMFY_OUTPUT_DIR = os.environ.get("COMFY_OUTPUT_DIR", f"{_COMFY_ROOT}/output")
COMFY_INPUT_DIR = os.environ.get("COMFY_INPUT_DIR", f"{_COMFY_ROOT}/input")

# Qwen-Image-Edit-2511 NVS models (CLIPLoader pinned device:"cpu" + POST /free — the fp8
# edit UNET + the VL text-encoder otherwise peak the 32GB ceiling and trip the watchdog).
QWEN_EDIT_UNET = os.environ.get("QWEN_EDIT_UNET", "qwen_image_edit_2511_fp8mixed.safetensors")
QWEN_CLIP = os.environ.get("QWEN_CLIP", "qwen_2.5_vl_7b_fp8_scaled.safetensors")
QWEN_VAE = os.environ.get("QWEN_VAE", "qwen_image_vae.safetensors")

# --- single-image 2.5D estimators (build2p5d step) ------------------------
MARIGOLD_NORMALS = os.environ.get("MARIGOLD_NORMALS", "prs-eth/marigold-normals-v1-1")
MARIGOLD_IID = os.environ.get("MARIGOLD_IID", "prs-eth/marigold-iid-appearance-v1-1")  # albedo + roughness + metallicity
DEPTH_ANYTHING = os.environ.get("DEPTH_ANYTHING", "depth-anything/Depth-Anything-V2-Base-hf")  # Base = Apache; NOT Large (NC)
BIREFNET = os.environ.get("BIREFNET", "ZhengPeng7/BiRefNet")  # MIT matte for the alpha cut

# --- contract -------------------------------------------------------------
DIRECTIONS = ["front", "front_left", "left", "back_left", "back", "back_right", "right", "front_right"]
# the 7 the NVS step synthesizes from the front (front is the concept itself)
NVS_VIEWS = ["front_right", "right", "back_right", "back", "back_left", "left", "front_left"]
TILE = 512
SCHEMA_VERSION = "2.0.0"
LAYERS = ["albedo", "normal", "mask", "depth"]  # mask = R:AO G:roughness B:emissive
