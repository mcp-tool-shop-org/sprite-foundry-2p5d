# sprite-foundry-2p5d

**The no-mesh 2.5D sprite pipeline.** Turns a single front concept into an 8-direction, 4-layer, engine-relightable 2.5D character — with **no 3D mesh** in the loop. Replaces the retired `trellis-sprite-pipeline`, whose single-image→3D reconstruction destroyed faces and fine detail on anything that wasn't hooded.

## Why no mesh

A single-image 3D reconstruction (TRELLIS) is lossy: it blobs faces and softens detail, so the rendered sprite is always a downgrade from the concept. This pipeline keeps the concept's 2D fidelity and gets the *other* two things the mesh was doing — directional consistency and the normal/depth for 2.5D — without it:

```
front concept (Qwen-Image + house LoRA)
  │
  ├─ nvs.py        Qwen-Image-Edit-2511 → 8 directional 2D views (identity held, faces survive)
  │
  └─ build2p5d.py  per view, from the 2D sprite alone:
        BiRefNet         → alpha matte
        Marigold-IID     → DE-LIT albedo + roughness   (strips baked shading; engine lights it)
        Marigold-normals → real surface normals         (folds, volume — true 2.5D)
        Depth-Anything-V2-Base → depth
        (AO from depth)
     → foot-anchored 512px 4-layer pack (albedo / normal / mask[R=AO,G=rough,B=emissive] / depth)
     → engine relight
```

## Layers (per direction, 512×512, schema 2.0.0)

| Layer | From | Use |
|---|---|---|
| `albedo` | Marigold-IID (de-lit) | the sprite; the engine lights it |
| `normal` | Marigold-normals | real-time directional lighting (true surface) |
| `mask` | AO (depth) + roughness (Marigold-IID) | R=AO, G=roughness, B=emissive |
| `depth` | Depth-Anything-V2-Base | parallax / tooling |

## Run

```bash
# one character (TRELLIS2_PY = the rig's torch-CUDA venv; ComfyUI live, watchdog armed):
TRELLIS2_PY pipeline.py --concept noble_front.png --slug noble \
    --subject "royal-blue coat with gold trim" --out out/

# a roster (two phases — NVS under ComfyUI, then estimators after /free):
TRELLIS2_PY pipeline.py --roster roster.json --out out/
# roster.json = [{"slug","concept","subject"}, ...]
```

Steps can also run standalone: `COMFY_PY nvs.py …` then `TRELLIS2_PY build2p5d.py --views <dir> --slug <s> --out <pack>`.

## Licensing (all commercial-usable)

Qwen-Image / Qwen-Image-Edit-2511 (Apache-2.0) · Marigold (Apache code; SD2-derived weights under OpenRAIL — commercial OK) · **Depth-Anything-V2-Base** (Apache — the *Large* variant is CC-BY-NC, never use it) · BiRefNet (MIT). No InsightFace / IP-Adapter / FLUX.1-dev / IC-Light (all non-commercial), no lossy 3D mesh.

## Ops guardrails (carried from the studio)

- Arm the GPU watchdog before any run; `ollama stop` residents.
- NVS pins the Qwen-Edit text encoder to CPU (`device:"cpu"`) + POSTs `/free` — the fp8 edit UNET + the VL encoder otherwise peak the 32 GB ceiling.
- The roster's two-phase design (NVS, then `/free`, then estimators) keeps Qwen-Edit (~20 GB) and the diffusers estimators (~7.5 GB) from coexisting.

## Status

Validated end-to-end on the noble (face preserved, identity-consistent 8-direction turnaround, real 2.5D relight). Tuning knobs that earned their place: BiRefNet alpha (clean cut vs a pale-bg threshold), foot-anchored union-bbox framing, key+fill+AO+roughness-spec relight.
