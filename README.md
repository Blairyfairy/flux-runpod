# FLUX.1-dev — RunPod Serverless Worker

A GPU-backed [RunPod Serverless](https://www.runpod.io/serverless-gpu) worker that runs [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) for text-to-image generation. Send a prompt as JSON, get back a base64-encoded PNG.

## Pull

```bash
docker pull blairyfairy/flux-runpod:v1
```

## Overview

| | |
|---|---|
| Base image | `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime` |
| Model | `black-forest-labs/FLUX.1-dev`, loaded once at worker startup |
| Pipeline | 🤗 Diffusers `FluxPipeline` with `enable_model_cpu_offload()` so it fits on 24 GB-class GPUs |
| Entrypoint | `python3 -u src/handler.py` via the RunPod Python SDK (`serverless.start()`) |
| Verified on | RunPod's 24 GB Pro tier (RTX 4090 workers) |

Model weights (~24 GB) are **not baked into the image** — they're pulled from the Hugging Face Hub the first time a worker starts. Attach a RunPod network volume and the worker will cache the weights there (`/runpod-volume/hf-cache`) so later cold starts don't re-download; without one, it falls back to an image-local cache that resets every cold start.

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) | Yes | Hugging Face token with access to the gated FLUX.1-dev repo. Accept the model's license on Hugging Face before first use. |

If the token is missing, the worker logs a clear warning and the model load fails fast instead of hanging.

## Input

```json
{
  "input": {
    "prompt": "A cat holding a sign that says hello world",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 28,
    "guidance_scale": 3.5,
    "seed": 42
  }
}
```

Only `prompt` is required.

- `height` / `width` — default `1024`; rounded to the nearest 16 px, clamped between 256 and 1536
- `num_inference_steps` — default `28`; clamped between 1 and 100
- `guidance_scale` — default `3.5`
- `seed` — optional, for reproducible output

## Output

```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "parameters": {
    "prompt": "A cat holding a sign that says hello world",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 28,
    "guidance_scale": 3.5,
    "seed": 42
  }
}
```

On failure, the handler returns `{"error": "...", "trace": "..."}` instead of crashing, so a failed job still reports a clear message in the RunPod console.

## Deploying on RunPod

1. Create a Serverless Endpoint using this image
2. Choose a GPU with at least 24 GB VRAM
3. Set `HF_TOKEN` as an endpoint environment variable / secret
4. (Recommended) attach a network volume to persist the model cache across cold starts
5. Send requests through the RunPod API/SDK using the input shape above

## Licensing note

FLUX.1-dev is distributed under Black Forest Labs' FLUX.1 [dev] Non-Commercial License — free for personal, research, and evaluation use, but commercial use requires a separate license from Black Forest Labs. This image only packages the inference code; check the [model license](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md) before using generated outputs commercially.
