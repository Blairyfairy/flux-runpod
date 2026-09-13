"""
RunPod Serverless handler for FLUX.1-dev text-to-image generation.

Key design points that prevent the most common worker failures:
- HF_TOKEN is read from the environment (set it on the RunPod endpoint).
- Cache directory prefers a network volume when present so the model is
  not re-downloaded on every cold start.
- Model is loaded once at worker start (outside the handler).
- Clear logging so you can see exactly why a worker dies in the RunPod console.
"""

import base64
import io
import os
import traceback

import torch
from diffusers import FluxPipeline
import runpod

MODEL_ID = "black-forest-labs/FLUX.1-dev"

# Prefer a persistent network volume when it is attached.
# RunPod mounts network volumes at /runpod-volume on serverless workers.
# Fall back to the image-local path otherwise.
if os.path.isdir("/runpod-volume"):
    cache_root = "/runpod-volume/hf-cache"
    os.makedirs(cache_root, exist_ok=True)
    os.environ["HF_HOME"] = cache_root
    os.environ["TRANSFORMERS_CACHE"] = cache_root
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_root
    print(f"Using network volume cache: {cache_root}")
else:
    print("No /runpod-volume found – using image-local cache /models/hf-cache")

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not hf_token:
    print("WARNING: HF_TOKEN is not set. Gated model download will fail.")
else:
    print("HF_TOKEN is present.")

print(f"Loading {MODEL_ID} ...")

try:
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        token=hf_token,
    )
    # CPU offload keeps the model usable on 24 GB cards
    pipe.enable_model_cpu_offload()
    print("Model loaded — worker is ready.")
except Exception as e:
    print("FATAL: Model failed to load")
    print(traceback.format_exc())
    # Re-raise so the worker is marked unhealthy and RunPod restarts it
    # with a clear log instead of silently accepting jobs that will fail.
    raise


def _image_to_base64(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _round_to_16(value: int) -> int:
    return max(256, min(1536, (value // 16) * 16))


def _parse_input(job_input: dict) -> dict:
    prompt = job_input.get("prompt")
    if not prompt or not isinstance(prompt, str):
        raise ValueError("`input.prompt` is required and must be a non-empty string.")

    return {
        "prompt": prompt,
        "height": _round_to_16(int(job_input.get("height", 1024))),
        "width": _round_to_16(int(job_input.get("width", 1024))),
        "num_inference_steps": max(1, min(100, int(job_input.get("num_inference_steps", 28)))),
        "guidance_scale": float(job_input.get("guidance_scale", 3.5)),
        "seed": job_input.get("seed"),
    }


def handler(job):
    try:
        params = _parse_input(job.get("input", {}))

        generator = None
        if params["seed"] is not None:
            generator = torch.Generator("cpu").manual_seed(int(params["seed"]))

        result = pipe(
            prompt=params["prompt"],
            height=params["height"],
            width=params["width"],
            guidance_scale=params["guidance_scale"],
            num_inference_steps=params["num_inference_steps"],
            max_sequence_length=512,
            generator=generator,
        )

        return {
            "image_base64": _image_to_base64(result.images[0]),
            "parameters": params,
        }

    except Exception as exc:
        return {
            "error": str(exc),
            "trace": traceback.format_exc(),
        }


runpod.serverless.start({"handler": handler})
