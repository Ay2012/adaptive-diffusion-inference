from pathlib import Path
import csv
import time

import torch
from diffusers import StableDiffusionPipeline


MODEL_PATH = "artifacts/models/stable-diffusion-v1-5"
OUTPUT_DIR = Path("artifacts/outputs/baseline")
RESULTS_DIR = Path("artifacts/results")
CSV_PATH = RESULTS_DIR / "baseline_metrics.csv"

PROMPTS = [
    "a photo of an astronaut riding a horse on mars",
    "a cinematic portrait of a medieval king in golden armor",
    "a cozy cabin in snowy mountains at sunset",
    "a futuristic city street in the rain at night",
    "a bowl of fresh fruit on a wooden table, realistic photography",
]

SEED = 42
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
HEIGHT = 512
WIDTH = 512


def get_device_and_dtype():
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    else:
        print("MPS is not available. Falling back to CPU, which may be very slow. Consider using a GPU for better performance or use CUDA for your device")

def slugify(text: str) -> str:
    cleaned = "".join(c.lower() if c.isalnum() else "_" for c in text)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")[:80]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device, dtype = get_device_and_dtype()

    pipe = StableDiffusionPipeline.from_pretrained(MODEL_PATH, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False)
    pipe = pipe.to(device)

    rows = []

    for i, prompt in enumerate(PROMPTS, start=1):
        generator = torch.Generator(device=device).manual_seed(SEED)

        start = time.perf_counter()
        result = pipe(
            prompt,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            height=HEIGHT,
            width=WIDTH,
            generator=generator,
        )
        latency_seconds = time.perf_counter() - start

        image = result.images[0]
        image_name = f"{i:02d}_{slugify(prompt)}.png"
        image_path = OUTPUT_DIR / image_name
        image.save(image_path)

        rows.append({
            "prompt_id": i,
            "prompt": prompt,
            "seed": SEED,
            "steps": NUM_INFERENCE_STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "height": HEIGHT,
            "width": WIDTH,
            "device": device,
            "dtype": str(dtype),
            "latency_seconds": round(latency_seconds, 4),
            "image_path": str(image_path),
        })

        print(f"[{i}/{len(PROMPTS)}] Saved: {image_path} | latency={latency_seconds:.4f}s")

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nMetrics saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()