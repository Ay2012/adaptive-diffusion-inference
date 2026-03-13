from pathlib import Path
import os

from diffusers import StableDiffusionPipeline
import torch


MODEL_ID = "sd-legacy/stable-diffusion-v1-5"
MODEL_NAME = "stable-diffusion-v1-5"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVE_DIR = PROJECT_ROOT / "artifacts" / "models" / MODEL_NAME
CACHE_DIR = PROJECT_ROOT / "artifacts" / "hf_cache"


def main() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(CACHE_DIR))

   
    if torch.backends.mps.is_available():
        dtype = torch.float16
        device = "mps"

    else:
        print("MPS is not available. Falling back to CPU, which may be very slow. Consider using a GPU for better performance or use CUDA for your device")
        dtype = torch.float32
        device = "cpu"

    print(f"Project root : {PROJECT_ROOT}")
    print(f"HF cache     : {os.environ['HF_HOME']}")
    print(f"Model ID     : {MODEL_ID}")
    print(f"Save path    : {SAVE_DIR}")
    print(f"Device       : {device}")
    print(f"Dtype        : {dtype}")
    print("Downloading and saving model...\n")

    
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

   
    pipe.save_pretrained(SAVE_DIR)

    print("\nDone.")
    print(f"Model saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()