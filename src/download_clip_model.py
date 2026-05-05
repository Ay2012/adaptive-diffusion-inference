import argparse
import os
from pathlib import Path


MODEL_ID = "openai/clip-vit-base-patch32"
MODEL_NAME = "clip-vit-base-patch32"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVE_DIR = PROJECT_ROOT / "artifacts" / "models" / MODEL_NAME
CACHE_DIR = PROJECT_ROOT / "artifacts" / "hf_cache"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and save the local CLIP scoring model."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    parse_args(argv)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(CACHE_DIR))

    print(f"Project root : {PROJECT_ROOT}")
    print(f"HF cache     : {os.environ['HF_HOME']}")
    print(f"Model ID     : {MODEL_ID}")
    print(f"Save path    : {SAVE_DIR}")
    print("Downloading and saving CLIP model...\n")

    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(MODEL_ID)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)

    model.save_pretrained(SAVE_DIR)
    processor.save_pretrained(SAVE_DIR)

    print("\nDone.")
    print(f"CLIP model saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
