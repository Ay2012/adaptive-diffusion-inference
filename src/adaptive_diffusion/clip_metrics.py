from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_CLIP_MODEL_PATH = "artifacts/models/clip-vit-base-patch32"


class ClipMetricError(RuntimeError):
    """Raised when CLIP scoring cannot produce a valid score."""


@dataclass(frozen=True)
class LocalClipScorer:
    model_path: str = DEFAULT_CLIP_MODEL_PATH
    device: str | None = None
    model: Any | None = field(default=None, repr=False)
    processor: Any | None = field(default=None, repr=False)
    torch_module: Any | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        path = Path(self.model_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"CLIP model path not found: {path}. "
                "Run `python src/download_clip_model.py` first."
            )

        torch = self.torch_module or _import_torch()
        device = self.device or _get_device(torch)
        model = self.model
        processor = self.processor

        if model is None or processor is None:
            if not (path / "config.json").exists():
                raise FileNotFoundError(
                    f"CLIP model files not found in: {path}. "
                    "Run `python src/download_clip_model.py` first."
                )
            model, processor = _load_model_and_processor(str(path))

        model = model.to(device)
        if hasattr(model, "eval"):
            model.eval()

        object.__setattr__(self, "model_path", str(path))
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "processor", processor)
        object.__setattr__(self, "torch_module", torch)

    def score(self, prompt: str, image: Any) -> float:
        stripped_prompt = prompt.strip()
        if not stripped_prompt:
            raise ValueError("prompt is required for CLIP scoring.")

        inputs = self.processor(
            text=[stripped_prompt],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = _move_inputs_to_device(inputs, self.device)
        text_inputs = _select_inputs(inputs, ("input_ids", "attention_mask"))
        image_inputs = _select_inputs(inputs, ("pixel_values",))

        if "input_ids" not in text_inputs:
            raise ClipMetricError("CLIP processor did not return text input IDs.")
        if "pixel_values" not in image_inputs:
            raise ClipMetricError("CLIP processor did not return image pixel values.")

        with self.torch_module.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
            image_features = self.model.get_image_features(**image_inputs)

            text_features = _extract_feature_tensor(text_features, "text")
            image_features = _extract_feature_tensor(image_features, "image")
            text_features = _normalize(text_features)
            image_features = _normalize(image_features)
            cosine = (text_features * image_features).sum(dim=-1).item()

        return max(0.0, min(1.0, (float(cosine) + 1.0) / 2.0))


def _import_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "torch is required to run CLIP scoring. Install it before scoring images."
        ) from exc
    return torch


def _get_device(torch_module: Any) -> str:
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_model_and_processor(model_path: str) -> tuple[Any, Any]:
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "transformers is required to load the CLIP model. "
            "Install it before scoring images."
        ) from exc

    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)
    return model, processor


def _move_inputs_to_device(inputs: Any, device: str) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)

    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def _select_inputs(inputs: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: inputs[key] for key in keys if key in inputs}


def _extract_feature_tensor(features: Any, kind: str) -> Any:
    if hasattr(features, "norm"):
        return features

    pooled_features = getattr(features, "pooler_output", None)
    if pooled_features is not None and hasattr(pooled_features, "norm"):
        return pooled_features

    raise ClipMetricError(
        f"CLIP {kind} features did not include pooled embeddings."
    )


def _normalize(features: Any) -> Any:
    return features / features.norm(dim=-1, keepdim=True)
