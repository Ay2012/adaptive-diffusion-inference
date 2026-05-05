from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable

from adaptive_diffusion.clip_metrics import (
    DEFAULT_CLIP_MODEL_PATH,
    LocalClipScorer,
)
from adaptive_diffusion.early_stopping import LatentConvergenceEarlyStopper
from adaptive_diffusion.step_controller import OllamaStepController


DEFAULT_MODEL_PATH = "artifacts/models/stable-diffusion-v1-5"
DEFAULT_OUTPUT_ROOT = "artifacts/outputs"
DEFAULT_PROMPT_FILE = "prompts_complexity.txt"
DEFAULT_RESULTS_DIR = "artifacts/results"
DEFAULT_RAW_STEPS = 50

EARLY_STOP_TAU = 0.03
EARLY_STOP_PATIENCE = 3
EARLY_STOP_MIN_STEPS = 8

CSV_FIELDNAMES = [
    "prompt",
    "adaptive_latency",
    "raw_latency",
    "adaptive_clip_score",
    "raw_clip_score",
]


@dataclass(frozen=True)
class BenchmarkConfig:
    run_name: str
    prompt_file: str = DEFAULT_PROMPT_FILE
    model_path: str = DEFAULT_MODEL_PATH
    output_root: str = DEFAULT_OUTPUT_ROOT
    results_dir: str = DEFAULT_RESULTS_DIR
    raw_steps: int = DEFAULT_RAW_STEPS
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512
    seed: int = 42
    ollama_model: str | None = None
    clip_model_path: str = DEFAULT_CLIP_MODEL_PATH


@dataclass(frozen=True)
class SinglePromptConfig:
    model_path: str = DEFAULT_MODEL_PATH
    raw_steps: int = DEFAULT_RAW_STEPS
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512
    seed: int = 42
    ollama_model: str | None = None


@dataclass(frozen=True)
class SinglePromptResult:
    prompt: str
    adaptive_image: Any
    raw_image: Any
    adaptive_latency: float
    raw_latency: float
    adaptive_steps: int


def load_prompts(prompt_file: str) -> list[str]:
    path = Path(prompt_file)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    prompts = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not prompts:
        raise ValueError(f"Prompt file is empty: {path}")
    return prompts


def import_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "torch is required to run the benchmark. Install it before running inference."
        ) from exc
    return torch


def get_device_and_dtype(torch_module: Any) -> tuple[str, Any]:
    if torch_module.backends.mps.is_available():
        return "mps", torch_module.float16

    print("MPS is not available. Falling back to CPU, which may be slow.")
    return "cpu", torch_module.float32


def load_pipeline(model_path: str, device: str, dtype: Any) -> Any:
    try:
        from diffusers import StableDiffusionPipeline
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "diffusers is required to load the Stable Diffusion pipeline."
        ) from exc

    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def run_benchmark(
    config: BenchmarkConfig,
    *,
    pipeline: Any | None = None,
    step_controller: Any | None = None,
    clip_scorer: Any | None = None,
    torch_module: Any | None = None,
    perf_counter: Callable[[], float] = time.perf_counter,
) -> Path:
    _validate_config(config)
    scorer = clip_scorer or _build_clip_scorer(config)

    prompts = load_prompts(config.prompt_file)
    output_dir = Path(config.output_root) / config.run_name
    adaptive_dir = output_dir / "adaptive"
    raw_dir = output_dir / "raw"
    results_dir = Path(config.results_dir)
    csv_path = results_dir / f"{config.run_name}.csv"

    adaptive_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    torch = torch_module or import_torch()
    device, dtype = get_device_and_dtype(torch)
    pipe = pipeline or load_pipeline(config.model_path, device, dtype)
    controller = step_controller or OllamaStepController(model=config.ollama_model)

    rows = []
    for index, prompt in enumerate(prompts, start=1):
        prompt_seed = config.seed + index - 1

        adaptive_image, adaptive_latency, _adaptive_steps = _run_adaptive_path(
            pipe=pipe,
            prompt=prompt,
            controller=controller,
            config=config,
            torch_module=torch,
            device=device,
            seed=prompt_seed,
            perf_counter=perf_counter,
        )
        raw_image, raw_latency = _run_raw_path(
            pipe=pipe,
            prompt=prompt,
            config=config,
            torch_module=torch,
            device=device,
            seed=prompt_seed,
            perf_counter=perf_counter,
        )

        adaptive_clip_score = scorer.score(prompt, adaptive_image)
        raw_clip_score = scorer.score(prompt, raw_image)

        adaptive_image.save(adaptive_dir / f"{index}.png")
        raw_image.save(raw_dir / f"{index}.png")

        rows.append(
            {
                "prompt": prompt,
                "adaptive_latency": round(adaptive_latency, 4),
                "raw_latency": round(raw_latency, 4),
                "adaptive_clip_score": round(adaptive_clip_score, 4),
                "raw_clip_score": round(raw_clip_score, 4),
            }
        )
        print(
            f"[{index}/{len(prompts)}] prompt benchmarked | "
            f"adaptive={adaptive_latency:.4f}s | raw={raw_latency:.4f}s | "
            f"adaptive_clip={adaptive_clip_score:.4f} | raw_clip={raw_clip_score:.4f}"
        )

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nMetrics saved to: {csv_path}")
    return csv_path


def run_single_prompt(
    prompt: str,
    config: SinglePromptConfig | None = None,
    *,
    pipeline: Any | None = None,
    step_controller: Any | None = None,
    torch_module: Any | None = None,
    perf_counter: Callable[[], float] = time.perf_counter,
) -> SinglePromptResult:
    resolved_config = config or SinglePromptConfig()
    _validate_single_prompt_config(resolved_config)

    stripped_prompt = prompt.strip()
    if not stripped_prompt:
        raise ValueError("prompt is required.")

    torch = torch_module or import_torch()
    device, dtype = get_device_and_dtype(torch)
    pipe = pipeline or load_pipeline(resolved_config.model_path, device, dtype)
    controller = step_controller or OllamaStepController(
        model=resolved_config.ollama_model
    )

    adaptive_image, adaptive_latency, adaptive_steps = _run_adaptive_path(
        pipe=pipe,
        prompt=stripped_prompt,
        controller=controller,
        config=resolved_config,
        torch_module=torch,
        device=device,
        seed=resolved_config.seed,
        perf_counter=perf_counter,
    )
    raw_image, raw_latency = _run_raw_path(
        pipe=pipe,
        prompt=stripped_prompt,
        config=resolved_config,
        torch_module=torch,
        device=device,
        seed=resolved_config.seed,
        perf_counter=perf_counter,
    )

    return SinglePromptResult(
        prompt=stripped_prompt,
        adaptive_image=adaptive_image,
        raw_image=raw_image,
        adaptive_latency=adaptive_latency,
        raw_latency=raw_latency,
        adaptive_steps=adaptive_steps,
    )


def _run_adaptive_path(
    *,
    pipe: Any,
    prompt: str,
    controller: Any,
    config: BenchmarkConfig | SinglePromptConfig,
    torch_module: Any,
    device: str,
    seed: int,
    perf_counter: Callable[[], float],
) -> tuple[Any, float, int]:
    start = perf_counter()
    decision = controller.get_step_decision(prompt)
    early_stopper = LatentConvergenceEarlyStopper(
        tau=EARLY_STOP_TAU,
        patience=EARLY_STOP_PATIENCE,
        min_steps=EARLY_STOP_MIN_STEPS,
        num_inference_steps=decision.num_inference_steps,
    )
    result = pipe(
        prompt=prompt,
        num_inference_steps=decision.num_inference_steps,
        guidance_scale=config.guidance_scale,
        height=config.height,
        width=config.width,
        generator=_build_generator(torch_module, device, seed),
        callback_on_step_end=early_stopper,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    latency = perf_counter() - start
    return result.images[0], latency, decision.num_inference_steps


def _run_raw_path(
    *,
    pipe: Any,
    prompt: str,
    config: BenchmarkConfig | SinglePromptConfig,
    torch_module: Any,
    device: str,
    seed: int,
    perf_counter: Callable[[], float],
) -> tuple[Any, float]:
    start = perf_counter()
    result = pipe(
        prompt=prompt,
        num_inference_steps=config.raw_steps,
        guidance_scale=config.guidance_scale,
        height=config.height,
        width=config.width,
        generator=_build_generator(torch_module, device, seed),
    )
    latency = perf_counter() - start
    return result.images[0], latency


def _build_generator(torch_module: Any, device: str, seed: int) -> Any:
    return torch_module.Generator(device=device).manual_seed(seed)


def _build_clip_scorer(config: BenchmarkConfig) -> LocalClipScorer:
    return LocalClipScorer(model_path=config.clip_model_path)


def _validate_config(config: BenchmarkConfig) -> None:
    if not config.run_name.strip():
        raise ValueError("run_name is required.")
    if config.raw_steps < 1:
        raise ValueError("raw_steps must be at least 1.")
    if config.height < 1 or config.width < 1:
        raise ValueError("height and width must be positive.")
    if not config.clip_model_path.strip():
        raise ValueError("clip_model_path is required.")


def _validate_single_prompt_config(config: SinglePromptConfig) -> None:
    if config.raw_steps < 1:
        raise ValueError("raw_steps must be at least 1.")
    if config.height < 1 or config.width < 1:
        raise ValueError("height and width must be positive.")
