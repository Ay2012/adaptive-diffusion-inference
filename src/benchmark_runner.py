import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from adaptive_diffusion import DEFAULT_DECISION_CACHE


DEFAULT_MODEL_PATH = "artifacts/models/stable-diffusion-v1-5"
DEFAULT_OUTPUT_ROOT = "artifacts/outputs"
DEFAULT_RESULTS_DIR = "artifacts/results"
DEFAULT_DECISION_CACHE_PATH = DEFAULT_DECISION_CACHE

DEFAULT_PROMPTS = [
    "a photo of an astronaut riding a horse on mars",
    "a cinematic portrait of a medieval king in golden armor",
    "a cozy cabin in snowy mountains at sunset",
    "a futuristic city street in the rain at night",
    "a bowl of fresh fruit on a wooden table, realistic photography",
]


def import_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "torch is required to run the benchmark. Install it before running inference."
        ) from exc

    return torch


def get_device_and_dtype():
    torch = import_torch()
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    print("MPS is not available. Falling back to CPU, which may be slow.")
    return "cpu", torch.float32


def slugify(text: str) -> str:
    cleaned = "".join(c.lower() if c.isalnum() else "_" for c in text)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")[:80]


def image_appears_blank(image: Any) -> bool:
    if not hasattr(image, "getbbox") or not hasattr(image, "getextrema"):
        return False

    bbox = image.getbbox()
    extrema = image.getextrema()

    if bbox is not None:
        return False

    if isinstance(extrema, tuple) and extrema:
        if all(
            isinstance(channel_extrema, tuple)
            and len(channel_extrema) == 2
            and channel_extrema[0] == 0
            and channel_extrema[1] == 0
            for channel_extrema in extrema
        ):
            return True

    return False


def load_prompts(prompt_file: str | None):
    if prompt_file is None:
        return DEFAULT_PROMPTS

    path = Path(prompt_file)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
            raise ValueError("JSON prompt file must be a list of strings.")
        return data

    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError("Prompt file is empty.")
    return prompts


def apply_scheduler(pipe: Any, scheduler_name: str):
    scheduler_name = scheduler_name.lower()
    config = pipe.scheduler.config

    if scheduler_name == "default":
        return pipe

    try:
        from diffusers import (
            DPMSolverMultistepScheduler,
            EulerAncestralDiscreteScheduler,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "diffusers is required to switch schedulers. Install it before running inference."
        ) from exc

    if scheduler_name == "euler_a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(config)
        return pipe

    if scheduler_name == "dpmpp_2m_karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            config,
            use_karras_sigmas=True,
        )
        return pipe

    if scheduler_name == "dpmpp_2m_sde_karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            config,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
            euler_at_final=True,
        )
        return pipe

    raise ValueError(
        "Unsupported scheduler. Use one of: "
        "default, euler_a, dpmpp_2m_karras, dpmpp_2m_sde_karras"
    )


def load_pipeline(
    model_path: str,
    device: str,
    dtype,
    enable_attention_slicing: bool = False,
):
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

    if device == "mps" and enable_attention_slicing:
        pipe.enable_attention_slicing()

    pipe.set_progress_bar_config(disable=True)
    return pipe


def build_step_controller(
    min_steps: int,
    max_steps: int,
    decision_cache: str,
    use_decision_cache: bool,
    ollama_model: str | None,
):
    from adaptive_diffusion import OllamaStepController

    return OllamaStepController(
        model=ollama_model,
        min_steps=min_steps,
        max_steps=max_steps,
        cache_path=decision_cache,
        use_cache=use_decision_cache,
    )


def run_benchmark(
    model_path: str,
    prompts: list[str],
    run_name: str,
    steps: int | None,
    guidance_scale: float,
    scheduler_name: str,
    height: int,
    width: int,
    seed: int,
    output_root: str,
    results_dir: str,
    step_policy: str = "fixed",
    min_steps: int = 1,
    max_steps: int = 100,
    decision_cache: str = DEFAULT_DECISION_CACHE_PATH,
    use_decision_cache: bool = True,
    ollama_model: str | None = None,
    enable_attention_slicing: bool = False,
):
    output_dir = Path(output_root) / run_name
    results_dir = Path(results_dir)
    csv_path = results_dir / f"{run_name}.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    device, dtype = get_device_and_dtype()
    pipe = load_pipeline(
        model_path=model_path,
        device=device,
        dtype=dtype,
        enable_attention_slicing=enable_attention_slicing,
    )
    pipe = apply_scheduler(pipe, scheduler_name=scheduler_name)
    step_controller = None
    if step_policy == "ollama_exact":
        step_controller = build_step_controller(
            min_steps=min_steps,
            max_steps=max_steps,
            decision_cache=decision_cache,
            use_decision_cache=use_decision_cache,
            ollama_model=ollama_model,
        )

    rows = []

    for i, prompt in enumerate(prompts, start=1):
        torch = import_torch()
        generator = torch.Generator(device=device).manual_seed(seed)

        decision_confidence = ""
        decision_reason = ""
        decision_source = ""
        decision_subject_count = ""
        decision_scene_density = ""
        decision_realism_requirement = ""
        decision_lighting_complexity = ""
        decision_fine_detail_burden = ""
        selected_steps = steps
        selected_ollama_model = ""

        if step_policy == "ollama_exact":
            assert step_controller is not None
            decision = step_controller.get_step_decision(prompt)
            selected_steps = decision.num_inference_steps
            decision_confidence = decision.confidence
            decision_reason = decision.reason
            decision_source = decision.source
            decision_subject_count = decision.subject_count
            decision_scene_density = decision.scene_density
            decision_realism_requirement = decision.realism_requirement
            decision_lighting_complexity = decision.lighting_complexity
            decision_fine_detail_burden = decision.fine_detail_burden
            selected_ollama_model = step_controller.model

        if selected_steps is None:
            raise ValueError("No step count was resolved for this run.")

        start = time.perf_counter()
        result = pipe(
            prompt=prompt,
            num_inference_steps=selected_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        )
        latency_seconds = time.perf_counter() - start

        image = result.images[0]
        if image_appears_blank(image):
            raise RuntimeError(
                "Generation produced an all-black image. This often indicates an unstable "
                "device/runtime combination on Apple Silicon. Retry with the safer default "
                "runner settings (attention slicing disabled), and if it still happens, "
                "fall back to a more conservative runtime configuration."
            )
        image_name = f"{i:02d}_{slugify(prompt)}.png"
        image_path = output_dir / image_name
        image.save(image_path)

        row = {
            "run_name": run_name,
            "prompt_id": i,
            "prompt": prompt,
            "seed": seed,
            "steps": selected_steps,
            "step_policy": step_policy,
            "selected_steps": selected_steps,
            "guidance_scale": guidance_scale,
            "scheduler": scheduler_name,
            "height": height,
            "width": width,
            "device": device,
            "dtype": str(dtype),
            "model_path": model_path,
            "decision_confidence": decision_confidence,
            "decision_reason": decision_reason,
            "decision_source": decision_source,
            "decision_subject_count": decision_subject_count,
            "decision_scene_density": decision_scene_density,
            "decision_realism_requirement": decision_realism_requirement,
            "decision_lighting_complexity": decision_lighting_complexity,
            "decision_fine_detail_burden": decision_fine_detail_burden,
            "ollama_model": selected_ollama_model,
            "latency_seconds": round(latency_seconds, 4),
            "image_path": str(image_path),
        }
        rows.append(row)

        print(
            f"[{i}/{len(prompts)}] "
            f"Saved: {image_path} | steps={selected_steps} | scheduler={scheduler_name} | latency={latency_seconds:.4f}s"
        )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nMetrics saved to: {csv_path}")


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Reusable Stable Diffusion benchmark runner.")
    parser.add_argument("--run-name", type=str, required=True, help="Name of this benchmark run.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--step-policy", type=str, default="fixed", choices=["fixed", "ollama_exact"])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--enable-attention-slicing",
        action="store_true",
        help="Enable attention slicing. Disabled by default because it can be unstable on some MPS setups.",
    )
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="default",
        choices=["default", "euler_a", "dpmpp_2m_karras", "dpmpp_2m_sde_karras"],
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--min-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--decision-cache", type=str, default=DEFAULT_DECISION_CACHE_PATH)
    parser.add_argument(
        "--no-decision-cache",
        action="store_true",
        help="Disable the Ollama step-decision cache for this run.",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default=None,
        help="Optional Ollama model override for ollama_exact runs.",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Optional .txt or .json file with prompts. Defaults to built-in prompt list.",
    )
    args = parser.parse_args(argv)

    if args.step_policy == "fixed" and args.steps is None:
        parser.error("--steps is required when --step-policy is fixed.")
    if args.step_policy != "fixed" and args.steps is not None:
        parser.error("--steps can only be used when --step-policy is fixed.")
    if args.min_steps < 1:
        parser.error("--min-steps must be at least 1.")
    if args.max_steps < args.min_steps:
        parser.error("--max-steps must be greater than or equal to --min-steps.")

    return args


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    prompts = load_prompts(args.prompt_file)

    run_benchmark(
        model_path=args.model_path,
        prompts=prompts,
        run_name=args.run_name,
        steps=args.steps,
        step_policy=args.step_policy,
        guidance_scale=args.guidance_scale,
        scheduler_name=args.scheduler,
        height=args.height,
        width=args.width,
        seed=args.seed,
        output_root=args.output_root,
        results_dir=args.results_dir,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        decision_cache=args.decision_cache,
        use_decision_cache=not args.no_decision_cache,
        ollama_model=args.ollama_model,
        enable_attention_slicing=args.enable_attention_slicing,
    )


if __name__ == "__main__":
    main()
