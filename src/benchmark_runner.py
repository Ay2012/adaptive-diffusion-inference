from __future__ import annotations

import argparse

from adaptive_diffusion import (
    BenchmarkConfig,
    DEFAULT_CLIP_MODEL_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PROMPT_FILE,
    DEFAULT_RAW_STEPS,
    DEFAULT_RESULTS_DIR,
    run_benchmark,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run adaptive and raw Stable Diffusion latency benchmarks."
    )
    parser.add_argument("--run-name", required=True, help="Name of this benchmark run.")
    parser.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--raw-steps", type=int, default=DEFAULT_RAW_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ollama-model", default=None)
    parser.add_argument("--clip-model-path", default=DEFAULT_CLIP_MODEL_PATH)

    args = parser.parse_args(argv)
    if args.raw_steps < 1:
        parser.error("--raw-steps must be at least 1.")
    if args.height < 1 or args.width < 1:
        parser.error("--height and --width must be positive.")
    if not args.clip_model_path.strip():
        parser.error("--clip-model-path is required.")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_benchmark(
        BenchmarkConfig(
            run_name=args.run_name,
            prompt_file=args.prompt_file,
            model_path=args.model_path,
            output_root=args.output_root,
            results_dir=args.results_dir,
            raw_steps=args.raw_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            seed=args.seed,
            ollama_model=args.ollama_model,
            clip_model_path=args.clip_model_path,
        )
    )


if __name__ == "__main__":
    main()
