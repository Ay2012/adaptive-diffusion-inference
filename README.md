# Model-Agnostic Runtime Adaptors for Efficient Diffusion Inference

This repo benchmarks Stable Diffusion v1.5 under fixed-step baselines and an Ollama-guided exact-step controller. The current experiment keeps the model, scheduler family, resolution, and guidance scale fixed so the adaptation logic only changes `num_inference_steps`.

## Project Goal

The research goal is to test whether a lightweight pre-inference controller can predict the minimum number of diffusion steps needed for a prompt while still preserving image quality. In the current prototype, Ollama makes that decision before the diffusion pipeline runs, and the benchmark records the selected step count alongside latency and output paths.

## Current Scope

- Download and save Stable Diffusion v1.5 locally
- Run fixed-step baselines when needed for comparison
- Run an `ollama_exact` mode that predicts one exact step count per prompt
- Cache Ollama decisions for repeatability
- Save generated images and per-prompt benchmark metadata to CSV

## Project Structure

```text
.
├── artifacts/
│   ├── hf_cache/
│   ├── models/
│   ├── outputs/
│   └── results/
├── src/
│   ├── adaptive_diffusion/
│   │   ├── llm/
│   │   └── step_controller.py
│   ├── benchmark_runner.py
│   ├── baseline.py
│   └── download_model.py
├── requirements.txt
└── README.md
```

## Setup

Create and activate a virtual environment from the project root:

```text
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```text
pip install -r requirements.txt
```

## Model Download

Download and pin the Stable Diffusion v1.5 weights locally:

```text
python src/download_model.py
```

This saves the model under:

```text
artifacts/models/stable-diffusion-v1-5/
```

## Fixed-Step Baselines

Run the reusable benchmark runner in fixed mode:

```text
python src/benchmark_runner.py \
  --run-name baseline_50_expanded \
  --step-policy fixed \
  --steps 50 \
  --prompt-file prompts_complexity.txt
```

Each run:

- loads the local Stable Diffusion v1.5 pipeline
- generates one image per prompt
- saves images to `artifacts/outputs/<run-name>/`
- saves per-prompt metrics to `artifacts/results/<run-name>.csv`

The tracked 50-step reference run on this branch is `artifacts/results/baseline_50_expanded.csv`, which corresponds to the 8 prompts in `prompts_complexity.txt`.

## Ollama Exact-Step Mode

The adaptive mode calls Ollama once per prompt before image generation. Ollama predicts one exact integer step count, and that number is passed directly to the diffusion pipeline.

Set up Ollama locally first:

```text
export OLLAMA_URL=http://localhost:11434
export OLLAMA_MODEL=phi4-mini
```

Run the adaptive benchmark:

```text
python src/benchmark_runner.py \
  --run-name complexity_ollama_expanded_v2 \
  --step-policy ollama_exact \
  --prompt-file prompts_complexity.txt
```

Optional controls:

```text
python src/benchmark_runner.py \
  --run-name ollama_exact_capped \
  --step-policy ollama_exact \
  --min-steps 1 \
  --max-steps 100 \
  --decision-cache artifacts/results/ollama_step_cache.jsonl \
  --ollama-model phi4-mini
```

### Reproducibility

The exact-step controller stays deterministic by keeping these pieces fixed:

- one system prompt
- one user prompt template
- one JSON schema
- `temperature=0.0`

Every prompt decision is cached in `artifacts/results/ollama_step_cache.jsonl` by normalized prompt, Ollama model, prompt-template version, and schema version. Disable cache reads and writes for a run with:

```text
python src/benchmark_runner.py --run-name ollama_no_cache --step-policy ollama_exact --no-decision-cache
```

## CSV Fields

The benchmark CSV includes:

- `run_name`
- `prompt_id`
- `prompt`
- `seed`
- `steps`
- `step_policy`
- `selected_steps`
- `guidance_scale`
- `scheduler`
- `height`
- `width`
- `device`
- `dtype`
- `model_path`
- `decision_confidence`
- `decision_reason`
- `decision_source`
- `decision_subject_count`
- `decision_scene_density`
- `decision_realism_requirement`
- `decision_lighting_complexity`
- `decision_fine_detail_burden`
- `ollama_model`
- `latency_seconds`
- `image_path`

In fixed mode, the decision-specific columns are left blank. In `ollama_exact` mode, they show the controller output and whether the decision came from `ollama` or `cache`.

## Notes

- `artifacts/hf_cache/` stores the Hugging Face cache and should not be committed.
- `artifacts/models/` stores local model weights and should stay out of GitHub.
- `artifacts/outputs/` and `artifacts/results/` can be tracked selectively for experiment records.
- The current tracked experiment records are the 8-prompt fixed-step baseline in `artifacts/results/baseline_50_expanded.csv` and the 8-prompt adaptive run in `artifacts/results/complexity_ollama_expanded_v2.csv`.
- The current prototype evaluates quality mainly by comparing adaptive outputs to fixed-step baselines, especially the existing 50-step reference runs.

## License

This project is currently for academic and research use.
