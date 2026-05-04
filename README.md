# Model-Agnostic Runtime Adaptors for Efficient Diffusion Inference

This project benchmarks one simple question: how much latency changes when a prompt
uses an adaptive diffusion path instead of a raw fixed-step generation path.

For each prompt, the runner now performs both paths:

- adaptive: Ollama chooses `num_inference_steps`, latent-convergence early stopping is attached, then the image is generated
- raw: the image is generated directly with a fixed step count

The benchmark writes one CSV row per prompt with exactly three columns:

```text
prompt,adaptive_latency,raw_latency
```

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
│   │   │   └── ollama_client.py
│   │   ├── benchmark.py
│   │   ├── early_stopping.py
│   │   └── step_controller.py
│   ├── benchmark_runner.py
│   └── download_model.py
├── prompts_complexity.txt
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

Download and pin the Stable Diffusion v1.5 weights locally:

```text
python src/download_model.py
```

This saves the model under:

```text
artifacts/models/stable-diffusion-v1-5/
```

## Ollama

The adaptive path calls Ollama once per prompt to choose an integer step count
between 5 and 50.

```text
export OLLAMA_URL=http://localhost:11434
export OLLAMA_MODEL=phi4-mini
```

You can override the model for a single run with `--ollama-model`.

## Running The Benchmark

```text
python src/benchmark_runner.py \
  --run-name latency_eval \
  --prompt-file prompts_complexity.txt
```

Optional controls:

```text
python src/benchmark_runner.py \
  --run-name latency_eval \
  --prompt-file prompts_complexity.txt \
  --raw-steps 50 \
  --guidance-scale 7.5 \
  --height 512 \
  --width 512 \
  --seed 42 \
  --ollama-model phi4-mini
```

Each run:

- loads prompts from a text file, one non-empty prompt per line
- loads the local Stable Diffusion pipeline once
- runs adaptive and raw generation for every prompt
- saves adaptive images to `artifacts/outputs/<run-name>/adaptive/`
- saves raw images to `artifacts/outputs/<run-name>/raw/`
- saves latency results to `artifacts/results/<run-name>.csv`

Latency excludes pipeline load time and image saving time. `adaptive_latency`
includes the Ollama step decision, early-stop setup, and image generation.
`raw_latency` includes only raw image generation.

## Running The Streamlit UI

```text
streamlit run src/streamlit_app.py
```

The UI accepts one prompt at a time, runs the adaptive and raw paths, and shows
both generated images with their latencies. UI images are kept in memory for
display and are not saved to `artifacts/outputs/`.

## Notes

- `artifacts/hf_cache/` stores the Hugging Face cache and should not be committed.
- `artifacts/models/` stores local model weights and should stay out of GitHub.
- `artifacts/outputs/` and `artifacts/results/` are generated experiment outputs.

## License

This project is currently for academic and research use.
