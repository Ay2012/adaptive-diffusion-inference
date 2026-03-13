# Model-Agnostic Runtime Adaptors for Efficient Diffusion Inference

This project studies how to reduce diffusion model inference cost while preserving output quality. The current baseline uses Stable Diffusion v1.5 with Hugging Face Diffusers. The first stage of the project focuses on downloading the model, running fixed baseline generations, and recording latency for later comparison.

## Project Goal

The long-term goal is to build a runtime adapter that improves diffusion inference efficiency without retraining the underlying model. The baseline stage establishes a reproducible setup for measuring inference latency and output quality before introducing optimization methods such as adaptive sampling, scheduler changes, and early stopping strategies.

## Current Scope

- Download and save Stable Diffusion v1.5 locally
- Run baseline text-to-image inference
- Save generated images
- Record per-image latency and run metadata in CSV format
- Prepare a clean structure for future optimization experiments

## Project Structure

```text
.
├── artifacts/
│   ├── hf_cache/
│   ├── models/
│   ├── outputs/
│   └── results/
├── scripts/
│   ├── download_model.py
│   └── run_baseline.py
├── src/
├── requirements.txt
└── README.md
``` 

## Setup

Create and activate a virtual environment from the project root:
``` text 
python3 -m venv venv
source venv/bin/activate
```

## Install dependencies:
``` text
pip install -r requirements.txt
```

## Model Download

The model is downloaded from Hugging Face and saved locally for reproducible baseline experiments.

Run:
``` text
python scripts/download_model.py
```

This saves the baseline model to:
```text 
artifacts/models/stable-diffusion-v1-5/
```

## Baseline Inference

Run baseline generation with fixed prompts and settings:
``` text
python scripts/run_baseline.py
```

This script:

* loads the local Stable Diffusion v1.5 model

* generates one image per prompt

* saves output images in artifacts/outputs/baseline/

* saves per-image metrics in artifacts/results/baseline_metrics.csv

## Metrics Recorded

The baseline CSV currently stores:

* prompt_id

* prompt

* seed

* steps

* guidance_scale

* height

* width

* device

* dtype

* latency_seconds

* image_path

At this stage, the main measured metric is inference latency per generated image.

## Baseline Configuration

Current baseline settings:

* Model: Stable Diffusion v1.5

* Resolution: 512 x 512

* Guidance scale: 7.5

* Seed: 42

* Inference steps: 50

These values may later be compared against lower-step baselines such as 20-step generation.

Notes

* artifacts/hf_cache/ is used for Hugging Face cache and is not intended for version control.

* artifacts/models/ stores the pinned local model used for experiments.

* artifacts/outputs/ and artifacts/results/ can be tracked selectively to document baseline runs.

* Large model weights and cache files should not be pushed to GitHub.

## Next Steps

* Add 20-step baseline comparison

* Evaluate different schedulers

* Log aggregate metrics such as mean latency

* Add quality metrics such as CLIP score and LPIPS

* Implement and benchmark runtime adaptation methods

## License

This project is currently for academic and research use.