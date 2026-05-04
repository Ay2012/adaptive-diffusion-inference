from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from adaptive_diffusion import (
    DEFAULT_MODEL_PATH,
    DEFAULT_RAW_STEPS,
    SinglePromptConfig,
    SinglePromptResult,
    StepDecisionError,
    run_single_prompt,
)
from adaptive_diffusion.benchmark import get_device_and_dtype, import_torch, load_pipeline


def main() -> None:
    st.set_page_config(page_title="Adaptive vs Raw Generation", layout="wide")
    st.title("Adaptive vs Raw Generation")

    settings = _render_controls()
    prompt = st.text_area("Prompt", height=140)
    submitted = st.button("Submit", type="primary")

    if submitted:
        if not prompt.strip():
            st.warning("Enter a prompt before submitting.")
        else:
            st.session_state.pop("last_result", None)
            _run_generation(prompt, settings)

    last_result = st.session_state.get("last_result")
    if last_result is not None:
        _render_result(last_result)


def _render_controls() -> SinglePromptConfig:
    with st.sidebar:
        st.header("Advanced")
        model_path = st.text_input("Model path", value=DEFAULT_MODEL_PATH)
        ollama_model = st.text_input(
            "Ollama model", value=os.getenv("OLLAMA_MODEL", "phi4-mini")
        )
        raw_steps = st.number_input(
            "Raw steps",
            min_value=1,
            max_value=100,
            value=DEFAULT_RAW_STEPS,
            step=1,
        )
        seed = st.number_input("Seed", min_value=0, value=42, step=1)
        guidance_scale = st.number_input(
            "Guidance scale",
            min_value=0.0,
            max_value=30.0,
            value=7.5,
            step=0.5,
        )
        width = st.number_input(
            "Width",
            min_value=64,
            max_value=1024,
            value=512,
            step=64,
        )
        height = st.number_input(
            "Height",
            min_value=64,
            max_value=1024,
            value=512,
            step=64,
        )

    return SinglePromptConfig(
        model_path=model_path.strip() or DEFAULT_MODEL_PATH,
        raw_steps=int(raw_steps),
        guidance_scale=float(guidance_scale),
        height=int(height),
        width=int(width),
        seed=int(seed),
        ollama_model=ollama_model.strip() or None,
    )


def _run_generation(prompt: str, config: SinglePromptConfig) -> None:
    try:
        with st.spinner("Generating adaptive and raw images..."):
            pipe = _load_cached_pipeline(config.model_path)
            result = run_single_prompt(prompt, config, pipeline=pipe)
    except FileNotFoundError as exc:
        st.error(str(exc))
    except StepDecisionError as exc:
        st.error(f"Ollama step selection failed: {exc}")
    except RuntimeError as exc:
        st.error(str(exc))
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Generation failed: {exc}")
    else:
        st.session_state["last_result"] = result


@st.cache_resource(show_spinner=False)
def _load_cached_pipeline(model_path: str):
    path = Path(model_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Model path not found: {path}")

    torch = import_torch()
    device, dtype = get_device_and_dtype(torch)
    return load_pipeline(str(path), device, dtype)


def _render_result(result: SinglePromptResult) -> None:
    st.subheader("Results")

    adaptive_latency = _format_seconds(result.adaptive_latency)
    raw_latency = _format_seconds(result.raw_latency)
    latency_delta = result.raw_latency - result.adaptive_latency
    faster_label = "Adaptive faster" if latency_delta >= 0 else "Raw faster"

    metric_cols = st.columns(3)
    metric_cols[0].metric("Adaptive latency", adaptive_latency)
    metric_cols[1].metric("Raw latency", raw_latency)
    metric_cols[2].metric(faster_label, _format_seconds(abs(latency_delta)))

    image_cols = st.columns(2)
    with image_cols[0]:
        st.markdown(f"**Adaptive Run ({result.adaptive_steps} steps)**")
        st.image(result.adaptive_image, use_container_width=True)
    with image_cols[1]:
        st.markdown("**Raw Run**")
        st.image(result.raw_image, use_container_width=True)


def _format_seconds(value: float) -> str:
    return f"{value:.2f}s"


if __name__ == "__main__":
    main()
