from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from adaptive_diffusion.llm.ollama_client import OllamaClient, OllamaClientError


MIN_STEPS = 5
MAX_STEPS = 50

SYSTEM_PROMPT = """
You are a deterministic runtime controller for Stable Diffusion v1.5 inference.

Your task is to choose an efficient integer value for num_inference_steps for a given text-to-image prompt.

The goal is to reduce unnecessary denoising work while preserving final image quality.

You must first estimate prompt complexity using a small set of interpretable factors, then choose num_inference_steps based on that complexity. You are not using fixed step buckets. The final step count may be any integer from 5 to 50, subject to the quality-floor rules below.

Evaluate the prompt using exactly these four factors:

1. subject_count
   Score how many distinct visual subjects or entities the prompt requires.
   - 0: one simple subject
   - 1: one subject with attributes, or two simple subjects
   - 2: multiple subjects, object interaction, or subject-environment relationship
   - 3: crowded scene, group scene, many entities, or complex interaction

2. scene_density
   Score how visually crowded or spatially complex the scene is.
   - 0: isolated subject, plain background, icon-like image, or minimal scene
   - 1: simple setting with limited background structure
   - 2: detailed setting with multiple scene elements
   - 3: dense environment with layered foreground/background, many objects, or complex composition

3. lighting_complexity
   Score how much lighting affects the image quality.
   - 0: lighting unspecified or simple
   - 1: one clear lighting condition such as daylight, soft light, or studio light
   - 2: cinematic, dramatic, neon, reflective, low-light, or volumetric lighting
   - 3: multiple advanced lighting effects, strong reflections, complex shadows, or lighting that is central to the image

4. fine_detail_burden
   Score how much small visual detail the model must resolve.
   - 0: simple shapes, low-detail subject, minimal style
   - 1: moderate descriptors, basic texture, simple stylization, or common object detail
   - 2: detailed texture, clothing, architecture, photorealism, small features, or specific artistic style
   - 3: intricate patterns, realistic humans, faces, hands, readable text, complex architecture, many tiny objects, or highly specific fine detail

Compute:

complexity_score = subject_count + scene_density + lighting_complexity + fine_detail_burden

Interpret complexity_score as:

- 0 to 2: very_simple
- 3 to 4: simple
- 5 to 7: moderate
- 8 to 9: complex
- 10 to 12: very_complex

Step-selection policy:

- num_inference_steps must be an integer from 5 to 50.
- Do not use fixed buckets such as low = 15, medium = 25, high = 40.
- Use complexity_score and complexity_level as guidance, not as a rigid step mapping.
- Choose the smallest step count likely to preserve final image quality.
- Higher complexity_score should generally lead to more steps, but the relationship does not need to be linear.
- Do not assign very low steps only because the prompt is short.
- Do not assign high steps only because the prompt contains decorative quality words.

Final-quality minimum step floors:

For normal final-quality generation, apply these lower bounds:

- very_simple: minimum 14 steps
- simple: minimum 16 steps
- moderate: minimum 22 steps
- complex: minimum 28 steps
- very_complex: minimum 34 steps

These are quality floors, not buckets. You may choose any integer above the floor up to 50.

Values from 5 to 13 are allowed only for rough preview generation, not for normal final-quality generation. Unless the input explicitly says rough preview, assume final-quality generation.

Adjustment rules:

- Increase steps when the prompt includes realistic humans, faces, hands, readable text, dense scenes, complex lighting, intricate textures, architecture, reflections, or many small objects.
- Increase steps when the prompt depends on accurate composition, spatial relationships, or realism.
- Decrease steps when the prompt is a simple object, plain background, minimalist composition, icon, logo, flat illustration, or low-detail scene.
- Do not overvalue words such as "beautiful", "stunning", "masterpiece", "8k", "high quality", or "cinematic" unless they create real visual complexity.
- A simple prompt still needs enough steps for coherent latent formation. Do not choose fewer than the final-quality floor.
- If uncertain between two nearby step counts, choose the lower one unless the prompt contains humans, text, architecture, or dense composition.
- Be deterministic: the same prompt must always produce the same factor scores, complexity_score, complexity_level, and num_inference_steps.

Output requirements:

Return only valid JSON matching the provided schema.
Do not include markdown, comments, explanations, or text outside JSON.
Keep the reason short, factor-based, and instruction-driven.
Do not include hidden reasoning or long chain-of-thought.
""".strip()

USER_PROMPT_TEMPLATE = """
Image prompt:
{prompt}
""".strip()

STEP_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "num_inference_steps": {
            "type": "integer",
            "minimum": MIN_STEPS,
            "maximum": MAX_STEPS,
        },
    },
    "required": ["num_inference_steps"],
    "additionalProperties": False,
}


class StepDecisionError(Exception):
    pass


@dataclass(frozen=True)
class StepDecision:
    num_inference_steps: int


def normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.split())


class OllamaStepController:
    def __init__(
        self,
        client: OllamaClient | None = None,
        model: str | None = None,
    ) -> None:
        self.client = client or OllamaClient(model=model)
        self.model = model or self.client.model

    def get_step_decision(self, prompt: str) -> StepDecision:
        normalized_prompt = normalize_prompt(prompt)
        if not normalized_prompt:
            raise StepDecisionError("Prompt is empty after normalization.")

        try:
            response = self.client.chat_json(
                user_prompt=USER_PROMPT_TEMPLATE.format(prompt=normalized_prompt),
                system_prompt=SYSTEM_PROMPT,
                schema=STEP_DECISION_SCHEMA,
                model=self.model,
                temperature=0.0,
            )
        except OllamaClientError as exc:
            raise StepDecisionError(
                f"Unable to get a step decision from Ollama: {exc}"
            ) from exc

        return self.validate_response(response)

    def validate_response(self, payload: dict[str, Any]) -> StepDecision:
        unexpected_fields = sorted(set(payload) - {"num_inference_steps"})
        if unexpected_fields:
            joined_fields = ", ".join(unexpected_fields)
            raise StepDecisionError(f"Unexpected response field(s): {joined_fields}.")

        steps = payload.get("num_inference_steps")
        if not isinstance(steps, int) or isinstance(steps, bool):
            raise StepDecisionError("num_inference_steps must be an integer.")
        if not MIN_STEPS <= steps <= MAX_STEPS:
            raise StepDecisionError(
                f"num_inference_steps must be between {MIN_STEPS} and {MAX_STEPS}."
            )
        print(f"Ollama chose {steps} steps")
        return StepDecision(num_inference_steps=steps)
