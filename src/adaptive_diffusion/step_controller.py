from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Literal

from adaptive_diffusion.llm.ollama_client import OllamaClient, OllamaClientError


PROMPT_TEMPLATE_VERSION = "exact_steps_v2"
SCHEMA_VERSION = "exact_steps_v2"
DEFAULT_DECISION_CACHE = "artifacts/results/ollama_step_cache.jsonl"

SYSTEM_PROMPT = """
You are a deterministic controller for Stable Diffusion v1.5 inference.
Estimate the minimum number of denoising steps needed to preserve image quality
for a single text-to-image prompt.
Before choosing the step count, reason about the prompt using a small set of
interpretable factors: subject_count, scene_density, realism_requirement,
lighting_complexity, and fine_detail_burden.
Keep the reasoning lightweight and instruction-driven.
Return only valid JSON that matches the provided schema.
""".strip()

USER_PROMPT_TEMPLATE = """
Stable Diffusion setup:
- model: Stable Diffusion v1.5
- resolution: 512x512
- guidance_scale: 7.5

Task:
Given the image prompt below, predict the minimum number of denoising steps that
should preserve quality for the generated image.

Constraints:
- num_inference_steps must be an integer between {min_steps} and {max_steps}
- confidence must be a number between 0 and 1
- reason must be a short explanation
- subject_count must be a small integer estimate for the number of prominent subjects
- scene_density must be one of: low, medium, high
- realism_requirement must be one of: low, medium, high
- lighting_complexity must be one of: low, medium, high
- fine_detail_burden must be one of: low, medium, high

Choose the exact step count only after considering those factors. Avoid generic
replies; use the prompt details to make a specific decision.

Image prompt:
{prompt}
""".strip()

FACTOR_LEVELS = ("low", "medium", "high")
FactorLevel = Literal["low", "medium", "high"]

STEP_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "num_inference_steps": {"type": "integer"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
        "subject_count": {"type": "integer"},
        "scene_density": {"type": "string", "enum": list(FACTOR_LEVELS)},
        "realism_requirement": {"type": "string", "enum": list(FACTOR_LEVELS)},
        "lighting_complexity": {"type": "string", "enum": list(FACTOR_LEVELS)},
        "fine_detail_burden": {"type": "string", "enum": list(FACTOR_LEVELS)},
    },
    "required": [
        "num_inference_steps",
        "confidence",
        "reason",
        "subject_count",
        "scene_density",
        "realism_requirement",
        "lighting_complexity",
        "fine_detail_burden",
    ],
    "additionalProperties": False,
}


class StepDecisionError(Exception):
    pass


@dataclass(frozen=True)
class StepDecision:
    num_inference_steps: int
    confidence: float
    reason: str
    subject_count: int | None
    scene_density: FactorLevel | None
    realism_requirement: FactorLevel | None
    lighting_complexity: FactorLevel | None
    fine_detail_burden: FactorLevel | None
    source: Literal["cache", "ollama"]


def normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.split())


def build_cache_key(
    normalized_prompt: str,
    ollama_model: str,
    prompt_template_version: str = PROMPT_TEMPLATE_VERSION,
    schema_version: str = SCHEMA_VERSION,
) -> str:
    cache_payload = {
        "normalized_prompt": normalized_prompt,
        "ollama_model": ollama_model,
        "prompt_template_version": prompt_template_version,
        "schema_version": schema_version,
    }
    serialized = json.dumps(cache_payload, sort_keys=True, separators=(",", ":"))
    return sha256(serialized.encode("utf-8")).hexdigest()


class OllamaStepController:
    def __init__(
        self,
        client: OllamaClient | None = None,
        model: str | None = None,
        min_steps: int = 1,
        max_steps: int = 100,
        cache_path: str | Path | None = DEFAULT_DECISION_CACHE,
        use_cache: bool = True,
    ):
        self.client = client or OllamaClient(model=model)
        self.model = model or self.client.model
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.use_cache = use_cache
        self._cache_index: dict[str, dict[str, Any]] | None = None
        self._cache_prompt_index: dict[tuple[str, str], dict[str, Any]] | None = None

    def get_step_decision(self, prompt: str) -> StepDecision:
        normalized_prompt = normalize_prompt(prompt)
        if not normalized_prompt:
            raise StepDecisionError("Prompt is empty after normalization.")

        cache_key = build_cache_key(normalized_prompt, self.model)

        if self.use_cache:
            cached_decision = self._get_cached_decision(
                cache_key=cache_key,
                normalized_prompt=normalized_prompt,
            )
            if cached_decision is not None:
                return cached_decision

        try:
            response = self.client.chat_json(
                user_prompt=USER_PROMPT_TEMPLATE.format(
                    prompt=normalized_prompt,
                    min_steps=self.min_steps,
                    max_steps=self.max_steps,
                ),
                system_prompt=SYSTEM_PROMPT,
                schema=STEP_DECISION_SCHEMA,
                model=self.model,
                temperature=0.0,
            )
        except OllamaClientError as exc:
            raise StepDecisionError(f"Unable to get a step decision from Ollama: {exc}") from exc

        decision = self.validate_response(response, source="ollama")

        if self.use_cache:
            self._store_cached_decision(cache_key, normalized_prompt, decision, response)

        return decision

    def validate_response(
        self,
        payload: dict[str, Any],
        source: Literal["cache", "ollama"],
    ) -> StepDecision:
        raw_steps = payload.get("num_inference_steps")
        if not isinstance(raw_steps, int) or isinstance(raw_steps, bool):
            raise StepDecisionError("num_inference_steps must be an integer.")
        if not self.min_steps <= raw_steps <= self.max_steps:
            raise StepDecisionError(
                f"num_inference_steps must be between {self.min_steps} and {self.max_steps}."
            )

        raw_confidence = payload.get("confidence")
        if not isinstance(raw_confidence, (int, float)) or isinstance(raw_confidence, bool):
            raise StepDecisionError("confidence must be a number.")
        confidence = float(raw_confidence)
        if not 0.0 <= confidence <= 1.0:
            raise StepDecisionError("confidence must be between 0 and 1.")

        raw_reason = payload.get("reason")
        if not isinstance(raw_reason, str):
            raise StepDecisionError("reason must be a string.")
        reason = raw_reason.strip()
        if not reason:
            raise StepDecisionError("reason must be a non-empty string.")

        factor_field_names = (
            "subject_count",
            "scene_density",
            "realism_requirement",
            "lighting_complexity",
            "fine_detail_burden",
        )
        has_any_factor_fields = any(field_name in payload for field_name in factor_field_names)
        if source == "cache" and not has_any_factor_fields:
            return StepDecision(
                num_inference_steps=raw_steps,
                confidence=confidence,
                reason=reason,
                subject_count=None,
                scene_density=None,
                realism_requirement=None,
                lighting_complexity=None,
                fine_detail_burden=None,
                source=source,
            )

        raw_subject_count = payload.get("subject_count")
        if not isinstance(raw_subject_count, int) or isinstance(raw_subject_count, bool):
            raise StepDecisionError("subject_count must be an integer.")
        if raw_subject_count < 1:
            raise StepDecisionError("subject_count must be at least 1.")

        def validate_level(field_name: str) -> str:
            value = payload.get(field_name)
            if not isinstance(value, str):
                raise StepDecisionError(f"{field_name} must be a string.")
            normalized = value.strip().lower()
            if normalized not in FACTOR_LEVELS:
                raise StepDecisionError(
                    f"{field_name} must be one of: {', '.join(FACTOR_LEVELS)}."
                )
            return normalized

        scene_density = validate_level("scene_density")
        realism_requirement = validate_level("realism_requirement")
        lighting_complexity = validate_level("lighting_complexity")
        fine_detail_burden = validate_level("fine_detail_burden")

        return StepDecision(
            num_inference_steps=raw_steps,
            confidence=confidence,
            reason=reason,
            subject_count=raw_subject_count,
            scene_density=scene_density,
            realism_requirement=realism_requirement,
            lighting_complexity=lighting_complexity,
            fine_detail_burden=fine_detail_burden,
            source=source,
        )

    def _get_cached_decision(
        self,
        cache_key: str,
        normalized_prompt: str,
    ) -> StepDecision | None:
        cache_index = self._load_cache_index()
        cache_entry = cache_index.get(cache_key)
        if cache_entry is None and self._cache_prompt_index is not None:
            cache_entry = self._cache_prompt_index.get((normalized_prompt, self.model))
        if cache_entry is None:
            return None

        decision_payload = cache_entry.get("decision")
        if not isinstance(decision_payload, dict):
            return None

        try:
            return self.validate_response(decision_payload, source="cache")
        except StepDecisionError:
            return None

    def _store_cached_decision(
        self,
        cache_key: str,
        normalized_prompt: str,
        decision: StepDecision,
        raw_response: dict[str, Any],
    ) -> None:
        if self.cache_path is None:
            return

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        cache_entry = {
            "cache_key": cache_key,
            "normalized_prompt": normalized_prompt,
            "ollama_model": self.model,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "decision": {
                "num_inference_steps": decision.num_inference_steps,
                "confidence": decision.confidence,
                "reason": decision.reason,
                "subject_count": decision.subject_count,
                "scene_density": decision.scene_density,
                "realism_requirement": decision.realism_requirement,
                "lighting_complexity": decision.lighting_complexity,
                "fine_detail_burden": decision.fine_detail_burden,
            },
            "raw_response": raw_response,
        }

        with self.cache_path.open("a", encoding="utf-8") as cache_file:
            cache_file.write(json.dumps(cache_entry, ensure_ascii=True) + "\n")

        if self._cache_index is None:
            self._cache_index = {}
        self._cache_index[cache_key] = cache_entry

    def _load_cache_index(self) -> dict[str, dict[str, Any]]:
        if self._cache_index is not None:
            return self._cache_index

        cache_index: dict[str, dict[str, Any]] = {}
        cache_prompt_index: dict[tuple[str, str], dict[str, Any]] = {}
        if self.cache_path is not None and self.cache_path.exists():
            with self.cache_path.open(encoding="utf-8") as cache_file:
                for line in cache_file:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        cache_entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    cache_key = cache_entry.get("cache_key")
                    if isinstance(cache_key, str):
                        cache_index[cache_key] = cache_entry

                    normalized_prompt = cache_entry.get("normalized_prompt")
                    ollama_model = cache_entry.get("ollama_model")
                    if isinstance(normalized_prompt, str) and isinstance(ollama_model, str):
                        cache_prompt_index[(normalized_prompt, ollama_model)] = cache_entry

        self._cache_index = cache_index
        self._cache_prompt_index = cache_prompt_index
        return self._cache_index
