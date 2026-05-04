from __future__ import annotations

import math
from typing import Any


class LatentConvergenceEarlyStopper:
    def __init__(
        self,
        tau: float = 0.03,
        patience: int = 3,
        min_steps: int = 8,
        num_inference_steps = 50,
    ) -> None:
        if tau < 0:
            raise ValueError("tau must be non-negative.")
        if patience < 1:
            raise ValueError("patience must be at least 1.")
        if min_steps < 1:
            raise ValueError("min_steps must be at least 1.")

        self.tau = tau
        self.patience = patience
        self.min_steps = min_steps
        self.previous_latents: Any | None = None
        self.stable_count = 0
        self.min_step_fraction: float = 0.75
        self.num_inference_steps = num_inference_steps

    def __call__(
        self,
        pipeline: Any,
        step_index: int,
        timestep: Any,
        callback_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        normalized_change = self._normalized_change(callback_kwargs.get("latents"))
        if normalized_change is None:
            return callback_kwargs
        # print(f"Step {step_index}: normalized latent change = {normalized_change:.6f}")
        if step_index + 1 < self.num_inference_steps * self.min_step_fraction:
            return callback_kwargs
        if step_index + 1 < self.min_steps:
            self.stable_count = 0
            return callback_kwargs

        if normalized_change < self.tau:
            self.stable_count += 1
        else:
            self.stable_count = 0

        if self.stable_count >= self.patience:
            print("Early stopping triggered due to latent convergence.")
            setattr(pipeline, "_interrupt", True)

        return callback_kwargs

    def _normalized_change(self, latents: Any) -> float | None:
        if latents is None:
            return None

        current_latents = latents.detach().clone().float()

        if current_latents.numel() == 0:
            self.previous_latents = current_latents
            return None

        if self.previous_latents is None:
            self.previous_latents = current_latents
            return None

        eps = 1e-8

        delta_norm = (current_latents - self.previous_latents).norm(p=2)
        prev_norm = self.previous_latents.norm(p=2)

        relative_change = delta_norm / (prev_norm + eps)

        print(f"Relative latent change: {relative_change.item():.8f}")

        self.previous_latents = current_latents
        return float(relative_change.item())
