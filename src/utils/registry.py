from __future__ import annotations

from src.models.gaussian_copula_model import GaussianCopulaSynthesizer
from src.models.independent_baseline import IndependentBaselineSynthesizer


MODEL_REGISTRY = {
    "baseline": IndependentBaselineSynthesizer,
    "independent_baseline": IndependentBaselineSynthesizer,
    "copula": GaussianCopulaSynthesizer,
    "gaussian_copula": GaussianCopulaSynthesizer,
}


def create_model(model_name: str, seed: int):
    if model_name not in MODEL_REGISTRY:
        available_models = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model: {model_name}. Available models: {available_models}")
    return MODEL_REGISTRY[model_name](seed=seed)
