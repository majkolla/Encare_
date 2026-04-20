from __future__ import annotations

from src.models.adaptive_mixture_extra import AdaptiveMixtureExtraSynthesizer
from src.models.ctgan_model import CTGANSynthesizer
from src.models.gaussian_copula_model import GaussianCopulaSynthesizer
from src.models.hybrid_model import HybridSynthesizer
from src.models.independent_baseline import IndependentBaselineSynthesizer


MODEL_REGISTRY = {
    "baseline": IndependentBaselineSynthesizer,
    "independent_baseline": IndependentBaselineSynthesizer,
    "copula": GaussianCopulaSynthesizer,
    "gaussian_copula": GaussianCopulaSynthesizer,
    "ctgan": CTGANSynthesizer,
    "hybrid": HybridSynthesizer,
    "adaptive_extra": AdaptiveMixtureExtraSynthesizer,
}


def create_model(model_name: str, seed: int):
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {model_name}")
    return MODEL_REGISTRY[model_name](seed=seed)
