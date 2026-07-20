"""Public construction interface for Equinox reservoir models."""

from typing import Any, Dict, Optional

from jaxtyping import PRNGKeyArray

from memax.equinox.groups import Module
from memax.equinox.reservoir.deep_esn import DeepESN
from memax.equinox.reservoir.paralesn import ParalESN
from memax.equinox.reservoir.structured_esn import StructuredESN


RESERVOIR_MODEL_NAMES = ("DeepESN", "StructuredESN", "ParalESN")

_RESERVOIR_MODELS = {
    "DeepESN": DeepESN,
    "StructuredESN": StructuredESN,
    "ParalESN": ParalESN,
}


def build_reservoir_model(
    model_name: str,
    input_size: int,
    hidden_size: int,
    num_layers: int = 2,
    *,
    key: PRNGKeyArray,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Module:
    """Build one fixed reservoir model without a task-specific readout.

    Args:
        model_name: One of ``DeepESN``, ``StructuredESN``, or ``ParalESN``.
        input_size: Per-timestep input feature dimension.
        hidden_size: Reservoir feature size per layer. For ``ParalESN`` with
            ``concat=True``, this is the total feature size across all layers.
        num_layers: Number of stacked reservoir layers.
        key: JAX random key used to initialize the fixed reservoir.
        model_kwargs: Model-specific options such as ``spectral_radius``,
            ``reservoir_scaling``, reservoir configs, or ``concat``.
    """

    if model_name not in _RESERVOIR_MODELS:
        available = ", ".join(RESERVOIR_MODEL_NAMES)
        raise KeyError(
            f"unknown reservoir model {model_name!r}; available models: {available}"
        )

    kwargs = dict(model_kwargs or {})
    reserved = {"input_size", "hidden_size", "num_layers", "key"} & kwargs.keys()
    if reserved:
        names = ", ".join(sorted(reserved))
        raise ValueError(
            f"pass {names} as explicit builder arguments, not model_kwargs"
        )

    model_type = _RESERVOIR_MODELS[model_name]
    return model_type(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        key=key,
        **kwargs,
    )
