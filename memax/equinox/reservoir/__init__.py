"""Reservoir-computing recurrent cells for the Equinox backend.

The modules in this package only produce reservoir features. They deliberately
do not include a task-specific readout layer.
"""

from memax.equinox.reservoir.build import (
    RESERVOIR_MODEL_NAMES,
    RESERVOIR_MODEL_TYPES,
    build_reservoir_model,
)
from memax.equinox.reservoir.deep_esn import DeepESN
from memax.equinox.reservoir.paralesn import (
    MixerConfig,
    ParalESN,
    ReservoirConfig,
)
from memax.equinox.reservoir.structured_esn import StructuredESN

__all__ = [
    "DeepESN",
    "MixerConfig",
    "ParalESN",
    "RESERVOIR_MODEL_NAMES",
    "RESERVOIR_MODEL_TYPES",
    "ReservoirConfig",
    "StructuredESN",
    "build_reservoir_model",
]
