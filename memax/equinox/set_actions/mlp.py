"""Memory-free MLP layer executed through the GRAS scan interface."""

from typing import Callable, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import nn
from jaxtyping import Array, PRNGKeyArray, Shaped

from memax.equinox.gras import GRAS
from memax.equinox.groups import BinaryAlgebra, Resettable, SetAction
from memax.equinox.scans import set_action_scan
from memax.mtypes import Input, StartFlag


MLPState = Array
MLPStateWithReset = Tuple[MLPState, StartFlag]


class MLPSetAction(SetAction):
    """Apply one residual MLP independently at every timestep.

    ``carry`` is deliberately ignored. Running this operation with
    :func:`set_action_scan` therefore preserves the same sequence/reset contract
    as recurrent cells without allowing information to cross timestep
    boundaries.
    """

    recurrent_size: int
    linear: nn.Linear
    activation: eqx.Module

    def __init__(
        self,
        recurrent_size: int,
        activation: Callable[[Array], Array] = jax.nn.silu,
        *,
        key: PRNGKeyArray,
    ):
        if recurrent_size < 1:
            raise ValueError("recurrent_size must be positive")
        self.recurrent_size = int(recurrent_size)
        self.linear = nn.Linear(recurrent_size, recurrent_size, key=key)
        self.activation = nn.Lambda(activation)

    def __call__(self, carry: MLPState, input: MLPState) -> MLPState:
        del carry
        projected = self.linear(input)
        return projected + self.activation(projected)

    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> MLPState:
        del key
        return jnp.zeros((self.recurrent_size,))


class MLP(GRAS):
    """A trainable, memory-free MLP baseline with the standard GRAS API.

    The surrounding :class:`~memax.equinox.models.residual.ResidualModel`
    supplies the input embedding, per-layer mixer, and task readout. Each MLP
    layer contributes one ``Linear -> (identity + activation)`` transformation
    inside the sequential scan, but its result depends only on the current
    timestep's input.
    """

    algebra: BinaryAlgebra
    scan: Callable
    recurrent_size: int
    readout_dim: int

    def __init__(
        self,
        recurrent_size: int,
        activation: Callable[[Array], Array] = jax.nn.silu,
        trainable: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        if recurrent_size < 1:
            raise ValueError("recurrent_size must be positive")
        self.trainable = bool(trainable)
        self.recurrent_size = int(recurrent_size)
        self.readout_dim = int(recurrent_size)
        self.algebra = Resettable(
            MLPSetAction(recurrent_size, activation=activation, key=key)
        )
        self.scan = set_action_scan

    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> MLPStateWithReset:
        del key
        return x

    def backward_map(
        self,
        h: MLPStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Array:
        del x, key
        state, _ = h
        return state


__all__ = ["MLP", "MLPSetAction"]
