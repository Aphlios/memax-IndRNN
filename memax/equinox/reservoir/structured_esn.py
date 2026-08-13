"""Structured reservoir computing with fast Walsh-Hadamard transforms.

This implements the structured reservoir of Dong et al., NeurIPS 2020
(https://arxiv.org/abs/2006.07310). Its arrays are frozen by default and become
trainable with ``trainable=True``. No task-specific readout is included.
"""

from collections.abc import Sequence
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, Shaped

from memax.equinox.gras import GRAS
from memax.equinox.groups import BinaryAlgebra, Module, Resettable, SetAction
from memax.equinox.reservoir._frozen import reservoir_parameter
from memax.equinox.scans import set_action_scan
from memax.mtypes import Input


def _next_power_of_two(value: int) -> int:
    if value < 1:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def normalized_hadamard_transform(x: Array) -> Array:
    """Apply an orthonormal Walsh-Hadamard transform on the last axis."""

    size = x.shape[-1]
    if size < 1 or size & (size - 1):
        raise ValueError(
            "Hadamard transform size must be a positive power of two; "
            f"got {size}"
        )

    leading_shape = x.shape[:-1]
    transformed = x
    block_size = 1
    while block_size < size:
        paired = transformed.reshape(
            *leading_shape, size // (2 * block_size), 2, block_size
        )
        left = paired[..., 0, :]
        right = paired[..., 1, :]
        transformed = jnp.stack((left + right, left - right), axis=-2).reshape(
            *leading_shape, size
        )
        block_size *= 2
    return transformed / jnp.sqrt(jnp.asarray(size, dtype=x.dtype))


class StructuredESNSetAction(SetAction):
    """One structured-reservoir transition, frozen by default."""

    diagonals: tuple[Array, Array, Array]
    bias: Array
    input_size: int
    hidden_size: int
    transform_size: int
    reservoir_scaling: float
    input_scaling: float
    leaky_rate: float
    activation: str
    trainable: bool

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        reservoir_scaling: float = 0.95,
        input_scaling: float = 0.1,
        bias_scaling: float = 0.0,
        leaky_rate: float = 0.2,
        activation: str = "erf",
        trainable: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        if input_size < 1 or hidden_size < 1:
            raise ValueError("input_size and hidden_size must be positive")
        if reservoir_scaling < 0.0:
            raise ValueError("reservoir_scaling must be non-negative")
        if input_scaling < 0.0 or bias_scaling < 0.0:
            raise ValueError("input_scaling and bias_scaling must be non-negative")
        if not 0.0 <= leaky_rate <= 1.0:
            raise ValueError("leaky_rate must be in [0, 1]")
        activation = activation.lower()
        if activation not in {"erf", "tanh"}:
            raise ValueError("activation must be 'erf' or 'tanh'")

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.transform_size = _next_power_of_two(input_size + hidden_size)
        self.reservoir_scaling = float(reservoir_scaling)
        self.input_scaling = float(input_scaling)
        self.leaky_rate = float(leaky_rate)
        self.activation = activation
        self.trainable = bool(trainable)

        diagonal_key, bias_key = jax.random.split(key)
        diagonal_keys = jax.random.split(diagonal_key, 3)
        self.diagonals = tuple(
            jax.random.rademacher(
                diagonal_key_i, (self.transform_size,), dtype=jnp.float32
            )
            for diagonal_key_i in diagonal_keys
        )
        self.bias = bias_scaling * jax.random.normal(bias_key, (hidden_size,))

    def _structured_projection(self, state: Array, input: Array) -> Array:
        padding_size = self.transform_size - self.input_size - self.hidden_size
        padded = jnp.concatenate(
            (
                self.input_scaling * input,
                self.reservoir_scaling * state,
                jnp.zeros((padding_size,), dtype=input.dtype),
            )
        )
        transformed = padded
        for diagonal in reservoir_parameter(self.diagonals, self.trainable):
            transformed = normalized_hadamard_transform(diagonal * transformed)
        return jnp.sqrt(
            jnp.asarray(self.transform_size, dtype=transformed.dtype)
        ) * transformed[: self.hidden_size]

    def __call__(self, carry: Array, input: Array) -> Array:
        projected = self._structured_projection(carry, input)
        bias = reservoir_parameter(self.bias, self.trainable)
        if self.activation == "erf":
            activated = jax.lax.erf(projected + bias)
        else:
            activated = jnp.tanh(projected + bias)
        candidate = activated / jnp.sqrt(
            jnp.asarray(self.hidden_size, dtype=activated.dtype)
        )
        return (1.0 - self.leaky_rate) * carry + self.leaky_rate * candidate

    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> Array:
        del key
        return jnp.zeros((self.hidden_size,))


class StructuredESNCell(GRAS):
    """A structured reservoir cell following the repository's GRAS contract."""

    algebra: BinaryAlgebra
    scan: object
    input_size: int
    hidden_size: int

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        reservoir_scaling: float = 0.95,
        input_scaling: float = 0.1,
        bias_scaling: float = 0.0,
        leaky_rate: float = 0.2,
        activation: str = "erf",
        trainable: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.readout_dim = int(hidden_size)
        self.algebra = Resettable(
            StructuredESNSetAction(
                input_size=input_size,
                hidden_size=hidden_size,
                reservoir_scaling=reservoir_scaling,
                input_scaling=input_scaling,
                bias_scaling=bias_scaling,
                leaky_rate=leaky_rate,
                activation=activation,
                trainable=trainable,
                key=key,
            )
        )
        self.scan = set_action_scan

    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ):
        del key
        return x

    def backward_map(
        self,
        h,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Array:
        del x, key
        state, _ = h
        return state


class StructuredESN(Module):
    """One or more structured ESN cells, frozen by default, without a readout.

    Features from all layers are concatenated along the final axis.
    """

    layers: tuple[StructuredESNCell, ...]
    input_size: int
    hidden_size: int
    num_layers: int
    readout_dim: int
    trainable: bool

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        reservoir_scaling: float = 0.95,
        input_scaling: float = 0.1,
        bias_scaling: float = 0.0,
        leaky_rate: float = 0.2,
        activation: str = "erf",
        trainable: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        if input_size < 1 or hidden_size < 1 or num_layers < 1:
            raise ValueError("input_size, hidden_size, and num_layers must be positive")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.readout_dim = int(hidden_size * num_layers)
        self.trainable = bool(trainable)

        layer_keys = jax.random.split(key, num_layers)
        self.layers = tuple(
            StructuredESNCell(
                input_size=input_size if index == 0 else hidden_size,
                hidden_size=hidden_size,
                reservoir_scaling=reservoir_scaling,
                input_scaling=input_scaling,
                bias_scaling=bias_scaling,
                leaky_rate=leaky_rate,
                activation=activation,
                trainable=trainable,
                key=layer_keys[index],
            )
            for index in range(num_layers)
        )

    def __call__(
        self,
        h: Sequence,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ):
        if len(h) != self.num_layers:
            raise ValueError(f"expected {self.num_layers} layer states; got {len(h)}")
        _, start = x
        if key is None:
            layer_keys = (None,) * self.num_layers
        else:
            layer_keys = tuple(jax.random.split(key, self.num_layers))

        next_states = []
        features = []
        layer_input = x
        for layer, layer_state, layer_key in zip(self.layers, h, layer_keys):
            next_state, output = layer(layer_state, layer_input, key=layer_key)
            next_states.append(next_state)
            features.append(output)
            layer_input = (output, start)
        return tuple(next_states), jnp.concatenate(features, axis=-1)

    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> tuple:
        if key is None:
            layer_keys = (None,) * self.num_layers
        else:
            layer_keys = tuple(jax.random.split(key, self.num_layers))
        return tuple(
            layer.initialize_carry(layer_key)
            for layer, layer_key in zip(self.layers, layer_keys)
        )

    def latest_recurrent_state(self, hs) -> tuple:
        return tuple(
            layer.latest_recurrent_state(layer_hs)
            for layer, layer_hs in zip(self.layers, hs)
        )
