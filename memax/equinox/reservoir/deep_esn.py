"""Deep Echo State Network recurrent cells.

Reservoir parameter gradients are stopped by default while gradients still
flow through the reservoir to an upstream encoder. Pass ``trainable=True`` to
optimize the reservoir arrays. No readout is included;
:class:`DeepESN` returns the concatenated states of all layers.
"""

from collections.abc import Callable, Sequence
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, Shaped

from memax.equinox.gras import GRAS
from memax.equinox.groups import BinaryAlgebra, Module, Resettable, SetAction
from memax.equinox.reservoir._frozen import reservoir_parameter
from memax.equinox.scans import set_action_scan
from memax.mtypes import Input


def _validate_sizes(input_size: int, hidden_size: int) -> None:
    if input_size < 1 or hidden_size < 1:
        raise ValueError("input_size and hidden_size must be positive")


def _sparse_uniform(
    key: PRNGKeyArray,
    shape: tuple[int, ...],
    density: float,
    scale: float = 1.0,
) -> Array:
    """Sample a uniformly distributed matrix with an independent sparse mask."""

    if not 0.0 <= density <= 1.0:
        raise ValueError("density must be in [0, 1]")
    weight_key, mask_key = jax.random.split(key)
    weight = jax.random.uniform(
        weight_key, shape, minval=-scale, maxval=scale
    )
    if density < 1.0:
        weight = weight * jax.random.bernoulli(mask_key, density, shape)
    return weight


def esn_reservoir_init(
    key: PRNGKeyArray,
    shape: tuple[int, int],
    spectral_radius: float,
    density: float = 0.1,
) -> Array:
    """Initialize a sparse square reservoir with the requested spectral radius."""

    if shape[0] != shape[1]:
        raise ValueError(f"reservoir matrix must be square; got {shape}")
    if spectral_radius < 0.0:
        raise ValueError("spectral_radius must be non-negative")
    weight = _sparse_uniform(key, shape, density)
    current_radius = jnp.max(jnp.abs(jnp.linalg.eigvals(weight)))
    eps = jnp.finfo(weight.dtype).eps
    scale = jnp.where(current_radius > eps, spectral_radius / current_radius, 0.0)
    return weight * scale


def esn_input_init(
    key: PRNGKeyArray,
    shape: tuple[int, ...],
    scaling: float,
    density: float = 1.0,
) -> Array:
    """Initialize a fixed ESN input projection."""

    if scaling < 0.0:
        raise ValueError("input_scaling must be non-negative")
    return _sparse_uniform(key, shape, density, scale=scaling)


class ESNSetAction(SetAction):
    """One leaky ESN transition, frozen unless ``trainable=True``."""

    recurrent_kernel: Array
    input_kernel: Array
    bias: Array
    input_size: int
    hidden_size: int
    leaky_rate: float
    activation: Callable[[Array], Array]
    trainable: bool

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        spectral_radius: float = 0.95,
        leaky_rate: float = 0.2,
        input_scaling: float = 0.1,
        density: float = 0.04,
        activation: Callable[[Array], Array] = jax.nn.tanh,
        trainable: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        _validate_sizes(input_size, hidden_size)
        if not 0.0 <= leaky_rate <= 1.0:
            raise ValueError("leaky_rate must be in [0, 1]")

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.leaky_rate = float(leaky_rate)
        self.activation = activation
        self.trainable = bool(trainable)

        reservoir_key, input_key, bias_key = jax.random.split(key, 3)
        self.recurrent_kernel = esn_reservoir_init(
            reservoir_key,
            (hidden_size, hidden_size),
            spectral_radius,
            density,
        )
        self.input_kernel = esn_input_init(
            input_key, (hidden_size, input_size), input_scaling
        )
        self.bias = esn_input_init(bias_key, (hidden_size,), input_scaling)

    def __call__(self, carry: Array, input: Array) -> Array:
        recurrent_kernel = reservoir_parameter(self.recurrent_kernel, self.trainable)
        input_kernel = reservoir_parameter(self.input_kernel, self.trainable)
        bias = reservoir_parameter(self.bias, self.trainable)
        candidate = self.activation(
            recurrent_kernel @ carry + input_kernel @ input + bias
        )
        return (1.0 - self.leaky_rate) * carry + self.leaky_rate * candidate

    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> Array:
        del key
        return jnp.zeros((self.hidden_size,))


class ESNCell(GRAS):
    """A single ESN cell following the repository's :class:`GRAS` contract."""

    algebra: BinaryAlgebra
    scan: Callable
    input_size: int
    hidden_size: int

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        spectral_radius: float = 0.95,
        leaky_rate: float = 0.2,
        input_scaling: float = 0.1,
        density: float = 0.04,
        activation: Callable[[Array], Array] = jax.nn.tanh,
        trainable: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        self.trainable = bool(trainable)
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.readout_dim = int(hidden_size)
        self.algebra = Resettable(
            ESNSetAction(
                input_size=input_size,
                hidden_size=hidden_size,
                spectral_radius=spectral_radius,
                leaky_rate=leaky_rate,
                input_scaling=input_scaling,
                density=density,
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


class DeepESN(Module):
    """A stack of ESN cells, frozen by default, with no readout layer.

    The returned feature at every timestep is the concatenation of every
    layer's reservoir state and therefore has size ``hidden_size * num_layers``.
    """

    layers: tuple[ESNCell, ...]
    input_size: int
    hidden_size: int
    num_layers: int
    readout_dim: int
    trainable: bool

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        spectral_radius: float = 0.95,
        leaky_rate: float = 0.2,
        input_scaling: float = 0.1,
        density: float = 0.04,
        activation: Callable[[Array], Array] = jax.nn.tanh,
        trainable: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        _validate_sizes(input_size, hidden_size)
        if num_layers < 1:
            raise ValueError("num_layers must be positive")

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.readout_dim = int(hidden_size * num_layers)
        self.trainable = bool(trainable)

        layer_keys = jax.random.split(key, num_layers)
        self.layers = tuple(
            ESNCell(
                input_size=input_size if index == 0 else hidden_size,
                hidden_size=hidden_size,
                spectral_radius=spectral_radius,
                leaky_rate=leaky_rate,
                input_scaling=input_scaling,
                density=density,
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
