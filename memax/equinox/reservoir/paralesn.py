"""Parallel Echo State Network recurrent cells.

This adapts ParalESN's complex diagonal reservoirs to memax's GRAS contract.
The temporal recurrence is evaluated with an associative scan. Reservoir and
nonlinear convolutional-mixer arrays are frozen by default and become
trainable with ``trainable=True``; no task-specific readout is included.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, Shaped

from memax.equinox.gras import GRAS
from memax.equinox.groups import BinaryAlgebra, Module, Resettable, Semigroup
from memax.equinox.reservoir._frozen import reservoir_parameter
from memax.equinox.scans import semigroup_scan
from memax.mtypes import Input


@dataclass(frozen=True)
class ReservoirConfig:
    """Configuration for a ParalESN linear reservoir."""

    leaky: float = 1.0
    rho: tuple[float, float] = (0.0, 1.0)
    phase: tuple[float, float] = (0.0, math.tau)
    bias_scaling: float = 1.0


@dataclass(frozen=True)
class MixerConfig:
    """Configuration for the fixed nonlinear convolutional mixer."""

    in_scaling: float = 1.0
    bias_scaling: float = 0.0
    kernel_size: int = 3


def _as_reservoir_config(
    config: Optional[Union[ReservoirConfig, Mapping[str, Any]]],
) -> ReservoirConfig:
    if config is None:
        return ReservoirConfig()
    if isinstance(config, ReservoirConfig):
        return config
    if isinstance(config, Mapping):
        return ReservoirConfig(**dict(config))
    raise TypeError("reservoir_config must be a ReservoirConfig or mapping")


def _as_mixer_config(
    config: Optional[Union[MixerConfig, Mapping[str, Any]]],
) -> MixerConfig:
    if config is None:
        return MixerConfig()
    if isinstance(config, MixerConfig):
        return config
    if isinstance(config, Mapping):
        return MixerConfig(**dict(config))
    raise TypeError("mixer_config must be a MixerConfig or mapping")


def _validate_reservoir_config(config: ReservoirConfig) -> None:
    rho_min, rho_max = config.rho
    phase_min, phase_max = config.phase
    if not 0.0 < config.leaky <= 1.0:
        raise ValueError("reservoir leaky rate must be in (0, 1]")
    if not 0.0 <= rho_min <= rho_max <= 1.0:
        raise ValueError("reservoir rho must satisfy 0 <= min <= max <= 1")
    if phase_min > phase_max:
        raise ValueError("reservoir phase must satisfy min <= max")
    if config.bias_scaling < 0.0:
        raise ValueError("reservoir bias_scaling must be non-negative")


def _validate_mixer_config(config: MixerConfig) -> None:
    if config.in_scaling < 0.0:
        raise ValueError("mixer in_scaling must be non-negative")
    if config.bias_scaling < 0.0:
        raise ValueError("mixer bias_scaling must be non-negative")
    if config.kernel_size < 1:
        raise ValueError("mixer kernel_size must be positive")


def _complex_uniform(
    key: PRNGKeyArray,
    shape: tuple[int, ...],
    minimum: float = -1.0,
    maximum: float = 1.0,
) -> Array:
    real_key, imag_key = jax.random.split(key)
    real = jax.random.uniform(
        real_key, shape, minval=minimum, maxval=maximum, dtype=jnp.float32
    )
    imag = jax.random.uniform(
        imag_key, shape, minval=minimum, maxval=maximum, dtype=jnp.float32
    )
    return (real + 1j * imag).astype(jnp.complex64)


def _init_recurrent_kernel(
    key: PRNGKeyArray, hidden_size: int, config: ReservoirConfig
) -> Array:
    radius_key, phase_key = jax.random.split(key)
    rho_min, rho_max = config.rho
    phase_min, phase_max = config.phase
    squared_radii = jax.random.uniform(
        radius_key,
        (hidden_size,),
        minval=rho_min**2,
        maxval=rho_max**2,
        dtype=jnp.float32,
    )
    radii = jnp.sqrt(squared_radii)
    phases = jax.random.uniform(
        phase_key,
        (hidden_size,),
        minval=phase_min,
        maxval=phase_max,
        dtype=jnp.float32,
    )
    eigenvalues = radii * jnp.exp(1j * phases)
    return ((1.0 - config.leaky) + config.leaky * eigenvalues).astype(
        jnp.complex64
    )


class ParallelReservoirSemigroup(Semigroup):
    """Composition law for diagonal complex affine recurrences."""

    hidden_size: int

    def __init__(self, hidden_size: int):
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        self.hidden_size = int(hidden_size)

    def __call__(
        self, carry: tuple[Array, Array], input: tuple[Array, Array]
    ) -> tuple[Array, Array]:
        transition_i, projection_i = carry
        transition_j, projection_j = input
        return (
            transition_j * transition_i,
            transition_j * projection_i + projection_j,
        )

    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> tuple[Array, Array]:
        del key
        return (
            jnp.ones((self.hidden_size,), dtype=jnp.complex64),
            jnp.zeros((self.hidden_size,), dtype=jnp.complex64),
        )


class ParallelMixer(eqx.Module):
    """Local convolution followed by real ``tanh``, frozen by default."""

    weight: Array
    bias: Array
    kernel_size: int
    trainable: bool

    def __init__(
        self,
        config: Optional[Union[MixerConfig, Mapping[str, Any]]] = None,
        trainable: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        config = _as_mixer_config(config)
        _validate_mixer_config(config)
        weight_key, bias_key = jax.random.split(key)
        weight = _complex_uniform(weight_key, (config.kernel_size,))
        eps = jnp.finfo(jnp.float32).eps
        self.weight = (
            weight
            * jnp.asarray(config.in_scaling, dtype=jnp.float32)
            / jnp.maximum(jnp.linalg.norm(weight), eps)
        )
        self.bias = (
            _complex_uniform(bias_key, (1,))
            * jnp.asarray(config.bias_scaling, dtype=jnp.float32)
        )
        self.kernel_size = int(config.kernel_size)
        self.trainable = bool(trainable)

    def __call__(self, x: Array) -> Array:
        if x.ndim != 1:
            raise ValueError(f"mixer input must have shape (H,); got {x.shape}")
        left_padding = (self.kernel_size - 1) // 2
        right_padding = self.kernel_size - 1 - left_padding
        padded = jnp.pad(x, (left_padding, right_padding))
        indices = jnp.arange(x.shape[-1])[:, None] + jnp.arange(
            self.kernel_size
        )[None, :]
        windows = padded[indices]
        weight = reservoir_parameter(self.weight, self.trainable)
        bias = reservoir_parameter(self.bias, self.trainable)
        return jnp.tanh((windows @ weight + bias).real)


class ParalESNCell(GRAS):
    """One ParalESN cell, frozen by default, using an associative scan."""

    algebra: BinaryAlgebra
    scan: object
    recurrent_kernel: Array
    input_kernel: Optional[Array]
    input_scaling: Optional[Array]
    bias: Array
    mixer: ParallelMixer
    input_size: int
    hidden_size: int
    leaky: float
    trainable: bool

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        reservoir_config: Optional[
            Union[ReservoirConfig, Mapping[str, Any]]
        ] = None,
        mixer_config: Optional[Union[MixerConfig, Mapping[str, Any]]] = None,
        trainable: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        if input_size < 1 or hidden_size < 1:
            raise ValueError("input_size and hidden_size must be positive")
        reservoir_config = _as_reservoir_config(reservoir_config)
        mixer_config = _as_mixer_config(mixer_config)
        _validate_reservoir_config(reservoir_config)
        _validate_mixer_config(mixer_config)

        recurrent_key, input_key, bias_key, mixer_key = jax.random.split(key, 4)
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.readout_dim = int(hidden_size)
        self.leaky = float(reservoir_config.leaky)
        self.trainable = bool(trainable)
        self.recurrent_kernel = _init_recurrent_kernel(
            recurrent_key, hidden_size, reservoir_config
        )

        input_normalization = jnp.sqrt(
            jnp.maximum(0.0, 1.0 - jnp.abs(self.recurrent_kernel) ** 2)
        )
        if input_size == hidden_size:
            self.input_kernel = None
            self.input_scaling = input_normalization.astype(jnp.float32)
        else:
            kernel = _complex_uniform(input_key, (hidden_size, input_size))
            self.input_kernel = kernel * input_normalization[:, None]
            self.input_scaling = None
        self.bias = (
            _complex_uniform(bias_key, (hidden_size,))
            * jnp.asarray(reservoir_config.bias_scaling, dtype=jnp.float32)
        )
        self.mixer = ParallelMixer(
            mixer_config, trainable=trainable, key=mixer_key
        )
        self.algebra = Resettable(ParallelReservoirSemigroup(hidden_size))
        self.scan = semigroup_scan

    def _project_input(self, x: Array) -> Array:
        x = x.astype(jnp.complex64)
        if self.input_kernel is None:
            scaling = reservoir_parameter(self.input_scaling, self.trainable)
            projected = scaling * jnp.roll(x, shift=1, axis=-1)
        else:
            kernel = reservoir_parameter(self.input_kernel, self.trainable)
            projected = kernel @ x
        bias = reservoir_parameter(self.bias, self.trainable)
        return self.leaky * (projected + bias)

    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ):
        del key
        embedding, start = x
        transition = reservoir_parameter(self.recurrent_kernel, self.trainable)
        return (transition, self._project_input(embedding)), start

    def backward_map(
        self,
        h,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Array:
        del x, key
        (_, state), _ = h
        return self.mixer(state)


class ParalESN(Module):
    """A stack of parallel ESN cells, frozen by default, without a readout.

    With ``concat=False``, the last layer is returned. With ``concat=True``,
    ``hidden_size`` is split across layers and all layer outputs are
    concatenated, so the returned feature size remains exactly ``hidden_size``.
    """

    layers: tuple[ParalESNCell, ...]
    layer_sizes: tuple[int, ...]
    input_size: int
    hidden_size: int
    num_layers: int
    concat: bool
    readout_dim: int
    trainable: bool

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        concat: bool = False,
        reservoir_config: Optional[
            Union[ReservoirConfig, Mapping[str, Any]]
        ] = None,
        inter_reservoir_config: Optional[
            Union[ReservoirConfig, Mapping[str, Any]]
        ] = None,
        mixer_config: Optional[Union[MixerConfig, Mapping[str, Any]]] = None,
        inter_mixer_config: Optional[
            Union[MixerConfig, Mapping[str, Any]]
        ] = None,
        trainable: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        if input_size < 1 or hidden_size < 1 or num_layers < 1:
            raise ValueError("input_size, hidden_size, and num_layers must be positive")
        if concat and hidden_size < num_layers:
            raise ValueError("hidden_size must be at least num_layers when concat=True")

        reservoir_config = _as_reservoir_config(reservoir_config)
        mixer_config = _as_mixer_config(mixer_config)
        inter_reservoir_config = (
            reservoir_config
            if inter_reservoir_config is None
            else _as_reservoir_config(inter_reservoir_config)
        )
        inter_mixer_config = (
            mixer_config
            if inter_mixer_config is None
            else _as_mixer_config(inter_mixer_config)
        )
        _validate_reservoir_config(reservoir_config)
        _validate_reservoir_config(inter_reservoir_config)
        _validate_mixer_config(mixer_config)
        _validate_mixer_config(inter_mixer_config)

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.concat = bool(concat)
        self.readout_dim = int(hidden_size)
        self.trainable = bool(trainable)
        if concat:
            later_size = hidden_size // num_layers
            first_size = later_size + hidden_size % num_layers
            self.layer_sizes = (first_size,) + (later_size,) * (num_layers - 1)
        else:
            self.layer_sizes = (hidden_size,) * num_layers

        layer_keys = jax.random.split(key, num_layers)
        layers = []
        layer_input_size = input_size
        for index, layer_size in enumerate(self.layer_sizes):
            layers.append(
                ParalESNCell(
                    input_size=layer_input_size,
                    hidden_size=layer_size,
                    reservoir_config=(
                        reservoir_config if index == 0 else inter_reservoir_config
                    ),
                    mixer_config=mixer_config if index == 0 else inter_mixer_config,
                    trainable=trainable,
                    key=layer_keys[index],
                )
            )
            layer_input_size = layer_size
        self.layers = tuple(layers)

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
        outputs = []
        layer_input = x
        for layer, layer_state, layer_key in zip(self.layers, h, layer_keys):
            next_state, output = layer(layer_state, layer_input, key=layer_key)
            next_states.append(next_state)
            outputs.append(output)
            layer_input = (output, start)
        features = jnp.concatenate(outputs, axis=-1) if self.concat else outputs[-1]
        return tuple(next_states), features

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
