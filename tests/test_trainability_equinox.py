"""Unified trainability tests for Equinox memory layers and layer mixers."""

import equinox as eqx
import jax
import jax.numpy as jnp

from memax.equinox.set_actions.gru import GRU
from memax.equinox.train_utils import (
    build_model,
    build_named_model,
    nontrainable_parameters,
    trainable_parameters,
)


MEMORY_MODEL_NAMES = (
    "Identity",
    "MLP",
    "NMax",
    "FART",
    "FWP",
    "DeltaNet",
    "DeltaProduct",
    "GDN",
    "TTTL",
    "TTTL-RoPE",
    "FFM",
    "S6",
    "PSpherical",
    "LRU",
    "LinearRNN",
    "Stack",
    "Attention",
    "Attention-RoPE",
    "Attention-ALiBi",
    "GRU",
    "Elman",
    "ElmanReLU",
    "IndRNN",
    "Spherical",
    "MGU",
    "LSTM",
)


def _array_leaves(tree):
    return [leaf for leaf in jax.tree.leaves(tree) if eqx.is_inexact_array(leaf)]


def test_every_registered_memory_model_can_be_frozen():
    layer_kwargs = {
        name: {"trainable": False} for name in MEMORY_MODEL_NAMES
    }
    models = build_model(
        input=3,
        hidden=8,
        output=4,
        num_layers=1,
        models=MEMORY_MODEL_NAMES,
        layer_kwargs=layer_kwargs,
        key=jax.random.key(0),
    )

    assert set(models) == set(MEMORY_MODEL_NAMES)
    for model in models.values():
        assert all(not layer.trainable for layer in model.layers)
        assert not _array_leaves(trainable_parameters(model).layers)
        assert len(_array_leaves(nontrainable_parameters(model).layers)) == len(
            _array_leaves(model.layers)
        )


def test_frozen_memory_parameters_have_zero_gradient_but_inputs_do_not():
    layer = GRU(recurrent_size=5, trainable=False, key=jax.random.key(1))
    inputs = jax.random.normal(jax.random.key(2), (6, 5))
    starts = jnp.zeros((6,), dtype=bool)

    loss = lambda candidate, x: candidate(
        candidate.initialize_carry(), (x, starts)
    )[1].sum()
    parameter_gradients = eqx.filter_grad(lambda candidate: loss(candidate, inputs))(
        layer
    )
    input_gradients = jax.grad(lambda x: loss(layer, x))(inputs)

    assert _array_leaves(parameter_gradients)
    assert all(
        jnp.all(gradient == 0) for gradient in _array_leaves(parameter_gradients)
    )
    assert jnp.any(input_gradients != 0)


def test_layer_mixers_are_frozen_by_default_and_explicitly_trainable():
    default_model = build_named_model(
        model_name="GRU",
        input=3,
        hidden=8,
        output=4,
        num_layers=2,
        key=jax.random.key(3),
    )
    trainable_mixer_model = build_named_model(
        model_name="GRU",
        input=3,
        hidden=8,
        output=4,
        num_layers=2,
        model_kwargs={"mixer_trainable": True},
        key=jax.random.key(3),
    )

    assert all(not mixer.trainable for mixer in default_model.mixers)
    assert not _array_leaves(trainable_parameters(default_model).mixers)
    assert _array_leaves(nontrainable_parameters(default_model).mixers)

    assert all(mixer.trainable for mixer in trainable_mixer_model.mixers)
    assert _array_leaves(trainable_parameters(trainable_mixer_model).mixers)
    assert not _array_leaves(nontrainable_parameters(trainable_mixer_model).mixers)


def test_parameter_filters_partition_all_inexact_arrays():
    model = build_named_model(
        model_name="S6",
        input=3,
        hidden=8,
        output=4,
        num_layers=2,
        layer_kwargs={"S6": {"trainable": False}},
        model_kwargs={"mixer_trainable": True},
        key=jax.random.key(4),
    )

    all_parameters = _array_leaves(model)
    trainable = _array_leaves(trainable_parameters(model))
    nontrainable = _array_leaves(nontrainable_parameters(model))

    assert len(trainable) + len(nontrainable) == len(all_parameters)
