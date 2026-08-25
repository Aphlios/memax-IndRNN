"""Initialization and freezing tests for the Equinox LRU."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from memax.equinox.semigroups.lru import LRU
from memax.equinox.train_utils import (
    build_named_model,
    trainable_filter_spec,
    trainable_parameters,
)


def _array_leaves(tree):
    return [leaf for leaf in jax.tree.leaves(tree) if eqx.is_inexact_array(leaf)]


def _loss(model, inputs, starts):
    return model(model.initialize_carry(), (inputs, starts))[1].sum()


def test_lru_initialization_matches_parameterization():
    recurrent_size = 8
    hidden_size = 6
    model = LRU(
        recurrent_size=recurrent_size,
        hidden_size=hidden_size,
        key=jax.random.key(0),
    )
    lambdas = model.diag_lambda()

    assert model.trainable
    assert model.nu_log.shape == (recurrent_size,)
    assert model.theta_log.shape == (recurrent_size,)
    assert model.gamma_log.shape == (recurrent_size,)
    assert model.B_re.shape == (recurrent_size, hidden_size)
    assert model.B_im.shape == (recurrent_size, hidden_size)
    assert model.C_re.shape == (hidden_size, recurrent_size)
    assert model.C_im.shape == (hidden_size, recurrent_size)
    assert model.D.shape == (hidden_size,)
    assert jnp.all(jnp.abs(lambdas) >= model.r_min)
    assert jnp.all(jnp.abs(lambdas) <= model.r_max)
    assert jnp.allclose(
        jnp.exp(model.gamma_log),
        jnp.sqrt(1.0 - jnp.abs(lambdas) ** 2),
    )


def test_lru_trainable_flag_preserves_initial_forward_values():
    kwargs = dict(
        recurrent_size=8,
        hidden_size=6,
        key=jax.random.key(1),
    )
    trainable = LRU(**kwargs)
    frozen = LRU(**kwargs, trainable=False)
    inputs = jax.random.normal(jax.random.key(2), (7, 6))
    starts = jnp.array([True, False, False, True, False, False, False])

    trainable_states, trainable_outputs = trainable(
        trainable.initialize_carry(), (inputs, starts)
    )
    frozen_states, frozen_outputs = frozen(
        frozen.initialize_carry(), (inputs, starts)
    )

    assert not frozen.trainable
    assert eqx.tree_equal(trainable_states, frozen_states)
    assert jnp.allclose(trainable_outputs, frozen_outputs)


def test_lru_parameters_receive_gradients_by_default():
    model = LRU(
        recurrent_size=8,
        hidden_size=6,
        key=jax.random.key(3),
    )
    inputs = jax.random.normal(jax.random.key(4), (7, 6))
    starts = jnp.zeros((7,), dtype=bool)
    gradients = eqx.filter_grad(_loss)(model, inputs, starts)

    assert any(jnp.any(leaf != 0) for leaf in _array_leaves(gradients))


def test_frozen_lru_parameter_gradients_are_zero_but_input_gradient_flows():
    model = LRU(
        recurrent_size=8,
        hidden_size=6,
        trainable=False,
        key=jax.random.key(8),
    )
    inputs = jax.random.normal(jax.random.key(9), (7, 6))
    starts = jnp.zeros((7,), dtype=bool)
    gradients = eqx.filter_grad(_loss)(model, inputs, starts)
    input_gradient = jax.grad(lambda x: _loss(model, x, starts))(inputs)

    assert _array_leaves(gradients)
    assert all(jnp.all(leaf == 0) for leaf in _array_leaves(gradients))
    assert jnp.any(input_gradient != 0)


def test_frozen_lru_and_default_mixers_are_excluded_but_io_maps_stay_trainable():
    model = build_named_model(
        model_name="LRU",
        input=3,
        hidden=8,
        output=4,
        num_layers=2,
        layer_kwargs={"LRU": {"trainable": False}},
        key=jax.random.key(5),
    )
    parameters = trainable_parameters(model)

    assert all(not layer.trainable for layer in model.layers)
    assert not _array_leaves(parameters.layers)
    assert _array_leaves(parameters.map_in)
    assert not _array_leaves(parameters.mixers)
    assert _array_leaves(parameters.map_out)


def test_frozen_lru_is_unchanged_by_adamw_update():
    model = build_named_model(
        model_name="LRU",
        input=3,
        hidden=8,
        output=4,
        num_layers=2,
        layer_kwargs={"LRU": {"trainable": False}},
        key=jax.random.key(6),
    )
    inputs = jax.random.normal(jax.random.key(7), (7, 3))
    starts = jnp.zeros((7,), dtype=bool)
    gradients = eqx.filter_grad(_loss)(model, inputs, starts)
    gradients = eqx.filter(gradients, trainable_filter_spec(model))
    parameters = trainable_parameters(model)
    optimizer = optax.adamw(1e-3)
    optimizer_state = optimizer.init(parameters)
    updates, _ = optimizer.update(gradients, optimizer_state, params=parameters)
    updated = eqx.apply_updates(model, updates)

    assert eqx.tree_equal(model.layers, updated.layers)
    assert eqx.tree_equal(model.mixers, updated.mixers)
    assert not eqx.tree_equal(model.map_out, updated.map_out)
