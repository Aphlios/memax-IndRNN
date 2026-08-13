"""Tests for the memory-free Equinox MLP baseline."""

import equinox as eqx
import jax
import jax.numpy as jnp

from memax.equinox.models.residual import ResidualModel
from memax.equinox.set_actions.mlp import MLP
from memax.equinox.train_utils import build_model, build_named_model


def _gradient_norm(tree):
    return sum(
        jnp.linalg.norm(leaf)
        for leaf in jax.tree.leaves(tree)
        if eqx.is_inexact_array(leaf)
    )


def test_mlp_set_action_scan_matches_independent_timestep_evaluation():
    layer = MLP(recurrent_size=5, key=jax.random.key(0))
    inputs = jax.random.normal(jax.random.key(1), (7, 5))
    starts = jnp.array([True, False, False, True, False, False, False])

    states, outputs = layer(layer.initialize_carry(), (inputs, starts))
    linear = layer.algebra.algebra.linear
    projected = jax.vmap(linear)(inputs)
    expected = projected + jax.nn.silu(projected)

    assert jnp.allclose(outputs, expected)
    assert jnp.allclose(states[0], expected)


def test_mlp_has_no_cross_timestep_memory():
    model = build_named_model(
        model_name="MLP",
        input=3,
        hidden=8,
        output=4,
        num_layers=2,
        key=jax.random.key(2),
    )
    inputs = jax.random.normal(jax.random.key(3), (6, 3))
    changed_inputs = inputs.at[:-1].set(
        jax.random.normal(jax.random.key(4), (5, 3))
    )
    starts = jnp.zeros((6,), dtype=bool)

    _, outputs = model(model.initialize_carry(), (inputs, starts))
    _, changed_outputs = model(
        model.initialize_carry(), (changed_inputs, starts)
    )

    assert jnp.allclose(outputs[-1], changed_outputs[-1])


def test_mlp_is_registered_and_trainable():
    model = build_model(
        input=3,
        hidden=8,
        output=4,
        num_layers=2,
        models=["MLP"],
        key=jax.random.key(5),
    )["MLP"]
    inputs = jax.random.normal(jax.random.key(6), (7, 3))
    starts = jnp.zeros((7,), dtype=bool)
    gradients = eqx.filter_grad(
        lambda candidate: candidate(
            candidate.initialize_carry(), (inputs, starts)
        )[1].sum()
    )(model)

    assert isinstance(model, ResidualModel)
    assert model(model.initialize_carry(), (inputs, starts))[1].shape == (7, 4)
    assert _gradient_norm(gradients.layers) > 0
