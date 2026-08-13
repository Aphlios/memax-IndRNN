"""Behavioral tests for the Equinox reservoir-computing cells."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from memax.equinox.reservoir import (
    DeepESN,
    ParalESN,
    StructuredESN,
    build_reservoir_model,
)
from memax.equinox.reservoir.paralesn import ParalESNCell
from memax.equinox.reservoir.structured_esn import normalized_hadamard_transform
from memax.equinox.train_utils import (
    build_model,
    build_named_model,
    trainable_filter_spec,
    trainable_parameters,
)


def _models():
    return (
        DeepESN(
            input_size=3,
            hidden_size=8,
            num_layers=2,
            key=jax.random.key(0),
        ),
        StructuredESN(
            input_size=3,
            hidden_size=8,
            num_layers=2,
            key=jax.random.key(1),
        ),
        ParalESN(
            input_size=3,
            hidden_size=8,
            num_layers=2,
            key=jax.random.key(2),
        ),
        ParalESN(
            input_size=3,
            hidden_size=9,
            num_layers=2,
            concat=True,
            key=jax.random.key(3),
        ),
    )


@pytest.mark.parametrize("model", _models())
def test_reservoir_contract_is_jittable(model):
    timesteps = 7
    x = jax.random.normal(jax.random.key(4), (timesteps, model.input_size))
    start = jnp.array([True, False, False, True, False, False, False])

    states, features = eqx.filter_jit(model)(
        model.initialize_carry(), (x, start)
    )

    assert len(states) == model.num_layers
    assert features.shape == (timesteps, model.readout_dim)
    assert jnp.all(jnp.isfinite(features))


@pytest.mark.parametrize("model", _models())
def test_reset_matches_independent_sequences(model):
    x = jax.random.normal(jax.random.key(5), (9, model.input_size))
    packed_start = jnp.array(
        [True, False, False, False, True, False, False, False, False]
    )
    _, packed_features = model(model.initialize_carry(), (x, packed_start))

    _, first_features = model(
        model.initialize_carry(), (x[:4], packed_start[:4])
    )
    _, second_features = model(
        model.initialize_carry(), (x[4:], packed_start[4:])
    )
    independent_features = jnp.concatenate(
        (first_features, second_features), axis=0
    )

    assert jnp.allclose(packed_features, independent_features, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("model", _models())
def test_latest_state_continues_rollout(model):
    x = jax.random.normal(jax.random.key(6), (8, model.input_size))
    start = jnp.array([True, False, False, False, False, False, False, False])
    _, full_features = model(model.initialize_carry(), (x, start))

    first_states, first_features = model(
        model.initialize_carry(), (x[:3], start[:3])
    )
    carry = model.latest_recurrent_state(first_states)
    _, second_features = model(carry, (x[3:], start[3:]))
    chunked_features = jnp.concatenate((first_features, second_features), axis=0)

    assert jnp.allclose(full_features, chunked_features, atol=1e-5, rtol=1e-5)


def test_paralesn_parallel_scan_matches_sequential_recurrence():
    cell = ParalESNCell(input_size=3, hidden_size=7, key=jax.random.key(7))
    x = jax.random.normal(jax.random.key(8), (11, 3))
    start = jnp.array(
        [True, False, False, False, True, False, False, False, False, True, False]
    )
    states, parallel_features = cell(cell.initialize_carry(), (x, start))

    projected = jax.vmap(cell._project_input)(x)
    transition = jax.lax.stop_gradient(cell.recurrent_kernel)

    def step(carry, inputs):
        projection, start_t = inputs
        carry = jnp.where(start_t, jnp.zeros_like(carry), carry)
        next_carry = transition * carry + projection
        return next_carry, next_carry

    _, sequential_states = jax.lax.scan(
        step, jnp.zeros((7,), dtype=jnp.complex64), (projected, start)
    )
    sequential_features = jax.vmap(cell.mixer)(sequential_states)
    (_, parallel_states), _ = states

    assert jnp.allclose(parallel_states, sequential_states, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(parallel_features, sequential_features, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("model", _models())
def test_only_reservoir_parameter_gradients_are_stopped(model):
    x = jax.random.normal(jax.random.key(10), (6, 3))
    start = jnp.array([True, False, False, False, False, False])

    def loss_from_input(input_sequence):
        _, features = model(model.initialize_carry(), (input_sequence, start))
        return jnp.sum(features)

    input_gradient = jax.grad(loss_from_input)(x)
    model_gradient = eqx.filter_grad(
        lambda current_model: jnp.sum(
            current_model(current_model.initialize_carry(), (x, start))[1]
        )
    )(model)
    parameter_gradients = [
        leaf
        for leaf in jax.tree.leaves(model_gradient)
        if eqx.is_inexact_array(leaf)
    ]

    assert jnp.all(jnp.isfinite(input_gradient))
    assert jnp.any(jnp.abs(input_gradient) > 0)
    assert parameter_gradients
    assert all(jnp.all(gradient == 0) for gradient in parameter_gradients)


@pytest.mark.parametrize("model_name", ("DeepESN", "StructuredESN", "ParalESN"))
def test_trainable_reservoir_parameters_receive_gradients_and_updates(model_name):
    model = build_reservoir_model(
        model_name=model_name,
        input_size=3,
        hidden_size=8,
        num_layers=2,
        key=jax.random.key(18),
        model_kwargs={"trainable": True},
    )
    x = jax.random.normal(jax.random.key(19), (6, 3))
    start = jnp.array([True, False, False, False, False, False])

    def loss(current_model):
        _, features = current_model(
            current_model.initialize_carry(), (x, start)
        )
        return jnp.sum(features)

    gradients = eqx.filter_grad(loss)(model)
    gradient_leaves = [
        leaf
        for leaf in jax.tree.leaves(gradients)
        if eqx.is_inexact_array(leaf)
    ]
    updates = jax.tree.map(
        lambda gradient: None if gradient is None else -1e-3 * gradient,
        gradients,
    )
    updated_model = eqx.apply_updates(model, updates)
    parameter_leaves = [
        leaf for leaf in jax.tree.leaves(model) if eqx.is_inexact_array(leaf)
    ]
    updated_leaves = [
        leaf
        for leaf in jax.tree.leaves(updated_model)
        if eqx.is_inexact_array(leaf)
    ]

    assert model.trainable
    assert gradient_leaves
    assert all(jnp.all(jnp.isfinite(gradient)) for gradient in gradient_leaves)
    assert any(jnp.any(gradient != 0) for gradient in gradient_leaves)
    assert any(
        jnp.any(before != after)
        for before, after in zip(parameter_leaves, updated_leaves)
    )


@pytest.mark.parametrize("model_name", ("DeepESN", "StructuredESN", "ParalESN"))
def test_trainable_flag_does_not_change_initial_forward_values(model_name):
    kwargs = dict(
        model_name=model_name,
        input_size=3,
        hidden_size=8,
        num_layers=2,
        key=jax.random.key(20),
    )
    frozen = build_reservoir_model(**kwargs)
    trainable = build_reservoir_model(
        **kwargs, model_kwargs={"trainable": True}
    )
    x = jax.random.normal(jax.random.key(21), (6, 3))
    start = jnp.array([True, False, False, True, False, False])

    _, frozen_features = frozen(frozen.initialize_carry(), (x, start))
    _, trainable_features = trainable(
        trainable.initialize_carry(), (x, start)
    )

    assert not frozen.trainable
    assert trainable.trainable
    assert jnp.allclose(frozen_features, trainable_features)


def test_default_frozen_reservoir_is_excluded_from_adamw():
    model = DeepESN(
        input_size=3,
        hidden_size=8,
        num_layers=2,
        key=jax.random.key(22),
    )
    x = jax.random.normal(jax.random.key(23), (6, 3))
    start = jnp.array([True, False, False, False, False, False])
    loss = lambda current_model: jnp.sum(
        current_model(current_model.initialize_carry(), (x, start))[1]
    )
    gradients = eqx.filter_grad(loss)(model)
    gradients = eqx.filter(gradients, trainable_filter_spec(model))
    optimizer = optax.adamw(1e-3)
    parameters = trainable_parameters(model)
    optimizer_state = optimizer.init(parameters)
    updates, _ = optimizer.update(
        gradients, optimizer_state, params=parameters
    )
    updated_model = eqx.apply_updates(model, updates)

    assert not [leaf for leaf in jax.tree.leaves(parameters) if eqx.is_array(leaf)]
    assert eqx.tree_equal(model, updated_model)


def test_trainable_cnn_receives_gradients_through_reservoir():
    cnn = eqx.nn.Conv1d(
        in_channels=1,
        out_channels=3,
        kernel_size=3,
        padding=1,
        key=jax.random.key(12),
    )
    reservoir = DeepESN(
        input_size=3,
        hidden_size=8,
        num_layers=2,
        key=jax.random.key(13),
    )
    signal = jax.random.normal(jax.random.key(14), (1, 9))
    start = jnp.array([True, False, False, False, False, False, False, False, False])

    def loss(current_cnn):
        encoded = current_cnn(signal).T
        _, features = reservoir(
            reservoir.initialize_carry(), (encoded, start)
        )
        return jnp.sum(features)

    cnn_gradient = eqx.filter_grad(loss)(cnn)
    gradient_leaves = [
        leaf
        for leaf in jax.tree.leaves(cnn_gradient)
        if eqx.is_inexact_array(leaf)
    ]

    assert gradient_leaves
    assert any(jnp.any(jnp.abs(gradient) > 0) for gradient in gradient_leaves)


@pytest.mark.parametrize(
    ("model_name", "model_type", "expected_features"),
    (
        ("DeepESN", DeepESN, 16),
        ("StructuredESN", StructuredESN, 16),
        ("ParalESN", ParalESN, 8),
    ),
)
def test_public_reservoir_builders(model_name, model_type, expected_features):
    direct_model = build_reservoir_model(
        model_name=model_name,
        input_size=3,
        hidden_size=8,
        num_layers=2,
        key=jax.random.key(15),
    )
    named_model = build_named_model(
        model_name=model_name,
        input=3,
        hidden=8,
        num_layers=2,
        key=jax.random.key(16),
    )
    mapped_model = build_model(
        input=3,
        hidden=8,
        num_layers=2,
        models=[model_name],
        key=jax.random.key(17),
    )[model_name]

    assert isinstance(direct_model, model_type)
    assert isinstance(named_model, model_type)
    assert isinstance(mapped_model, model_type)
    assert direct_model.readout_dim == expected_features
    assert not hasattr(direct_model, "readout_layer")


def test_normalized_hadamard_transform_is_orthonormal():
    x = jax.random.normal(jax.random.key(11), (3, 8))
    transformed = normalized_hadamard_transform(x)

    assert jnp.allclose(
        jnp.linalg.norm(transformed, axis=-1),
        jnp.linalg.norm(x, axis=-1),
        atol=1e-6,
        rtol=1e-6,
    )
    assert jnp.allclose(
        normalized_hadamard_transform(transformed), x, atol=1e-6, rtol=1e-6
    )
