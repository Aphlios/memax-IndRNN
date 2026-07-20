"""Utilities for fixed reservoir parameters."""

import jax


def stop_parameter_gradient(parameter):
    """Treat only ``parameter`` as constant during autodiff.

    This is intentionally applied to reservoir parameter leaves, rather than
    to a cell's output. Consequently gradients still propagate through all
    reservoir operations to their inputs and to any trainable encoder before
    the reservoir.
    """

    return jax.tree.map(jax.lax.stop_gradient, parameter)
