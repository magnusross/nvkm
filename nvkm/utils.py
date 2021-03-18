from functools import partial
from typing import Callable

import jax.numpy as jnp
from jax import jit, vmap


@partial(jit, static_argnums=(0,))
def map2matrix(
    f: Callable, ts: jnp.DeviceArray, tps: jnp.DeviceArray, *args
) -> jnp.DeviceArray:
    return vmap(lambda ti: vmap(lambda tpi: f(ti, tpi, *args))(tps))(ts)


@jit
def l2p(l: float):
    """
    lengthscale to prescision
    """
    return 0.5 * (1 / l ** 2)


def p2l(p: float):
    """
    precision to lengthscale
    """
    return 1 / jnp.sqrt(2 * p)
