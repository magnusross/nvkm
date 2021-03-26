from jax import jit
import jax.numpy as jnp
from functools import partial


@jit
def f(d):
    return jnp.sum(d["a"][0]) ** 2


f({"a": [jnp.array([2.0, 3])]})


@partial(jit, static_argnums=(1,))
def m(x, N):
    return x * jnp.ones(N)


def n(x, N=10):
    return jnp.sum(m(x, N))


n(100.0)

