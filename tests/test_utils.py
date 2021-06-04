from nvkm import utils
from jax import vmap
import jax.numpy as jnp


def test_map_reduce():
    ta = jnp.arange(10000).astype(float)
    t1 = vmap(
        lambda t: utils.map_reduce(
            lambda x, y: t ** (2 / 3) + jnp.sin(x) + y ** 2, ta, ta
        )
    )(ta)

    t2 = jnp.sum(
        vmap(
            lambda ti: vmap(lambda xi, yi: ti ** (2 / 3) + jnp.sin(xi) + yi ** 2)(
                ta, ta
            )
        )(ta),
        axis=1,
    )
    assert jnp.all(jnp.isclose(t1, t2))


def test_map_reduce_complex():
    ta = jnp.arange(10000).astype(complex) + jnp.imag(jnp.arange(10000))

    t1 = vmap(
        lambda t: utils.map_reduce(
            lambda x, y: t ** (2 / 3) + jnp.sin(x) + y ** 2, ta, ta
        )
    )(ta)

    t2 = jnp.sum(
        vmap(
            lambda ti: vmap(lambda xi, yi: ti ** (2 / 3) + jnp.sin(xi) + yi ** 2)(
                ta, ta
            )
        )(ta),
        axis=1,
    )
    assert jnp.all(jnp.isclose(t1, t2))

